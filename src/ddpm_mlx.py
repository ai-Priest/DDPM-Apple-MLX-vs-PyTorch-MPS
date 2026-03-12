"""DDPM trained from scratch on CIFAR-10 · Apple MLX 0.31.0 · NHWC layout."""

import argparse
import csv
import math
import os
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — save only, no display
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Load BASE_DIR from .env
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

BASE_DIR    = Path(os.environ.get("BASE_DIR", Path(__file__).parent.parent))
SAMPLES_DIR = BASE_DIR / "samples"
BENCH_DIR   = BASE_DIR / "benchmark_results"
CKPT_DIR    = BASE_DIR / "checkpoints"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters  (override any value in .env)
# ---------------------------------------------------------------------------
T             = int(os.environ.get("T",             1000))
BETA_START    = float(os.environ.get("BETA_START",  1e-4))
BETA_END      = float(os.environ.get("BETA_END",    0.02))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE",    128))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-4))
NUM_EPOCHS    = int(os.environ.get("NUM_EPOCHS",    50))
NUM_WORKERS   = int(os.environ.get("NUM_WORKERS",   4))
EMB_DIM       = 256   # sinusoidal encoding dim  (architectural — keep in code)
TIME_EMB_DIM  = 512   # projected dim per ResBlock (architectural — keep in code)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def get_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0,1] → [-1,1]
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(BASE_DIR / "data"), train=True, download=True, transform=transform,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=NUM_WORKERS, drop_last=True)


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    """Sinusoidal encoding → Linear(256→512) → SiLU → Linear(512→512)."""

    def __init__(self, dim: int = EMB_DIM, time_emb_dim: int = TIME_EMB_DIM):
        super().__init__()
        self.dim     = dim
        self.linear1 = nn.Linear(dim, time_emb_dim)
        self.linear2 = nn.Linear(time_emb_dim, time_emb_dim)

    def __call__(self, t):
        half      = self.dim // 2
        log_scale = math.log(10000) / (half - 1)
        freqs     = mx.exp(-log_scale * mx.arange(half, dtype=mx.float32))
        args      = t.astype(mx.float32)[:, None] * freqs[None, :]
        emb       = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        emb       = nn.silu(self.linear1(emb))
        return self.linear2(emb)  # (B, TIME_EMB_DIM)


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """GroupNorm → SiLU → Conv → timestep add → GroupNorm → SiLU → Conv + residual."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int = TIME_EMB_DIM):
        super().__init__()
        # pytorch_compatible=True is required in MLX 0.31.0 for correct NHWC normalisation.
        self.norm1     = nn.GroupNorm(8, in_ch,  pytorch_compatible=True)
        self.conv1     = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2     = nn.GroupNorm(8, out_ch, pytorch_compatible=True)
        self.conv2     = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.shortcut  = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def __call__(self, x, temb):
        h = self.conv1(nn.silu(self.norm1(x)))
        h = h + self.time_proj(nn.silu(temb))[:, None, None, :]  # broadcast over H, W
        h = self.conv2(nn.silu(self.norm2(h)))
        return (self.shortcut(x) if self.shortcut else x) + h


# ---------------------------------------------------------------------------
# Spatial Self-Attention
# ---------------------------------------------------------------------------
class SpatialSelfAttention(nn.Module):
    """Pre-norm multi-head self-attention; applied at the 8×8 bottleneck only."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels, pytorch_compatible=True)
        self.attn = nn.MultiHeadAttention(dims=channels, num_heads=num_heads)

    def __call__(self, x):
        B, H, W, C = x.shape
        h = self.norm(x).reshape(B, H * W, C)
        h = self.attn(h, h, h).reshape(B, H, W, C)
        return x + h


# ---------------------------------------------------------------------------
# Upsample
# ---------------------------------------------------------------------------
def upsample(x):
    """2× nearest-neighbour upsample in NHWC — avoids ConvTranspose2d checkerboard artefacts."""
    B, H, W, C = x.shape
    x = x.reshape(B, H, 1, W, 1, C)
    x = mx.broadcast_to(x, (B, H, 2, W, 2, C))
    return x.reshape(B, H * 2, W * 2, C)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """
    Noise-prediction network ε_θ(x_t, t).
    Input/output: (B, 32, 32, 3) — NHWC throughout.
    Skip connections concatenated on axis=-1 (channel axis).
    """

    def __init__(self, time_emb_dim: int = TIME_EMB_DIM):
        super().__init__()
        self.time_emb = TimestepEmbedding(EMB_DIM, time_emb_dim)

        # Stem — project 3 RGB channels → 64 before the first GroupNorm.
        # GroupNorm(8, C) requires C % 8 == 0; 3 % 8 != 0 so we can't
        # pass raw pixels directly into a ResBlock.
        self.stem  = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # Encoder
        self.enc1a = ResBlock(64,  64);  self.enc1b = ResBlock(64,  64)
        self.down1 = nn.Conv2d(64,  64,  kernel_size=3, stride=2, padding=1)
        self.enc2a = ResBlock(64,  128); self.enc2b = ResBlock(128, 128)
        self.down2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.enc3a = ResBlock(128, 256); self.enc3b = ResBlock(256, 256)

        # Bottleneck
        self.mid1 = ResBlock(256, 256)
        self.attn = SpatialSelfAttention(256, num_heads=8)
        self.mid2 = ResBlock(256, 256)

        # Decoder — skip3 is at 8×8 so it's concatenated BEFORE the first upsample;
        # subsequent skips are at the matching resolution after each upsample.
        # up_smooth channels match the dec output feeding into each upsample.
        self.dec1a = ResBlock(512, 128); self.dec1b = ResBlock(128, 128)  # 512 = 256(h) + 256(skip3) at 8×8

        self.up_smooth1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec2a = ResBlock(256, 64);  self.dec2b = ResBlock(64, 64)    # 256 = 128(up) + 128(skip2) at 16×16

        self.up_smooth2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec3 = ResBlock(128, 64)                                      # 128 = 64(up) + 64(skip1) at 32×32

        # Output head
        self.out_norm = nn.GroupNorm(8, 64, pytorch_compatible=True)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def __call__(self, x, t):
        temb = self.time_emb(t)

        # Encoder
        h = self.enc1b(self.enc1a(self.stem(x), temb), temb);    skip1 = h
        h = self.enc2b(self.enc2a(self.down1(h), temb), temb); skip2 = h
        h = self.enc3b(self.enc3a(self.down2(h), temb), temb); skip3 = h

        # Bottleneck
        h = self.mid2(self.attn(self.mid1(h, temb)), temb)

        # Decoder
        # skip3 is at 8×8 — concat at bottleneck resolution before first upsample.
        h = self.dec1b(self.dec1a(mx.concatenate([h, skip3], axis=-1), temb), temb)

        # Upsample to 16×16, smooth, then concat skip2 (also 16×16).
        h = mx.concatenate([self.up_smooth1(upsample(h)), skip2], axis=-1)
        h = self.dec2b(self.dec2a(h, temb), temb)

        # Upsample to 32×32, smooth, then concat skip1 (also 32×32).
        h = self.dec3(mx.concatenate([self.up_smooth2(upsample(h)), skip1], axis=-1), temb)

        return self.out_conv(nn.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Gaussian Diffusion
# ---------------------------------------------------------------------------
class GaussianDiffusion:
    """Linear beta schedule · forward corruption · DDPM reverse sampling."""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T             = T
        betas              = mx.linspace(beta_start, beta_end, T)
        alphas             = 1.0 - betas
        alphas_cumprod     = mx.cumprod(alphas)
        self.betas         = betas
        self.alphas        = alphas
        self.alphas_cumprod                = alphas_cumprod
        self.sqrt_alphas_cumprod           = mx.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = mx.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x_0, t, noise):
        """x_t = √ᾱₜ · x_0 + √(1−ᾱₜ) · ε"""
        a  = self.sqrt_alphas_cumprod[t][:, None, None, None]
        b  = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return a * x_0 + b * noise

    def p_losses(self, model, x_0, t):
        """MSE between true noise and predicted noise: L = ||ε − ε_θ(x_t, t)||²"""
        noise = mx.random.normal(x_0.shape)
        return mx.mean((noise - model(self.q_sample(x_0, t, noise), t)) ** 2)

    def p_sample_loop(self, model, shape):
        """Reverse diffusion: x_T ~ N(0,I) → x_0 over T denoising steps."""
        x = mx.random.normal(shape)
        for step in reversed(range(self.T)):
            t_batch   = mx.array([step] * shape[0], dtype=mx.int32)
            beta_t    = self.betas[step]
            alpha_t   = self.alphas[step]
            ab_t      = self.alphas_cumprod[step]
            sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[step]

            mean = (1.0 / mx.sqrt(alpha_t)) * (x - beta_t / sqrt_1mab * model(x, t_batch))

            if step > 0:
                variance = beta_t * (1.0 - self.alphas_cumprod[step - 1]) / (1.0 - ab_t)
                x = mean + mx.sqrt(variance) * mx.random.normal(shape)
            else:
                x = mean

            # Materialise x each step — prevents a 1000-node graph buildup that causes OOM.
            mx.eval(x)

        return x


# ---------------------------------------------------------------------------
# DDIM Sampler (deterministic, η=0) — Song et al., ICLR 2021
# ---------------------------------------------------------------------------
class DDIMSampler:
    """Fast deterministic sampler using a subsequence of T timesteps.

    Uses the same trained weights as DDPM — no retraining required.
    Generates equivalent-quality images in num_steps instead of T steps.
    """

    def __init__(self, diffusion: GaussianDiffusion, num_steps: int = 50):
        self.diffusion = diffusion
        self.num_steps = num_steps
        step_size      = diffusion.T // num_steps
        self.timesteps = list(reversed(range(0, diffusion.T, step_size)))[:num_steps]

    def sample(self, model, shape):
        x = mx.random.normal(shape)
        for i, t in enumerate(self.timesteps):
            t_batch   = mx.array([t] * shape[0], dtype=mx.int32)
            ab_t      = self.diffusion.alphas_cumprod[t]
            sqrt_1mab = self.diffusion.sqrt_one_minus_alphas_cumprod[t]

            eps_pred = model(x, t_batch)

            # Predicted clean image x̂_0 = (x_t − √(1−ᾱₜ)·ε) / √ᾱₜ
            x0_pred = (x - sqrt_1mab * eps_pred) / mx.sqrt(ab_t)
            x0_pred = mx.clip(x0_pred, -1.0, 1.0)  # clamp for stability

            ab_prev = (self.diffusion.alphas_cumprod[self.timesteps[i + 1]]
                       if i < len(self.timesteps) - 1
                       else mx.array(1.0))

            # DDIM deterministic step (η=0): x_{t-1} = √ᾱ_{t-1}·x̂_0 + √(1−ᾱ_{t-1})·ε
            x = mx.sqrt(ab_prev) * x0_pred + mx.sqrt(1.0 - ab_prev) * eps_pred
            mx.eval(x)  # prevent graph buildup

        return x


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------
def save_checkpoint_mlx(model, epoch: int, loss: float, is_best: bool = False):
    flat = {}

    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                flat[key] = v

    _flatten(dict(model.parameters()))
    mx.savez(str(CKPT_DIR / f"epoch_{epoch:03d}.npz"), **flat)
    if is_best:
        mx.savez(str(CKPT_DIR / "best.npz"), **flat)


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------
def save_sample_grid(samples, path: Path, grid_size: int = 4):
    """Denormalise [-1,1] → [0,255] and save a grid_size × grid_size PNG."""
    imgs = ((np.array(samples) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    n    = grid_size
    h, w = imgs.shape[1], imgs.shape[2]
    grid = Image.new("RGB", (n * w, n * h))
    for i, img in enumerate(imgs):
        row, col = divmod(i, n)
        grid.paste(Image.fromarray(img), (col * w, row * h))
    grid.save(str(path))


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------
def save_loss_curve(csv_path: Path, png_path: Path, assets_dir: Path):
    epochs, losses = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["avg_loss"]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, color="#0071e3", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title("DDPM CIFAR-10 — Loss Convergence (MLX · M1 Max)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)

    assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(png_path), str(assets_dir / png_path.name))
    print(f"  → Loss curve saved: {png_path.relative_to(BASE_DIR)}")
    print(f"  → Loss curve copied to assets/")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    print("Loading CIFAR-10 …")
    loader    = get_dataloader()
    model = UNet()
    # MLX uses lazy weight initialisation: Conv2d weights have no concrete shape
    # until the first forward pass triggers evaluation. GroupNorm weights ARE
    # eager (allocated in __init__), but GroupNorm's __call__ reshapes the input
    # tensor — if that input flows through an uninitialised Conv2d first, its shape
    # is empty and group_size = C // num_groups = 0, causing the reshape to crash.
    # A dummy forward pass + mx.eval forces all Conv2d weights to materialise
    # before the training loop begins, eliminating the empty-tensor reshape error.
    mx.eval(model(mx.zeros((1, 32, 32, 3)), mx.zeros((1,), dtype=mx.int32)))
    diffusion = GaussianDiffusion(T=T, beta_start=BETA_START, beta_end=BETA_END)
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)

    value_and_grad_fn = nn.value_and_grad(
        model, lambda m, x, t: diffusion.p_losses(m, x, t)
    )

    # CSV loss log — one row per epoch
    csv_path = BENCH_DIR / "loss_log_mlx.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "avg_loss"])

    total_start  = time.time()
    peak_mem_all = 0.0
    epoch_times  = []
    final_loss   = 0.0
    best_loss    = float("inf")

    for epoch in range(NUM_EPOCHS):
        epoch_start  = time.time()
        epoch_losses = []

        for batch, _ in tqdm(loader, desc=f"Epoch {epoch + 1:03d}", leave=False):
            # torchvision → NCHW; transpose to NHWC for MLX Conv2d.
            x_0  = mx.array(batch.numpy().transpose(0, 2, 3, 1))
            t    = mx.random.randint(0, T, (x_0.shape[0],))

            loss, grads = value_and_grad_fn(model, x_0, t)
            optimizer.update(model, grads)
            # loss must be the first argument — MLX evaluates left-to-right.
            mx.eval(loss, model.parameters(), optimizer.state)
            # .item() breaks the graph reference; omitting this causes OOM.
            epoch_losses.append(loss.item())

        # Read memory after eval — before eval reflects graph metadata only.
        peak_mem     = mx.get_active_memory() / 1024 ** 3
        peak_mem_all = max(peak_mem_all, peak_mem)
        epoch_time   = time.time() - epoch_start
        epoch_times.append(epoch_time)
        final_loss = avg_loss = sum(epoch_losses) / len(epoch_losses)

        print(f"Epoch {epoch + 1:03d} | Time: {epoch_time:.2f}s | "
              f"Loss: {avg_loss:.6f} | Peak Mem: {peak_mem:.2f}GB")

        # CSV append
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_loss:.6f}"])

        # Checkpoint — every 10 epochs and whenever a new best loss is achieved
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        if (epoch + 1) % 10 == 0:
            save_checkpoint_mlx(model, epoch + 1, avg_loss, is_best)
        elif is_best:
            save_checkpoint_mlx(model, epoch + 1, avg_loss, is_best=True)

        if (epoch + 1) % 10 == 0:
            print("  → Generating samples …")
            t0 = time.time()
            if args.sampler == "ddim":
                sampler  = DDIMSampler(diffusion, args.ddim_steps)
                samples  = sampler.sample(model, shape=(16, 32, 32, 3))
                n_steps  = args.ddim_steps
            else:
                samples  = diffusion.p_sample_loop(model, shape=(16, 32, 32, 3))
                n_steps  = T
            mx.eval(samples)
            sample_time = time.time() - t0
            print(f"  → Sampling time (16 images, {n_steps} steps): {sample_time:.2f}s  "
                  f"({n_steps / sample_time:.1f} steps/sec)")
            out_path = SAMPLES_DIR / f"epoch_{epoch + 1:03d}.png"
            save_sample_grid(samples, out_path)
            print(f"  → Saved: {out_path.relative_to(BASE_DIR)}")

    total_time     = time.time() - total_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # Loss curve PNG
    save_loss_curve(csv_path, BENCH_DIR / "loss_curve_mlx.png", BASE_DIR / "assets")

    # DDIM vs DDPM speed comparison for benchmark card
    print("\n  → Timing DDPM sampler (1000 steps, 16 images) …")
    t0 = time.time()
    _s = diffusion.p_sample_loop(model, shape=(16, 32, 32, 3))
    mx.eval(_s)
    ddpm_sample_time = time.time() - t0

    print("  → Timing DDIM sampler (50 steps, 16 images) …")
    t0 = time.time()
    _s = DDIMSampler(diffusion, 50).sample(model, shape=(16, 32, 32, 3))
    mx.eval(_s)
    ddim_sample_time = time.time() - t0
    ddim_speedup     = ddpm_sample_time / ddim_sample_time

    card = (
        "============================================================\n"
        "  MLX DDPM CIFAR-10 — M1 Max Benchmark\n"
        "============================================================\n"
        f"  Epochs:          {NUM_EPOCHS}\n"
        f"  Batch Size:      {BATCH_SIZE}\n"
        f"  T (steps):       {T}\n"
        f"  Total Time:      {total_time:.2f}s\n"
        f"  Avg Epoch Time:  {avg_epoch_time:.2f}s\n"
        f"  Peak Memory:     {peak_mem_all:.3f} GB\n"
        f"  Final Loss:      {final_loss:.6f}\n"
        "------------------------------------------------------------\n"
        f"  DDPM sampling (1000 steps, 16 imgs): {ddpm_sample_time:.2f}s\n"
        f"  DDIM sampling (  50 steps, 16 imgs): {ddim_sample_time:.2f}s\n"
        f"  DDIM speedup:                        {ddim_speedup:.1f}x\n"
        "============================================================"
    )
    print(f"\n{card}")
    bench_path = BENCH_DIR / "benchmark.txt"
    bench_path.write_text(card + "\n")
    print(f"Benchmark saved to: {bench_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM CIFAR-10 training — MLX")
    parser.add_argument("--sampler",    choices=["ddpm", "ddim"], default="ddpm",
                        help="Sampler for periodic sample generation (default: ddpm)")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM steps when --sampler ddim (default: 50)")
    args = parser.parse_args()
    train(args)
