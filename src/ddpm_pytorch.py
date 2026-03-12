"""DDPM trained from scratch on CIFAR-10 · PyTorch MPS · NCHW layout.

Direct translation of ddpm_mlx.py into PyTorch. Key differences from MLX:
  - NCHW (B, C, H, W) throughout — PyTorch Conv2d convention
  - Skip connections concatenated on dim=1 (channel axis), not axis=-1
  - Eager execution — no graph management, no mx.eval()
  - nn.Upsample(scale_factor=2, mode='nearest') replaces the MLX broadcast upsample
"""

import argparse
import csv
import math
import os
import platform
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — save only, no display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Config — reads from .env.pytorch (never touches .env, which belongs to MLX)
# ---------------------------------------------------------------------------
_env_file = Path(__file__).parent.parent / ".env.pytorch"
if _env_file.is_file():
    for _line in _env_file.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

BASE_DIR      = Path(os.environ.get("BASE_DIR", Path(__file__).parent.parent))
SAMPLE_SUBDIR = os.environ.get("SAMPLE_SUBDIR", "pytorch")
BENCH_FILE    = os.environ.get("BENCH_FILE",    "benchmark_pytorch.txt")
MACHINE_LABEL = os.environ.get("MACHINE_LABEL", platform.node().split(".")[0])

SAMPLES_DIR = BASE_DIR / "samples" / SAMPLE_SUBDIR
BENCH_DIR   = BASE_DIR / "benchmark_results"
CKPT_DIR    = BASE_DIR / "checkpoints"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters — override any value in .env.pytorch
# ---------------------------------------------------------------------------
T             = int(os.environ.get("T",              1000))
BETA_START    = float(os.environ.get("BETA_START",   1e-4))
BETA_END      = float(os.environ.get("BETA_END",     0.02))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE",     128))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-4))
NUM_EPOCHS    = int(os.environ.get("NUM_EPOCHS",     50))
NUM_WORKERS   = int(os.environ.get("NUM_WORKERS",    4))
EMB_DIM       = 256   # sinusoidal encoding dim  (architectural constant)
TIME_EMB_DIM  = 512   # projected dim injected per ResBlock

# MPS device — Apple Silicon GPU backend
device = torch.device("mps")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def get_dataloader() -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0,1] → [-1,1]
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(BASE_DIR / "data"), train=True, download=True, transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=False,  # MPS uses unified memory; pin_memory is a CUDA optimisation
    )


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------
class TimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding → Linear(256→512) → SiLU → Linear(512→512)."""

    def __init__(self, dim: int = EMB_DIM, time_emb_dim: int = TIME_EMB_DIM):
        super().__init__()
        self.dim     = dim
        self.linear1 = nn.Linear(dim, time_emb_dim)
        self.linear2 = nn.Linear(time_emb_dim, time_emb_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half      = self.dim // 2
        log_scale = math.log(10000) / (half - 1)
        freqs     = torch.exp(-log_scale * torch.arange(half, dtype=torch.float32, device=t.device))
        args      = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        emb       = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)
        emb       = F.silu(self.linear1(emb))
        return self.linear2(emb)                                   # (B, TIME_EMB_DIM)


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """GroupNorm → SiLU → Conv → timestep add → GroupNorm → SiLU → Conv + residual."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int = TIME_EMB_DIM):
        super().__init__()
        self.norm1     = nn.GroupNorm(8, in_ch)
        self.conv1     = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2     = nn.GroupNorm(8, out_ch)
        self.conv2     = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.shortcut  = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Reshape timestep projection to (B, out_ch, 1, 1) for NCHW broadcast over H, W
        h = h + self.time_proj(F.silu(temb)).reshape(temb.shape[0], -1, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return (self.shortcut(x) if self.shortcut is not None else x) + h


# ---------------------------------------------------------------------------
# Spatial Self-Attention (bottleneck only)
# ---------------------------------------------------------------------------
class SpatialSelfAttention(nn.Module):
    """Pre-norm multi-head self-attention at the 8×8 bottleneck (seq_len = 64).

    Input/output: (B, C, H, W) NCHW. Internally reshaped to (B, H*W, C) for
    nn.MultiheadAttention with batch_first=True.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads,
                                          batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)   # (B, H*W, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """Noise-prediction network ε_θ(x_t, t). Input/output: (B, 3, 32, 32) NCHW.

    Encoder:    32×32 (64ch) → 16×16 (128ch) → 8×8 (256ch)
    Bottleneck: ResBlock → SpatialSelfAttention → ResBlock
    Decoder:    skip connections concatenated on dim=1 at matching resolution
                (concat BEFORE upsampling, not after)
    """

    def __init__(self, time_emb_dim: int = TIME_EMB_DIM):
        super().__init__()
        self.time_emb = TimestepEmbedding(EMB_DIM, time_emb_dim)

        # Stem: 3 → 64 before first GroupNorm (GroupNorm(8,C) requires C % 8 == 0)
        self.stem = nn.Conv2d(3, 64, kernel_size=3, padding=1)

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

        # Decoder — skip3 (256ch, 8×8): concat at bottleneck resolution → 512ch input
        self.dec1a = ResBlock(512, 128); self.dec1b = ResBlock(128, 128)
        self.up1      = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # skip2 (128ch, 16×16): concat after upsample → 256ch input
        self.dec2a = ResBlock(256, 64); self.dec2b = ResBlock(64, 64)
        self.up2      = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # skip1 (64ch, 32×32): concat after upsample → 128ch input
        self.dec3 = ResBlock(128, 64)

        # Output head
        self.out_norm = nn.GroupNorm(8, 64)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_emb(t)                                    # (B, TIME_EMB_DIM)

        # Encoder
        h     = self.stem(x)                                       # (B, 64,  32, 32)
        h     = self.enc1b(self.enc1a(h, temb), temb); skip1 = h  # (B, 64,  32, 32)
        h     = self.down1(h)                                      # (B, 64,  16, 16)
        h     = self.enc2b(self.enc2a(h, temb), temb); skip2 = h  # (B, 128, 16, 16)
        h     = self.down2(h)                                      # (B, 128,  8,  8)
        h     = self.enc3b(self.enc3a(h, temb), temb); skip3 = h  # (B, 256,  8,  8)

        # Bottleneck
        h = self.mid2(self.attn(self.mid1(h, temb)), temb)        # (B, 256,  8,  8)

        # Decoder
        h = torch.cat([h, skip3], dim=1)                          # (B, 512,  8,  8)
        h = self.dec1b(self.dec1a(h, temb), temb)                 # (B, 128,  8,  8)
        h = self.up_conv1(self.up1(h))                            # (B, 128, 16, 16)
        h = torch.cat([h, skip2], dim=1)                          # (B, 256, 16, 16)
        h = self.dec2b(self.dec2a(h, temb), temb)                 # (B, 64,  16, 16)
        h = self.up_conv2(self.up2(h))                            # (B, 64,  32, 32)
        h = torch.cat([h, skip1], dim=1)                          # (B, 128, 32, 32)
        h = self.dec3(h, temb)                                    # (B, 64,  32, 32)

        return self.out_conv(F.silu(self.out_norm(h)))            # (B,  3,  32, 32)


# ---------------------------------------------------------------------------
# Gaussian Diffusion
# ---------------------------------------------------------------------------
class GaussianDiffusion:
    """Linear beta schedule · forward corruption · DDPM reverse sampling."""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        # Precompute schedule on MPS so t-batch indexing stays device-local
        betas          = torch.linspace(beta_start, beta_end, T, device=device)
        alphas         = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas                         = betas
        self.alphas                        = alphas
        self.alphas_cumprod                = alphas_cumprod
        self.sqrt_alphas_cumprod           = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """x_t = √ᾱₜ · x_0 + √(1−ᾱₜ) · ε"""
        a = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return a * x_0 + b * noise

    def p_losses(self, model: nn.Module, x_0: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        """MSE between true noise and predicted noise: L = ||ε − ε_θ(x_t, t)||²"""
        noise = torch.randn_like(x_0)
        return F.mse_loss(noise, model(self.q_sample(x_0, t, noise), t))

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: tuple) -> torch.Tensor:
        """Reverse diffusion: x_T ~ N(0,I) → x_0 over T denoising steps."""
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.T)):
            t_batch   = torch.full((shape[0],), step, dtype=torch.long, device=device)
            beta_t    = self.betas[step]
            alpha_t   = self.alphas[step]
            ab_t      = self.alphas_cumprod[step]
            sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[step]

            mean = (1.0 / torch.sqrt(alpha_t)) * (x - beta_t / sqrt_1mab * model(x, t_batch))
            if step > 0:
                variance = beta_t * (1.0 - self.alphas_cumprod[step - 1]) / (1.0 - ab_t)
                x = mean + torch.sqrt(variance) * torch.randn_like(x)
            else:
                x = mean
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

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for i, t in enumerate(self.timesteps):
            t_batch   = torch.full((shape[0],), t, dtype=torch.long, device=device)
            ab_t      = self.diffusion.alphas_cumprod[t]
            sqrt_1mab = self.diffusion.sqrt_one_minus_alphas_cumprod[t]

            eps_pred = model(x, t_batch)

            # Predicted clean image x̂_0 = (x_t − √(1−ᾱₜ)·ε) / √ᾱₜ
            x0_pred = (x - sqrt_1mab * eps_pred) / torch.sqrt(ab_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)  # clamp for stability

            if i < len(self.timesteps) - 1:
                ab_prev = self.diffusion.alphas_cumprod[self.timesteps[i + 1]]
            else:
                ab_prev = torch.tensor(1.0, device=device)

            # DDIM deterministic step (η=0): x_{t-1} = √ᾱ_{t-1}·x̂_0 + √(1−ᾱ_{t-1})·ε
            x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps_pred

        return x


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------
def save_checkpoint_pytorch(model: nn.Module, epoch: int, loss: float,
                             is_best: bool = False) -> None:
    torch.save(model.state_dict(), str(CKPT_DIR / f"epoch_{epoch:03d}.pt"))
    if is_best:
        torch.save(model.state_dict(), str(CKPT_DIR / "best.pt"))


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------
def save_sample_grid(samples: torch.Tensor, path: Path, grid_size: int = 4) -> None:
    """Denormalise NCHW samples [-1,1] → [0,255] and save a grid_size×grid_size PNG."""
    imgs = samples.cpu().permute(0, 2, 3, 1).numpy()
    imgs = ((imgs + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
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
def save_loss_curve(csv_path: Path, png_path: Path, assets_dir: Path) -> None:
    epochs, losses = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["avg_loss"]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, color="#ee4b2b", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title(f"DDPM CIFAR-10 — Loss Convergence (PyTorch MPS · {MACHINE_LABEL})")
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
def train(args) -> None:
    print("Loading CIFAR-10 …")
    loader = get_dataloader()

    model = UNet().to(device)
    # Dummy forward pass: triggers MPS kernel compilation before timing begins
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32, device=device),
              torch.zeros(1, dtype=torch.long, device=device))
    print("Model initialised.")

    diffusion = GaussianDiffusion(T=T, beta_start=BETA_START, beta_end=BETA_END)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # CSV loss log — one row per epoch
    csv_path = BENCH_DIR / "loss_log_pytorch.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "avg_loss"])

    total_start  = time.time()
    peak_mem_all = 0.0
    epoch_times  = []
    final_loss   = 0.0
    best_loss    = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        torch.mps.synchronize()  # MPS is async — sync before wall-clock start
        epoch_start  = time.time()
        epoch_losses = []

        for batch, _ in tqdm(loader, desc=f"Epoch {epoch + 1:03d}", leave=False):
            x_0 = batch.to(device)
            t   = torch.randint(0, T, (x_0.shape[0],), device=device)
            optimizer.zero_grad()
            loss = diffusion.p_losses(model, x_0, t)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        torch.mps.synchronize()
        epoch_time    = time.time() - epoch_start
        # driver_allocated_memory: full Metal driver pool (caching allocator)
        # current_allocated_memory: live tensors only — fairer cross-framework comparison
        peak_mem      = torch.mps.driver_allocated_memory() / 1024 ** 3
        peak_mem_live = torch.mps.current_allocated_memory() / 1024 ** 3
        peak_mem_all  = max(peak_mem_all, peak_mem)
        epoch_times.append(epoch_time)
        final_loss = avg_loss = sum(epoch_losses) / len(epoch_losses)

        print(f"Epoch {epoch + 1:03d} | Time: {epoch_time:.2f}s | Loss: {avg_loss:.6f} | "
              f"Mem(driver): {peak_mem:.2f}GB | Mem(live): {peak_mem_live:.2f}GB")

        # CSV append
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_loss:.6f}"])

        # Checkpoint — every 10 epochs and whenever a new best loss is achieved
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        if (epoch + 1) % 10 == 0:
            save_checkpoint_pytorch(model, epoch + 1, avg_loss, is_best)
        elif is_best:
            save_checkpoint_pytorch(model, epoch + 1, avg_loss, is_best=True)

        if (epoch + 1) % 10 == 0:
            print("  → Generating samples …")
            model.eval()
            torch.mps.synchronize()
            t0 = time.time()
            if args.sampler == "ddim":
                sampler  = DDIMSampler(diffusion, args.ddim_steps)
                samples  = sampler.sample(model, shape=(16, 3, 32, 32))
                n_steps  = args.ddim_steps
            else:
                samples  = diffusion.p_sample_loop(model, shape=(16, 3, 32, 32))
                n_steps  = T
            torch.mps.synchronize()
            sample_time = time.time() - t0
            print(f"  → Sampling time (16 images, {n_steps} steps): {sample_time:.2f}s  "
                  f"({n_steps / sample_time:.1f} steps/sec)")
            out_path = SAMPLES_DIR / f"epoch_{epoch + 1:03d}.png"
            save_sample_grid(samples, out_path)
            print(f"  → Saved: {out_path.relative_to(BASE_DIR)}")

    torch.mps.synchronize()
    total_time     = time.time() - total_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # Loss curve PNG
    save_loss_curve(csv_path, BENCH_DIR / "loss_curve_pytorch.png", BASE_DIR / "assets")

    # DDIM vs DDPM speed comparison for benchmark card
    model.eval()
    print("\n  → Timing DDPM sampler (1000 steps, 16 images) …")
    torch.mps.synchronize()
    t0 = time.time()
    diffusion.p_sample_loop(model, shape=(16, 3, 32, 32))
    torch.mps.synchronize()
    ddpm_sample_time = time.time() - t0

    print("  → Timing DDIM sampler (50 steps, 16 images) …")
    torch.mps.synchronize()
    t0 = time.time()
    DDIMSampler(diffusion, 50).sample(model, shape=(16, 3, 32, 32))
    torch.mps.synchronize()
    ddim_sample_time = time.time() - t0
    ddim_speedup     = ddpm_sample_time / ddim_sample_time

    card = (
        "============================================================\n"
        f"  PyTorch MPS DDPM CIFAR-10 — {MACHINE_LABEL} Benchmark\n"
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
    bench_path = BENCH_DIR / BENCH_FILE
    bench_path.write_text(card + "\n")
    print(f"Benchmark saved to: {bench_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM CIFAR-10 training — PyTorch MPS")
    parser.add_argument("--sampler",    choices=["ddpm", "ddim"], default="ddpm",
                        help="Sampler for periodic sample generation (default: ddpm)")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM steps when --sampler ddim (default: 50)")
    args = parser.parse_args()
    train(args)
