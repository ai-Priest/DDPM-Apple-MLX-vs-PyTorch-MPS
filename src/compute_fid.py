"""Compute FID score between generated and real CIFAR-10 images.

Uses pytorch-fid for Fréchet Inception Distance computation.
Works with MLX-generated images (default) or PyTorch-generated images.
Note: pytorch-fid uses PyTorch internally regardless of --framework.

Usage:
    python src/compute_fid.py
    python src/compute_fid.py --framework mlx --n_images 1000 --device mps
    python src/compute_fid.py --framework pytorch --n_images 1000 --device cpu

N=1000 is the minimum for a stable FID estimate. N=5000–10000 gives more
reliable results. Default is 1000 for speed.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Parse arguments before conditional imports
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compute FID score")
parser.add_argument("--framework",   choices=["mlx", "pytorch"], default="mlx",
                    help="Framework used to generate images (default: mlx)")
parser.add_argument("--n_images",    type=int, default=1000,
                    help="Number of images to generate (default: 1000; min recommended)")
parser.add_argument("--batch_size",  type=int, default=16,
                    help="Batch size for generation (default: 16)")
parser.add_argument("--device",      default="mps",
                    help="Device for FID Inception v3 computation: mps or cpu (default: mps)")
parser.add_argument("--checkpoint",  default="checkpoints/best.npz",
                    help="MLX checkpoint path (default: checkpoints/best.npz)")
parser.add_argument("--checkpoint_pt", default="checkpoints/best.pt",
                    help="PyTorch checkpoint path (default: checkpoints/best.pt)")
parser.add_argument("--keep_dirs",   action="store_true",
                    help="Keep fid_generated/ and fid_real/ after scoring")
args = parser.parse_args()

import sys
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Framework-specific imports
# ---------------------------------------------------------------------------
if args.framework == "mlx":
    import mlx.core as mx
    from ddpm_mlx import (
        UNet, GaussianDiffusion,
        BASE_DIR, BENCH_DIR, CKPT_DIR, T, BETA_START, BETA_END,
    )

    def load_model():
        ckpt = BASE_DIR / args.checkpoint
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        model = UNet()
        mx.eval(model(mx.zeros((1, 32, 32, 3)), mx.zeros((1,), dtype=mx.int32)))
        raw  = mx.load(str(ckpt))
        flat = {k: mx.array(v) for k, v in raw.items()}

        def _unflatten(d):
            result = {}
            for key, value in d.items():
                parts = key.split(".")
                node = result
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = value
            return result

        model.update(_unflatten(flat))
        mx.eval(model.parameters())
        return model

    def generate_batch(model, diffusion, batch_size):
        """Returns numpy array (B, H, W, 3) in [0, 255] uint8."""
        samples = diffusion.p_sample_loop(model, shape=(batch_size, 32, 32, 3))
        mx.eval(samples)
        imgs = ((np.array(samples) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return imgs  # (B, H, W, C)

else:  # pytorch
    from ddpm_pytorch import (
        UNet, GaussianDiffusion, device,
        BASE_DIR, BENCH_DIR, CKPT_DIR, T, BETA_START, BETA_END,
    )

    def load_model():
        ckpt = BASE_DIR / args.checkpoint_pt
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        model = UNet().to(device)
        model.load_state_dict(torch.load(str(ckpt), map_location=device))
        model.eval()
        return model

    def generate_batch(model, diffusion, batch_size):
        """Returns numpy array (B, H, W, 3) in [0, 255] uint8."""
        samples = diffusion.p_sample_loop(model, shape=(batch_size, 3, 32, 32))
        imgs = samples.cpu().permute(0, 2, 3, 1).numpy()
        imgs = ((imgs + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return imgs  # (B, H, W, C)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_images_to_dir(imgs: np.ndarray, directory: Path, offset: int = 0) -> None:
    """Save (B, H, W, C) uint8 numpy array as individual PNGs."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        Image.fromarray(img).save(str(directory / f"{offset + i:06d}.png"))


def save_real_cifar10(out_dir: Path, n_images: int) -> None:
    """Save n_images CIFAR-10 test images as individual PNGs (no normalisation)."""
    print(f"  Preparing {n_images} real CIFAR-10 test images …")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = torchvision.datasets.CIFAR10(
        root=str(BASE_DIR / "data"), train=False, download=True,
        transform=transforms.ToTensor(),  # [0,1]
    )
    saved = 0
    for img_tensor, _ in dataset:
        if saved >= n_images:
            break
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(str(out_dir / f"{saved:06d}.png"))
        saved += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    gen_dir  = BASE_DIR / "fid_generated"
    real_dir = BASE_DIR / "fid_real"

    # --- Generate images ---
    print(f"Loading model …")
    model     = load_model()
    diffusion = GaussianDiffusion(T=T, beta_start=BETA_START, beta_end=BETA_END)

    print(f"Generating {args.n_images} images in batches of {args.batch_size} …")
    gen_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    while generated < args.n_images:
        batch = min(args.batch_size, args.n_images - generated)
        imgs  = generate_batch(model, diffusion, batch)
        save_images_to_dir(imgs, gen_dir, offset=generated)
        generated += batch
        print(f"  {generated}/{args.n_images}", end="\r")
    print()

    # --- Save real images ---
    save_real_cifar10(real_dir, args.n_images)

    # --- Compute FID ---
    try:
        from pytorch_fid import fid_score
    except ImportError:
        raise ImportError(
            "pytorch-fid not installed. Run: pip install pytorch-fid>=0.3.0"
        )

    print(f"\nComputing FID (device={args.device}) …")
    try:
        fid = fid_score.calculate_fid_given_paths(
            [str(real_dir), str(gen_dir)],
            batch_size=64,
            device=args.device,
            dims=2048,
        )
    except Exception as e:
        if args.device != "cpu":
            print(f"  MPS error ({e}), retrying on CPU …")
            fid = fid_score.calculate_fid_given_paths(
                [str(real_dir), str(gen_dir)],
                batch_size=64,
                device="cpu",
                dims=2048,
            )
        else:
            raise

    print(f"\nFID Score: {fid:.2f}")
    print(f"(N={args.n_images}, framework={args.framework}, device={args.device})")
    print("Note: N=1,000 is sufficient for comparison; publication-grade results use N≥10,000.")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    score_path = BENCH_DIR / "fid_score.txt"
    score_path.write_text(
        f"FID Score: {fid:.2f}\n"
        f"N={args.n_images} · framework={args.framework} · device={args.device}\n"
    )
    print(f"Saved: {score_path}")

    # --- Cleanup ---
    if not args.keep_dirs:
        shutil.rmtree(str(gen_dir))
        shutil.rmtree(str(real_dir))
        print("Cleaned up temp image directories.")


if __name__ == "__main__":
    main()
