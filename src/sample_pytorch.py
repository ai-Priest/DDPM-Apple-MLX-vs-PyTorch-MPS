"""Generate images from a saved PyTorch MPS checkpoint.

Usage:
    python src/sample_pytorch.py
    python src/sample_pytorch.py --checkpoint checkpoints/best.pt --n_samples 16
"""

import argparse
import math
import time
from pathlib import Path

import torch

# Reuse model architecture and config from the training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ddpm_pytorch import (
    UNet, GaussianDiffusion, DDIMSampler, device,
    BASE_DIR, SAMPLES_DIR, CKPT_DIR, T, BETA_START, BETA_END,
)
from ddpm_pytorch import save_sample_grid


def main(args) -> None:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = BASE_DIR / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = UNet().to(device)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = GaussianDiffusion(T=T, beta_start=BETA_START, beta_end=BETA_END)

    n = args.n_samples
    shape = (n, 3, 32, 32)  # NCHW

    print(f"Sampling {n} images ({args.sampler.upper()}, "
          f"{args.ddim_steps if args.sampler == 'ddim' else T} steps) …")
    torch.mps.synchronize()
    t0 = time.time()
    if args.sampler == "ddim":
        samples = DDIMSampler(diffusion, args.ddim_steps).sample(model, shape)
    else:
        samples = diffusion.p_sample_loop(model, shape)
    torch.mps.synchronize()
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s")

    # Grid size: largest integer g such that g*g <= n_samples
    grid_size = math.isqrt(n)
    samples_for_grid = samples[:grid_size * grid_size]

    timestamp = int(time.time())
    out_path  = SAMPLES_DIR / f"sampled_{timestamp}.png"
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    save_sample_grid(samples_for_grid, out_path, grid_size=grid_size)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a saved PyTorch MPS DDPM checkpoint")
    parser.add_argument("--checkpoint",  default="checkpoints/best.pt",
                        help="Path to checkpoint .pt (default: checkpoints/best.pt)")
    parser.add_argument("--n_samples",   type=int, default=16,
                        help="Number of images to generate (default: 16)")
    parser.add_argument("--sampler",     choices=["ddpm", "ddim"], default="ddpm",
                        help="Sampler to use (default: ddpm)")
    parser.add_argument("--ddim_steps",  type=int, default=50,
                        help="DDIM steps when --sampler ddim (default: 50)")
    args = parser.parse_args()
    main(args)
