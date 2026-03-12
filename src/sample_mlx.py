"""Generate images from a saved MLX checkpoint.

Usage:
    python src/sample_mlx.py
    python src/sample_mlx.py --checkpoint checkpoints/best.npz --n_samples 16
"""

import argparse
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Reuse model architecture and config from the training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ddpm_mlx import (
    UNet, GaussianDiffusion, DDIMSampler,
    BASE_DIR, SAMPLES_DIR, CKPT_DIR, T, BETA_START, BETA_END,
)
from ddpm_mlx import save_sample_grid


def load_checkpoint(model: UNet, checkpoint_path: Path) -> None:
    """Load flat .npz checkpoint back into the model via nested dict update."""
    raw = mx.load(str(checkpoint_path))
    # Convert to mx.array and unflatten dot-separated keys to nested dict
    flat = {k: mx.array(v) for k, v in raw.items()}

    def _unflatten(flat_dict: dict) -> dict:
        result: dict = {}
        for key, value in flat_dict.items():
            parts = key.split(".")
            d = result
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        return result

    model.update(_unflatten(flat))
    mx.eval(model.parameters())


def main(args) -> None:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = BASE_DIR / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = UNet()
    # Dummy pass to materialise lazy weights before loading
    mx.eval(model(mx.zeros((1, 32, 32, 3)), mx.zeros((1,), dtype=mx.int32)))
    load_checkpoint(model, checkpoint_path)

    diffusion = GaussianDiffusion(T=T, beta_start=BETA_START, beta_end=BETA_END)

    n = args.n_samples
    shape = (n, 32, 32, 3)  # NHWC

    print(f"Sampling {n} images ({args.sampler.upper()}, "
          f"{args.ddim_steps if args.sampler == 'ddim' else T} steps) …")
    t0 = time.time()
    if args.sampler == "ddim":
        samples = DDIMSampler(diffusion, args.ddim_steps).sample(model, shape)
    else:
        samples = diffusion.p_sample_loop(model, shape)
    mx.eval(samples)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s")

    # Grid size: largest integer n such that n*n <= n_samples
    import math
    grid_size = math.isqrt(n)
    samples_for_grid = samples[:grid_size * grid_size]

    timestamp = int(time.time())
    out_path  = SAMPLES_DIR / f"sampled_{timestamp}.png"
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    save_sample_grid(samples_for_grid, out_path, grid_size=grid_size)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from a saved MLX DDPM checkpoint")
    parser.add_argument("--checkpoint",  default="checkpoints/best.npz",
                        help="Path to checkpoint .npz (default: checkpoints/best.npz)")
    parser.add_argument("--n_samples",   type=int, default=16,
                        help="Number of images to generate (default: 16)")
    parser.add_argument("--sampler",     choices=["ddpm", "ddim"], default="ddpm",
                        help="Sampler to use (default: ddpm)")
    parser.add_argument("--ddim_steps",  type=int, default=50,
                        help="DDIM steps when --sampler ddim (default: 50)")
    args = parser.parse_args()
    main(args)
