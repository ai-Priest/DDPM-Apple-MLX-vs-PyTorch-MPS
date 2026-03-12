"""Standalone sampling speed benchmark — MLX or PyTorch MPS.

Loads the best checkpoint, runs p_sample_loop 3 times, reports mean ± std.
Appends a benchmark card to benchmark_results/benchmark_sampling.txt.

Usage:
    python src/benchmark_sampling.py --framework mlx
    python src/benchmark_sampling.py --framework pytorch
    python src/benchmark_sampling.py --framework mlx --n_images 16 --ddim_steps 50
"""

import argparse
import statistics
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse --framework before importing framework-specific modules
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Sampling speed benchmark")
parser.add_argument("--framework",  choices=["mlx", "pytorch"], default="mlx",
                    help="Framework to benchmark (default: mlx)")
parser.add_argument("--n_images",   type=int, default=16,
                    help="Number of images per run (default: 16)")
parser.add_argument("--sampler",    choices=["ddpm", "ddim"], default="ddpm",
                    help="Sampler to use (default: ddpm)")
parser.add_argument("--ddim_steps", type=int, default=50,
                    help="DDIM steps when --sampler ddim (default: 50)")
parser.add_argument("--n_runs",     type=int, default=3,
                    help="Number of timing runs (default: 3)")
args = parser.parse_args()

import sys
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Framework-specific imports and setup
# ---------------------------------------------------------------------------
if args.framework == "mlx":
    import mlx.core as mx
    from ddpm_mlx import (
        UNet, GaussianDiffusion, DDIMSampler,
        BASE_DIR, BENCH_DIR, CKPT_DIR, T, BETA_START, BETA_END,
    )

    def load_model():
        ckpt = CKPT_DIR / "best.npz"
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt}. Run training first.")
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

    def run_sampler(model, diffusion, n):
        shape = (n, 32, 32, 3)
        if args.sampler == "ddim":
            s = DDIMSampler(diffusion, args.ddim_steps).sample(model, shape)
        else:
            s = diffusion.p_sample_loop(model, shape)
        mx.eval(s)

    def timed_run(model, diffusion, n):
        t0 = time.time()
        run_sampler(model, diffusion, n)
        return time.time() - t0

    label   = "MLX"
    version = mx.__version__

else:  # pytorch
    import torch
    from ddpm_pytorch import (
        UNet, GaussianDiffusion, DDIMSampler, device,
        BASE_DIR, BENCH_DIR, CKPT_DIR, T, BETA_START, BETA_END, MACHINE_LABEL,
    )

    def load_model():
        ckpt = CKPT_DIR / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt}. Run training first.")
        model = UNet().to(device)
        model.load_state_dict(torch.load(str(ckpt), map_location=device))
        model.eval()
        return model

    def run_sampler(model, diffusion, n):
        shape = (n, 3, 32, 32)
        if args.sampler == "ddim":
            DDIMSampler(diffusion, args.ddim_steps).sample(model, shape)
        else:
            diffusion.p_sample_loop(model, shape)

    def timed_run(model, diffusion, n):
        torch.mps.synchronize()
        t0 = time.time()
        run_sampler(model, diffusion, n)
        torch.mps.synchronize()
        return time.time() - t0

    label   = f"PyTorch MPS · {MACHINE_LABEL}"
    version = torch.__version__


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def main():
    print(f"Loading best checkpoint …")
    model     = load_model()
    diffusion = GaussianDiffusion(T=T, beta_start=BETA_START, beta_end=BETA_END)
    n_steps   = args.ddim_steps if args.sampler == "ddim" else T

    print(f"Framework:  {label}")
    print(f"Sampler:    {args.sampler.upper()} ({n_steps} steps)")
    print(f"Images:     {args.n_images} per run")
    print(f"Runs:       {args.n_runs}")
    print()

    # Warm-up (not timed)
    print("Warm-up run …")
    run_sampler(model, diffusion, args.n_images)

    times = []
    for i in range(args.n_runs):
        t = timed_run(model, diffusion, args.n_images)
        times.append(t)
        print(f"  Run {i + 1}: {t:.2f}s")

    mean_t = statistics.mean(times)
    std_t  = statistics.stdev(times) if len(times) > 1 else 0.0
    steps_per_sec = n_steps / mean_t

    card = (
        "============================================================\n"
        f"  Sampling Benchmark — {label}\n"
        "============================================================\n"
        f"  Framework:       {label} {version}\n"
        f"  Sampler:         {args.sampler.upper()}\n"
        f"  Images:          {args.n_images}\n"
        f"  T (steps):       {n_steps}\n"
    )
    for i, t in enumerate(times, 1):
        card += f"  Run {i}:           {t:.2f}s\n"
    card += (
        f"  Mean:            {mean_t:.2f} ± {std_t:.2f}s\n"
        f"  Time/image:      {mean_t / args.n_images:.2f}s\n"
        f"  Steps/sec:       {steps_per_sec:.1f}\n"
        "============================================================"
    )
    print(f"\n{card}")

    out_path = BENCH_DIR / "benchmark_sampling.txt"
    with open(out_path, "a") as f:
        f.write(card + "\n\n")
    print(f"\nAppended to: {out_path}")


if __name__ == "__main__":
    main()
