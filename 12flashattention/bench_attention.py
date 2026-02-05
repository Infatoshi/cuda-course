import argparse
import time

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark math attention vs flash attention (SDPA).")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--causal", action="store_true")
    return parser.parse_args()


def to_dtype(dtype: str):
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def run_one(name, q, k, v, backend):
    enable_flash = backend == "flash"
    enable_math = backend == "math"
    enable_mem_efficient = False

    times_ms = []
    with torch.backends.cuda.sdp_kernel(
        enable_flash=enable_flash,
        enable_math=enable_math,
        enable_mem_efficient=enable_mem_efficient,
    ):
        for _ in range(ARGS.warmup):
            _ = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=ARGS.causal)
        torch.cuda.synchronize()

        for _ in range(ARGS.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=ARGS.causal)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

    avg_ms = sum(times_ms) / len(times_ms)
    print(f"{name}: {avg_ms:.3f} ms (batch={ARGS.batch}, heads={ARGS.heads}, seq={ARGS.seq}, dim={ARGS.dim}, dtype={ARGS.dtype})")


if __name__ == "__main__":
    ARGS = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    device = "cuda"
    dtype = to_dtype(ARGS.dtype)

    q = torch.randn(ARGS.batch, ARGS.heads, ARGS.seq, ARGS.dim, device=device, dtype=dtype)
    k = torch.randn(ARGS.batch, ARGS.heads, ARGS.seq, ARGS.dim, device=device, dtype=dtype)
    v = torch.randn(ARGS.batch, ARGS.heads, ARGS.seq, ARGS.dim, device=device, dtype=dtype)

    print("SDPA support:")
    print(torch.backends.cuda.sdp_kernel)
    print("Flash available:", torch.backends.cuda.flash_sdp_enabled())
    print("Math available:", torch.backends.cuda.math_sdp_enabled())

    run_one("math", q, k, v, backend="math")
    if torch.backends.cuda.flash_sdp_enabled():
        run_one("flash", q, k, v, backend="flash")
    else:
        print("flash backend not available on this build/runtime")
