#!/usr/bin/env python3
"""
gpu_burn.py — lightweight GPU background load generator (no argparse).

Edit the CONFIG section below to change behavior.
Press Ctrl-C to stop.
"""

import time
import threading
import torch


# ============================================================
# CONFIG
# ============================================================

SIZE = 4096              # Matrix dimension (NxN)
DTYPE = "fp16"           # "fp16", "bf16", or "fp32"
BURST_ITERS = 24         # GEMMs per cycle
SLEEP_MS = 12.0          # Sleep duration per cycle (ms)
WARMUP_ITERS = 5         # Warmup GEMMs before looping
DEVICE = "cuda"          # Typically "cuda"
SECONDS = 0              # 0 = run forever until Ctrl-C
TARGET_UTIL = 0.0        # Set between 0 and 1 to auto-adjust sleep (0 disables)


# ============================================================

def dtype_from_str(s: str):
    s = s.lower()
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    if s == "fp32":
        return torch.float32
    raise ValueError(s)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    dtype = dtype_from_str(DTYPE)

    sleep_ms = SLEEP_MS

    # Heuristic mapping from TARGET_UTIL -> sleep_ms (very approximate)
    if TARGET_UTIL > 0.0:
        u = max(0.05, min(0.95, float(TARGET_UTIL)))
        # baseline: util ~0.50 at sleep_ms=12 (from L40S recipe)
        sleep_ms = 12.0 * (0.50 / u)

    a = torch.randn(SIZE, SIZE, device=DEVICE, dtype=dtype)
    b = torch.randn(SIZE, SIZE, device=DEVICE, dtype=dtype)

    stop = {"flag": False}

    def burn():
        # warmup
        for _ in range(WARMUP_ITERS):
            _ = a @ b
        torch.cuda.synchronize()

        sleep_s = sleep_ms / 1000.0

        while not stop["flag"]:
            for _ in range(BURST_ITERS):
                _ = a @ b
            torch.cuda.synchronize()
            if sleep_s > 0:
                time.sleep(sleep_s)

    th = threading.Thread(target=burn, daemon=True)
    th.start()

    print(
        f"[gpu_burn] started | size={SIZE} dtype={DTYPE} "
        f"burst={BURST_ITERS} sleep_ms={sleep_ms:.2f}"
    )

    try:
        if SECONDS and SECONDS > 0:
            time.sleep(SECONDS)
        else:
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        stop["flag"] = True
        th.join(timeout=5.0)
        print("[gpu_burn] stopped.")


if __name__ == "__main__":
    main()
