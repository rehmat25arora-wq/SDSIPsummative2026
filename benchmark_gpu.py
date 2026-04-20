import time
import os
import glob
from FINd import FINDHasher
from FINd_optimised import FINDHasherOptimised
from FINd_gpu import FINDHasherGPU

path = 'meme_images/0000_12268686.jpg'
N = 50

orig = FINDHasher()
opt  = FINDHasherOptimised()
gpu  = FINDHasherGPU()

print("Warming up Numba CPU...")
opt.fromFile(path)
print("Warming up CUDA GPU...")
gpu.fromFile(path)
print("Done.\n")

# Correctness check
h_orig = str(orig.fromFile(path))
h_opt  = str(opt.fromFile(path))
h_gpu  = str(gpu.fromFile(path))
print(f"Original hash:  {h_orig}")
print(f"Optimised hash: {h_opt}  {'OK' if h_opt == h_orig else 'MISMATCH'}")
print(f"GPU hash:       {h_gpu}  {'OK' if h_gpu == h_orig else 'MISMATCH'}")
print()

def bench(h, n=N):
    t = time.time()
    for _ in range(n): h.fromFile(path)
    return (time.time() - t) / n

t0 = bench(orig)
t1 = bench(opt)
t2 = bench(gpu)

print(f"{'':40} {'Time':>8}  {'Speedup':>8}")
print("-" * 60)
print(f"{'Original (pure Python)':40} {t0*1000:>7.1f}ms  {'1.0x':>8}")
print(f"{'Optimised (numpy + Numba CPU)':40} {t1*1000:>7.1f}ms  {t0/t1:>7.1f}x")
print(f"{'GPU (CUDA via Numba)':40} {t2*1000:>7.1f}ms  {t0/t2:>7.1f}x")
print(f"\nCPU cores: {os.cpu_count()}")
