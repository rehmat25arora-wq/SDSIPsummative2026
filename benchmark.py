import time
import os
from FINd import FINDHasher
from FINd_optimised import FINDHasherOptimised

path = 'meme_images/0000_12268686.jpg'
orig = FINDHasher()
opt = FINDHasherOptimised()

print("Warming up Numba...")
opt.fromFile(path)
print("Done.\n")

t1 = time.time()
for _ in range(30): orig.fromFile(path)
orig_time = (time.time() - t1) / 30

t2 = time.time()
for _ in range(30): opt.fromFile(path)
opt_time = (time.time() - t2) / 30

print(f"Original:  {orig_time*1000:.1f}ms")
print(f"Optimised: {opt_time*1000:.1f}ms")
print(f"Speedup:   {orig_time/opt_time:.1f}x")
print(f"CPU cores: {os.cpu_count()}")
