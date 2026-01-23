import csv
import os

def read_times(path):
    times = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times[int(row["B"])] = float(row["ms"])
    return times

py_path = os.path.join("speedup_data", "py_times.csv")
cuda_path = os.path.join("speedup_data", "cuda_times.csv")

if not os.path.exists(py_path):
    raise FileNotFoundError(f"Missing {py_path}. Run ffn.py first.")
if not os.path.exists(cuda_path):
    raise FileNotFoundError(f"Missing {cuda_path}. Run speedup_kernel first.")

py_times = read_times(py_path)
cuda_times = read_times(cuda_path)

print("### SPEEDUP (PyTorch / CUDA kernel) ###")
for B in sorted(py_times.keys()):
    if B not in cuda_times:
        continue
    speedup = py_times[B] / cuda_times[B]
    print(f"B={B:>3}: {speedup:.3f}x (py={py_times[B]:.3f} ms, cuda={cuda_times[B]:.3f} ms)")
