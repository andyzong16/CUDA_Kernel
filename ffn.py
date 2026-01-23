import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class GEGLU_FFN(nn.Module):
    def __init__(self, hidden_size=4096, intermediate_size=12288):
        super().__init__()
        self.Wu = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.Wv = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.Wo = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        u = self.Wu(x)      # (B, 12288)
        v = self.Wv(x)      # (B, 12288)
        g = F.gelu(u)
        h = g * v
        return self.Wo(h)   # (B, 4096)

device = "cuda"
print("Device:", device)

ffn = GEGLU_FFN().to(device)
batch_sizes = [4, 8, 16, 32, 64, 128]

data_dir = "correctness_data"
os.makedirs(data_dir, exist_ok=True)
speedup_dir = "speedup_data"
os.makedirs(speedup_dir, exist_ok=True)

def write_bin(path, tensor):
    tensor.detach().to("cpu").contiguous().numpy().astype("float32").tofile(path)

def read_bin(path, shape):
    import numpy as np
    return np.fromfile(path, dtype=np.float32).reshape(shape)

torch.manual_seed(42)

# Export weights once for CUDA correctness test
write_bin(os.path.join(data_dir, "Wu.bin"), ffn.Wu.weight)
write_bin(os.path.join(data_dir, "Wv.bin"), ffn.Wv.weight)
write_bin(os.path.join(data_dir, "Wo.bin"), ffn.Wo.weight)

# Warm-up
py_times = []
for B in batch_sizes:
    x = torch.randn(B, 4096, device=device)
    for _ in range(5):
        _ = ffn(x)

for B in batch_sizes:
    x = torch.randn(B, 4096, device=device)

    # GPU sync before timing
    torch.cuda.synchronize()
    start = time.perf_counter()

    y = ffn(x)
    
    torch.cuda.synchronize()
    end = time.perf_counter()

    ms = (end - start) * 1000
    py_times.append((B, ms))
    print(f"Batch {B:>3}: {ms:.3f} ms")

    # Export inputs and outputs for CUDA correctness test
    write_bin(os.path.join(data_dir, f"x_B{B}.bin"), x)
    write_bin(os.path.join(data_dir, f"out_py_B{B}.bin"), y)

    cuda_out_path = os.path.join(data_dir, f"out_cuda_B{B}.bin")
    if os.path.exists(cuda_out_path):
        import numpy as np
        out_cuda = read_bin(cuda_out_path, (B, 4096))
        out_py = y.detach().cpu().numpy().astype("float32")
        diff = np.abs(out_py - out_cuda)
        print(
            f"  diff max={diff.max():.6f}, avg={diff.mean():.6f} "
            f"(compared to {cuda_out_path})"
        )

with open(os.path.join(speedup_dir, "py_times.csv"), "w") as f:
    f.write("B,ms\n")
    for B, ms in py_times:
        f.write(f"{B},{ms}\n")
