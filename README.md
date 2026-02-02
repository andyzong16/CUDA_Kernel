Implementation includes:
Optimized tiled kernel: Fused computation with improved arithmetic intensity through thread-level tiling
Correctness test: Compares CUDA output against ffn.py output
Speedup comparison: Compares CUDA kernel timing against ffn.py timing

Architecture Details:
Hidden size: 4096
Intermediate size: 12288 (3x expansion ratio)
Precision: FP32
Optimization technique: Thread-tiling to increase FLOPs/byte ratio and reduce memory bandwidth pressure

Requirements:
CUDA Toolkit (tested with CUDA 11.0+)
NVIDIA GPU with compute capability 6.0 or higher
C++11 compatible compiler
cuBLAS library


Kernel compilation:
```
nvcc -o correctness_test correctness_test.cu -lcublas -O3 -std=c++11
nvcc -o speedup_kernel speedup_kernel.cu -lcublas -O3 -std=c++11
```

With architecture-specific optimization (recommended):
For Ampere architecture (A6000, RTX 6000, RTX 30xx):
```
nvcc -o speedup_kernel speedup_kernel.cu -lcublas -O3 -std=c++11 \
  -gencode=arch=compute_75,code=sm_75 \
  -gencode=arch=compute_75,code=compute_75

nvcc -o correctness_test correctness_test.cu -lcublas -O3 -std=c++11 \
  -gencode=arch=compute_75,code=sm_75 \
  -gencode=arch=compute_75,code=compute_75
```
For Ada architecture (RTX 6000 Ada, RTX 40xx):
```
nvcc -o correctness_test correctness_test.cu -lcublas -O3 -std=c++11 -arch=sm_89
nvcc -o speedup_kernel speedup_kernel.cu -lcublas -O3 -std=c++11 -arch=sm_89
```

-lcublas: Links the cuBLAS library for optimized matrix operations
-O3: Maximum compiler optimization level
-std=c++11: C++11 standard required for chrono and random libraries
-arch=sm_XX: Specifies GPU compute capability for architecture-specific optimizations

Speedup & Correctness (compare CUDA timing to ffn.py timing):
```
python ffn.py
./correctness_test
./speedup_kernel
python speedup_compare.py
```

Expected output structure:
### CUDA OUTPUT DUMP ###
=== Testing B=4 ===
Wrote CUDA output: correctness_data/out_cuda_B4.bin
...

### SPEEDUP (PyTorch / CUDA kernel) ###
B=  4: <ratio>x (py=<ms> ms, cuda=<ms> ms)

The test suite compares CUDA output against ffn.py output

Batch sizes tested: 4, 8, 16, 32, 64, 128
Validation metrics: Max and average absolute difference (reported by ffn.py)
Random seed: Fixed seed (42 + batch_size) for reproducibility

Speedup Benchmarks
Each batch size undergoes timing in both Python and CUDA:

Warmup iterations (CUDA): 10
Benchmark iterations (CUDA): 100
Metrics reported:
Average execution time per iteration (ms)
Speedup ratio (PyTorch / CUDA)

Ensure CUDA toolkit is properly installed:
```
nvcc --version
```
Verify cuBLAS is available:
```
ls /usr/local/cuda/lib64/libcublas.so
```

Code Structure
kernel
├── GELU activation function (device)
├── Optimized implementation
│   ├── geglu_tiled_kernel (fused with tiling)
│   └── geglu_ffn_tiled (orchestration)

correctness_test.cu
└── Runs CUDA kernel and writes correctness_data/out_cuda_B*.bin

ffn.py
├── Runs PyTorch GEGLU FFN
├── Writes correctness_data inputs/outputs
└── Computes diffs if CUDA outputs exist

speedup_kernel.cu
└── Writes speedup_data/cuda_times.csv

speedup_compare.py
└── Reads speedup_data/*.csv and prints speedup