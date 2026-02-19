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


Kernel compilation:
```
nvcc -o correctness_test correctness_test.cu -O3 -std=c++11 -gencode=arch=compute_80,code=sm_80
nvcc -o speedup_kernel speedup_kernel.cu -O3 -std=c++11 -gencode=arch=compute_80,code=sm_80
```

Speedup & Correctness (compare CUDA timing to ffn.py timing):
```
python ffn.py
./correctness_test
./speedup_kernel
python speedup_compare.py
```

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