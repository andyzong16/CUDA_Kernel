#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "kernel"  

void speedup_kernel(int B, std::ofstream& out) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::mt19937 gen(42 + B);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    // ---------------- HOST (PINNED) ----------------
    __half *h_x, *h_Wu, *h_Wv, *h_Wo, *h_Wuv;

    CHECK_CUDA(cudaMallocHost(&h_x,  B * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_Wu, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_Wv, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_Wo, HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_Wuv, 2 * INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half)));

    // Pack Wuv = [Wu; Wv] stacked by rows (column-major)
    for (int col = 0; col < HIDDEN_SIZE; col++) {
        size_t base_u = (size_t)col * INTERMEDIATE_SIZE;
        size_t base_uv = (size_t)col * (2 * INTERMEDIATE_SIZE);
        for (int row = 0; row < INTERMEDIATE_SIZE; row++) {
            h_Wuv[base_uv + row] = h_Wu[base_u + row];
            h_Wuv[base_uv + row + INTERMEDIATE_SIZE] = h_Wv[base_u + row];
        }
    }

    // ---------------- DEVICE ----------------
    __half *d_x, *d_Wuv, *d_Wo;
    __half *d_UV, *d_output;

    CHECK_CUDA(cudaMalloc(&d_x,  B * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_Wuv, 2 * INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_Wo, HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(__half)));

    CHECK_CUDA(cudaMalloc(&d_UV, 2 * B * INTERMEDIATE_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_output, B * HIDDEN_SIZE * sizeof(__half)));

    // ---------------- TRANSFERS ----------------
    CHECK_CUDA(cudaMemcpy(d_x,  h_x,  B * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wuv, h_Wuv, 2 * INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo, HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(__half), cudaMemcpyHostToDevice));

    const int WARMUP = 15;
    const int ITERS  = 500;

    // ---------------- WARMUP ----------------
    for (int i = 0; i < WARMUP; i++)
        geglu_ffn(handle, d_x, d_Wuv, d_Wo, d_UV, d_output, B);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        geglu_ffn(handle, d_x, d_Wuv, d_Wo, d_UV, d_output, B);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / ITERS;

    out << B << "," << avg_ms << "\n";

    // ---------------- CLEANUP ----------------
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    cudaFree(d_x);
    cudaFree(d_Wuv);
    cudaFree(d_Wo);
    cudaFree(d_UV);
    cudaFree(d_output);
    cudaFreeHost(h_x);
    cudaFreeHost(h_Wu);
    cudaFreeHost(h_Wv);
    cudaFreeHost(h_Wo);
    cudaFreeHost(h_Wuv);

    CHECK_CUBLAS(cublasDestroy(handle));
}

int main() {
    int batch_sizes[] = {4, 8, 16, 32, 64, 128};

    std::ofstream out("cuda_times.csv");
    if (!out) {
        std::cerr << "Failed to write cuda_times.csv" << std::endl;
        return 1;
    }
    out << "B,ms\n";

    for (int B : batch_sizes) {
        speedup_kernel(B, out);
    }

    return 0;
}
