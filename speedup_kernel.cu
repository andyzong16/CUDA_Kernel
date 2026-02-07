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
    __half *h_x, *h_Wu, *h_Wv, *h_Wo;

    CHECK_CUDA(cudaMallocHost(&h_x,  B * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_Wu, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_Wv, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMallocHost(&h_Wo, HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(__half)));

    // Fill random
    for (int i = 0; i < B * HIDDEN_SIZE; i++)
        h_x[i] = __float2half(dist(gen));

    for (int i = 0; i < INTERMEDIATE_SIZE * HIDDEN_SIZE; i++) {
        h_Wu[i] = __float2half(dist(gen));
        h_Wv[i] = __float2half(dist(gen));
    }

    for (int i = 0; i < HIDDEN_SIZE * INTERMEDIATE_SIZE; i++)
        h_Wo[i] = __float2half(dist(gen));

    // ---------------- DEVICE ----------------
    __half *d_x, *d_Wu, *d_Wv, *d_Wo;
    __half *d_U, *d_V, *d_intermediate, *d_output;

    CHECK_CUDA(cudaMalloc(&d_x,  B * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_Wu, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_Wv, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_Wo, HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(__half)));

    CHECK_CUDA(cudaMalloc(&d_U,  B * INTERMEDIATE_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_V,  B * INTERMEDIATE_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_intermediate, B * INTERMEDIATE_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_output,       B * HIDDEN_SIZE * sizeof(__half)));

    // ---------------- TRANSFERS ----------------
    CHECK_CUDA(cudaMemcpy(d_x,  h_x,  B * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wu, h_Wu, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo, HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(__half), cudaMemcpyHostToDevice));

    const int WARMUP = 15;
    const int ITERS  = 500;

    // ---------------- WARMUP ----------------
    for (int i = 0; i < WARMUP; i++) {
        geglu_ffn(
            handle,
            d_x,
            d_Wu,
            d_Wv,
            d_Wo,
            d_U,
            d_V,
            d_intermediate,
            d_output,
            B
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---------------- TIMING ----------------
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) {
        geglu_ffn(
            handle,
            d_x,
            d_Wu,
            d_Wv,
            d_Wo,
            d_U,
            d_V,
            d_intermediate,
            d_output,
            B
        );
    }
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
    cudaFree(d_Wu);
    cudaFree(d_Wv);
    cudaFree(d_Wo);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_intermediate);
    cudaFree(d_output);

    cudaFreeHost(h_x);
    cudaFreeHost(h_Wu);
    cudaFreeHost(h_Wv);
    cudaFreeHost(h_Wo);

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
