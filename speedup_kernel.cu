#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// Reuse the optimized kernel implementation.
#include "kernel"

void benchmark(int B, std::ofstream& out) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::mt19937 gen(42 + B);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    std::vector<float> h_x(B * HIDDEN_SIZE);
    std::vector<float> h_Wu(INTERMEDIATE_SIZE * HIDDEN_SIZE);
    std::vector<float> h_Wv(INTERMEDIATE_SIZE * HIDDEN_SIZE);
    std::vector<float> h_Wo(HIDDEN_SIZE * INTERMEDIATE_SIZE);

    for (auto& v : h_x) v = dist(gen);
    for (auto& v : h_Wu) v = dist(gen);
    for (auto& v : h_Wv) v = dist(gen);
    for (auto& v : h_Wo) v = dist(gen);

    float *d_x, *d_Wu, *d_Wv, *d_Wo, *d_intermediate, *d_output;

    CHECK_CUDA(cudaMalloc(&d_x, B * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wu, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo, HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_intermediate, B * INTERMEDIATE_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, B * HIDDEN_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), B * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wu, h_Wu.data(), INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv.data(), INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo.data(), HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    const int WARMUP = 10;
    const int ITERS = 100;

    for (int i = 0; i < WARMUP; i++) {
        geglu_ffn_tiled(handle, d_x, d_Wu, d_Wv, d_Wo, d_intermediate, d_output, B);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        geglu_ffn_tiled(handle, d_x, d_Wu, d_Wv, d_Wo, d_intermediate, d_output, B);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(end - start).count() / ITERS;
    std::cout << "Batch " << B << ": " << ms << " ms" << std::endl;
    out << B << "," << ms << "\n";

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_Wu));
    CHECK_CUDA(cudaFree(d_Wv));
    CHECK_CUDA(cudaFree(d_Wo));
    CHECK_CUDA(cudaFree(d_intermediate));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUBLAS(cublasDestroy(handle));
}

int main() {
    int batch_sizes[] = {4, 8, 16, 32, 64, 128};

    std::ofstream out("speedup_data/cuda_times.csv");
    if (!out) {
        std::cerr << "Failed to write speedup_data/cuda_times.csv" << std::endl;
        return 1;
    }
    out << "B,ms\n";

    std::cout << "### CUDA KERNEL TIMINGS ###" << std::endl;
    for (int B : batch_sizes) {
        benchmark(B, out);
    }

    return 0;
}
