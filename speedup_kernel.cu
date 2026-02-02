#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "kernel"

void speedup_kernel(int B, std::ofstream& out) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::mt19937 gen(42 + B);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    std::vector<float> h_x(B * HIDDEN_SIZE);
    std::vector<float> h_Wu(INTERMEDIATE_SIZE * HIDDEN_SIZE);
    std::vector<float> h_Wv(INTERMEDIATE_SIZE * HIDDEN_SIZE);
    std::vector<float> h_Wo(HIDDEN_SIZE * INTERMEDIATE_SIZE);

    for (auto& v : h_x)  v = dist(gen);
    for (auto& v : h_Wu) v = dist(gen);
    for (auto& v : h_Wv) v = dist(gen);
    for (auto& v : h_Wo) v = dist(gen);

    // Convert FP32 to FP16
    std::vector<__half> h_x_fp16(h_x.size());
    std::vector<__half> h_Wu_fp16(h_Wu.size());
    std::vector<__half> h_Wv_fp16(h_Wv.size());
    std::vector<__half> h_Wo_fp16(h_Wo.size());

    for (size_t i = 0; i < h_x.size(); i++) h_x_fp16[i] = __float2half(h_x[i]);
    for (size_t i = 0; i < h_Wu.size(); i++) h_Wu_fp16[i] = __float2half(h_Wu[i]);
    for (size_t i = 0; i < h_Wv.size(); i++) h_Wv_fp16[i] = __float2half(h_Wv[i]);
    for (size_t i = 0; i < h_Wo.size(); i++) h_Wo_fp16[i] = __float2half(h_Wo[i]);

    __half *d_x, *d_Wu, *d_Wv, *d_Wo;
    __half *d_U, *d_V, *d_intermediate, *d_output;

    CHECK_CUDA(cudaMalloc(&d_x, B * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_Wu, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_Wv, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_Wo, HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(__half)));

    CHECK_CUDA(cudaMalloc(&d_U, B * INTERMEDIATE_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_V, B * INTERMEDIATE_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_intermediate, B * INTERMEDIATE_SIZE * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_output, B * HIDDEN_SIZE * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_x,  h_x_fp16.data(),
                          B * HIDDEN_SIZE * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wu, h_Wu_fp16.data(),
                          INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv_fp16.data(),
                          INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo_fp16.data(),
                          HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(__half),
                          cudaMemcpyHostToDevice));

    const int WARMUP = 10;
    const int ITERS  = 100;

    // ------------------------------------------------------------
    // Warmup
    // ------------------------------------------------------------
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

    // ------------------------------------------------------------
    // Timing
    // ------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();
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
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    float ms =
        std::chrono::duration<float, std::milli>(end - start).count() / ITERS;

    std::cout << "Batch " << B << ": " << ms << " ms" << std::endl;
    out << B << "," << ms << "\n";

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_Wu));
    CHECK_CUDA(cudaFree(d_Wv));
    CHECK_CUDA(cudaFree(d_Wo));
    CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_V));
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

    std::cout << "### CUDA cuBLAS + GEGLU KERNEL TIMINGS ###" << std::endl;
    for (int B : batch_sizes) {
        speedup_kernel(B, out);
    }

    return 0;
}
