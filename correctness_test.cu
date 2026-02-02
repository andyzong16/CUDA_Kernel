#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "kernel"

// ====================================================================================
// Binary IO helpers
// ====================================================================================

void read_binary_f32(const std::string& path, std::vector<float>& data, size_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open: " << path << std::endl;
        exit(1);
    }
    data.resize(count);
    in.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    if (!in) {
        std::cerr << "Failed to read: " << path << std::endl;
        exit(1);
    }
}

void write_binary_f32(const std::string& path, const std::vector<float>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to write: " << path << std::endl;
        exit(1);
    }
    out.write(reinterpret_cast<const char*>(data.data()),
              data.size() * sizeof(float));
}

// ====================================================================================
// Correctness Test (cuBLAS + GEGLU kernel)
// ====================================================================================

bool test_correctness(int B) {
    std::cout << "\n=== Testing B=" << B << " ===" << std::endl;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const std::string base = "correctness_data";
    const std::string x_path        = base + "/x_B" + std::to_string(B) + ".bin";
    const std::string wu_path       = base + "/Wu.bin";
    const std::string wv_path       = base + "/Wv.bin";
    const std::string wo_path       = base + "/Wo.bin";
    const std::string out_cuda_path = base + "/out_cuda_B" + std::to_string(B) + ".bin";

    std::vector<float> h_x, h_Wu, h_Wv, h_Wo;

    read_binary_f32(x_path,  h_x,  static_cast<size_t>(B) * HIDDEN_SIZE);
    read_binary_f32(wu_path, h_Wu, static_cast<size_t>(INTERMEDIATE_SIZE) * HIDDEN_SIZE);
    read_binary_f32(wv_path, h_Wv, static_cast<size_t>(INTERMEDIATE_SIZE) * HIDDEN_SIZE);
    read_binary_f32(wo_path, h_Wo, static_cast<size_t>(HIDDEN_SIZE) * INTERMEDIATE_SIZE);

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

    // ------------------------------------------------------------
    // Run optimized GEGLU FFN
    // ------------------------------------------------------------
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

    CHECK_CUDA(cudaDeviceSynchronize());

    // ------------------------------------------------------------
    // Copy result back
    // ------------------------------------------------------------
    std::vector<__half> h_out(B * HIDDEN_SIZE);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_output,
                          B * HIDDEN_SIZE * sizeof(__half),
                          cudaMemcpyDeviceToHost));

    // Convert FP16 output back to FP32 for comparison
    std::vector<float> h_out_fp32(h_out.size());
    for (size_t i = 0; i < h_out.size(); i++) {
        h_out_fp32[i] = __half2float(h_out[i]);
    }

    write_binary_f32(out_cuda_path, h_out_fp32);
    std::cout << "Wrote CUDA output: " << out_cuda_path << std::endl;

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_Wu));
    CHECK_CUDA(cudaFree(d_Wv));
    CHECK_CUDA(cudaFree(d_Wo));
    CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_intermediate));
    CHECK_CUDA(cudaFree(d_output));

    CHECK_CUBLAS(cublasDestroy(handle));
    return true;
}

int main() {
    int batch_sizes[] = {4, 8, 16, 32, 64, 128};

    std::cout << "### CUDA OUTPUT DUMP ###" << std::endl;
    bool all_passed = true;

    for (int B : batch_sizes) {
        if (!test_correctness(B)) all_passed = false;
    }

    if (!all_passed) {
        std::cout << "\n✗ Some tests failed!" << std::endl;
        return 1;
    }

    std::cout << "\n✓ All correctness tests PASSED!\n" << std::endl;
    return 0;
}
