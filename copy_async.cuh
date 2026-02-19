#pragma once
namespace kernel {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define CP_ASYNC_SM80_ENABLED
#endif

// cp async

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(CP_ASYNC_SM80_ENABLED)
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
#endif
}

__device__ __forceinline__ void cp_async_fence() {
#if defined(CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// load 128 bytes values from global to shared memory async
template <typename T>
__device__ __forceinline__ void load_smem(T *smem_ptr, T const *gmem_ptr) {
#ifdef CP_ASYNC_SM80_ENABLED
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                   smem_int_ptr),
               "l"(gmem_ptr),
               "n"(16),
               "r"(16));
#endif
}

template <typename T>
__device__ __forceinline__ void ldsm_divergence(uint32_t smem_int_ptr,
                                                uint32_t *R) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
}

// ldmatrix
template <typename T>
__device__ __forceinline__ void ldsm(T *__restrict__ smem_ptr, uint32_t *R) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
}
template <typename T>
__device__ __forceinline__ void ldsm_t(T *__restrict__ smem_ptr, uint32_t *R) {
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
}

} // namespace kernel