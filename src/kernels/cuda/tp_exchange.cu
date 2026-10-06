// src/kernels/cuda/tp_exchange.cu - see include/strata/kernels/tp_exchange.hpp.
#include "strata/kernels/tp_exchange.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int kFlagStride = 16;   // one flag per 64-byte line

// x += peer's x for this block's slice.  `step` is per block and advanced here, so a captured launch replays.
__global__ void __launch_bounds__(256) tp_allreduce_kernel(float* __restrict__ x, int n, float* mine,
                                                           volatile uint32_t* my_flags, const float* theirs,
                                                           const volatile uint32_t* their_flags,
                                                           const volatile uint32_t* release, uint32_t* step,
                                                           int stride) {
    const int b = blockIdx.x, per = (n + gridDim.x - 1) / gridDim.x, lo = b * per, hi = min(n, lo + per);
    const uint32_t s = step[b] + 1;
    const size_t par = (size_t) (s & 1u) * (size_t) stride;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) mine[par + i] = x[i];
    __threadfence_system();   // every thread's own stores, before the flag
    __syncthreads();
    if (threadIdx.x == 0) {
        my_flags[b * kFlagStride] = s;
        __threadfence_system();
        while (their_flags[b * kFlagStride] < s && *release == 0u) strata_spin_pause();
        __threadfence_system();
        step[b] = s;
    }
    __syncthreads();
    const volatile float* t = theirs + par;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) x[i] += t[i];
}

bool ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("tp exchange: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

bool host_alloc(void** p, size_t bytes) {
#if defined(STRATA_USE_HIP)
    // fine-grained (coherent): a spinning kernel must see the other card's stores, not a cached line
    return hipHostMalloc(p, bytes, hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent) == hipSuccess;
#else
    return cudaHostAlloc(p, bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess;
#endif
}

}  // namespace

bool TpExchange::init(int dev0, int dev1, int64_t max_floats, std::string& err) {
    dev_[0] = dev0;
    dev_[1] = dev1;
    max_ = max_floats;
    int prev = 0;
    cudaGetDevice(&prev);
    for (int r = 0; r < 2; ++r) {
        if (!host_alloc((void**) &h_buf_[r], (size_t) 2 * (size_t) max_ * sizeof(float)) ||
            !host_alloc((void**) &h_flag_[r], (size_t) kBlocks * kFlagStride * sizeof(uint32_t))) {
            err = "tp exchange: the pinned exchange buffers do not fit";
            return false;
        }
        for (int i = 0; i < kBlocks * kFlagStride; ++i) ((volatile uint32_t*) h_flag_[r])[i] = 0;
    }
    if (!host_alloc((void**) &h_release_, 64)) { err = "tp exchange: the release word does not fit"; return false; }
    *(volatile uint32_t*) h_release_ = 0;
    for (int d = 0; d < 2; ++d) {
        if (!ok(cudaSetDevice(dev_[d]), "set device", err)) return false;
        for (int r = 0; r < 2; ++r) {
            if (!ok(cudaHostGetDevicePointer((void**) &m_buf_[d][r], h_buf_[r], 0), "map buffer", err) ||
                !ok(cudaHostGetDevicePointer((void**) &m_flag_[d][r], h_flag_[r], 0), "map flags", err))
                return false;
        }
        if (!ok(cudaHostGetDevicePointer((void**) &m_release_[d], h_release_, 0), "map release", err) ||
            !ok(cudaMalloc((void**) &d_step_[d], kBlocks * sizeof(uint32_t)), "step counters", err) ||
            !ok(cudaMemset(d_step_[d], 0, kBlocks * sizeof(uint32_t)), "step counters", err))
            return false;
    }
    cudaSetDevice(prev);
    return true;
}

TpExchange::~TpExchange() {
    release();
    int prev = 0;
    cudaGetDevice(&prev);
    for (int d = 0; d < 2; ++d)
        if (d_step_[d]) { cudaSetDevice(dev_[d]); cudaDeviceSynchronize(); cudaFree(d_step_[d]); }
    cudaSetDevice(prev);
    for (int r = 0; r < 2; ++r) {
        if (h_buf_[r]) cudaFreeHost(h_buf_[r]);
        if (h_flag_[r]) cudaFreeHost(h_flag_[r]);
    }
    if (h_release_) cudaFreeHost(h_release_);
}

void TpExchange::allreduce(int rank, float* x, int64_t n, void* stream) const {
    if (n <= 0) return;
    if (n > max_) { std::fprintf(stderr, "tp exchange: %lld floats > capacity %lld\n", (long long) n, (long long) max_); std::abort(); }
    const int me = rank, peer = rank ^ 1;
    tp_allreduce_kernel<<<kBlocks, 256, 0, (cudaStream_t) stream>>>(x, (int) n, m_buf_[me][me], m_flag_[me][me],
                                                                    m_buf_[me][peer], m_flag_[me][peer],
                                                                    m_release_[me], d_step_[me], (int) max_);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "tp exchange: %s\n", cudaGetErrorString(e)); std::abort(); }
}

void TpExchange::release() {
    if (h_release_) *(volatile uint32_t*) h_release_ = 1u;
}

}  // namespace strata::kernels
