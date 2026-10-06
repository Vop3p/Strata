// include/strata/kernels/tp_exchange.hpp - the two-GPU sum for tensor-parallel decode.
//
// Each rank holds a partial x (n floats, the same n on both).  allreduce() leaves x = x_0 + x_1 on BOTH ranks,
// bitwise the same (a + b == b + a).  No P2P: each rank publishes its partial into fine-grained host memory and
// reads the peer's from there (E232: ~7 us for 10 KB, ~11 us for 40 KB on two RX 6900 XT over PCIe 4.0 x8; RCCL
// does not run on this box, and peer stores into the other card's VRAM are never seen).
//
// Block b of the kernel owns slice b of x and one flag per rank, so no grid-wide sync and no PCIe atomics are
// needed.  The step a block waits for comes from a per-block device counter the kernel itself advances, so the
// launch can be captured into a graph and replayed: both ranks run the same sequence of exchanges, so their
// counters agree.  Buffers alternate by step parity (the faster rank can be one step ahead).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels {

class TpExchange {
public:
    static constexpr int kBlocks = 20;

    TpExchange() = default;
    ~TpExchange();
    TpExchange(const TpExchange&) = delete;
    TpExchange& operator=(const TpExchange&) = delete;

    /// Host-mapped buffers for `max_floats` per rank; per-block counters on each device.
    bool init(int dev0, int dev1, int64_t max_floats, std::string& err);
    /// x (on rank `rank`'s device) += the peer's x.  Every rank must issue the same sequence of calls.
    void allreduce(int rank, float* x, int64_t n, void* stream) const;
    /// #267: every spinning exchange returns (its result is then garbage; the engine is going away).
    void release();
    int device(int rank) const { return dev_[rank]; }

private:
    int dev_[2] = {-1, -1};
    int64_t max_ = 0;
    float* h_buf_[2] = {nullptr, nullptr};       // rank r publishes into h_buf_[r]: [2 parities][max_]
    uint32_t* h_flag_[2] = {nullptr, nullptr};   // rank r raises h_flag_[r][b * 16]
    uint32_t* h_release_ = nullptr;
    float* m_buf_[2][2] = {};                    // [as seen from device d][rank]
    uint32_t* m_flag_[2][2] = {};
    uint32_t* m_release_[2] = {};
    uint32_t* d_step_[2] = {nullptr, nullptr};   // per-block step counters on each device
};

}  // namespace strata::kernels
