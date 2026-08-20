/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: CANN Open Software License Agreement Version 2.0
 */

#include <atomic>
#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/sinkhorn_normalize_tiling.h"

#define SN_TILE_TARGET (SN_TILE_TARGET)

extern "C" void sinkhorn_normalize_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                           uint8_t *x, uint8_t *y, uint8_t *tiling);

namespace ascend_kernel {

namespace {

std::atomic<uint64_t> g_callCount{0};

int64_t QueryVectorCoreNum()
{
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t n = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &n);
    TORCH_CHECK(ret == ACL_SUCCESS && n > 0, "failed to get NPU vector core count");
    return n;
}

inline int64_t GetVectorCoreNum()
{
    static const int64_t cached = QueryVectorCoreNum();
    return cached;
}

struct TilingCache {
    bool valid = false;
    uint32_t totalMats = 0;
    uint32_t repeat = 0;
    float eps = 0.0f;
    SinkhornTilingData meta{};
    void *dev = nullptr;   
};

TilingCache &GetTilingCache()
{
    static TilingCache c;
    return c;
}

void FillTiling(SinkhornTilingData &t, uint32_t totalMats, uint32_t repeat, float eps)
{
    SinkhornComputeTiling(t, totalMats, repeat, eps,
                          static_cast<uint32_t>(GetVectorCoreNum()), SN_TILE_TARGET);
}

} // namespace

uint64_t sinkhorn_normalize_call_count()
{
    return g_callCount.load(std::memory_order_relaxed);
}

at::Tensor sinkhorn_normalize_torch(const at::Tensor& x, int64_t repeat, double eps)
{
    TORCH_CHECK(x.scalar_type() == at::kFloat, "only FP32 supported");
    TORCH_CHECK(x.is_privateuseone(), "x must be on NPU");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims (..., mhc, mhc)");
    TORCH_CHECK(x.size(-1) == SN_MHC && x.size(-2) == SN_MHC,
        "last two dims must be 4x4, got ", x.sizes());

    const int64_t totalElements = x.numel();
    TORCH_CHECK(totalElements > 0, "input tensor must not be empty");
    const int64_t totalMats = totalElements / SN_MAT_SIZE;
    TORCH_CHECK(totalMats * SN_MAT_SIZE == totalElements, "input numel must be multiple of 16");

    at::Tensor y = at::empty_like(x);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    const uint32_t nMats = static_cast<uint32_t>(totalMats);
    const uint32_t nRepeat = static_cast<uint32_t>(repeat);
    const float fEps = static_cast<float>(eps);

    TilingCache &c = GetTilingCache();
    if (!c.valid || c.totalMats != nMats || c.repeat != nRepeat || c.eps != fEps) {
        FillTiling(c.meta, nMats, nRepeat, fEps);
        if (c.dev == nullptr) {
            auto ret = aclrtMalloc(&c.dev, sizeof(SinkhornTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
            TORCH_CHECK(ret == ACL_SUCCESS && c.dev != nullptr, "aclrtMalloc for tiling failed");
        }
        auto ret = aclrtMemcpy(c.dev, sizeof(SinkhornTilingData), &c.meta,
                               sizeof(SinkhornTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemcpy for tiling failed");
        c.totalMats = nMats;
        c.repeat = nRepeat;
        c.eps = fEps;
        c.valid = true;
    }
    const uint32_t blockDim = c.meta.blockNum;
    uint8_t *tilingPtr = reinterpret_cast<uint8_t *>(c.dev);

    sinkhorn_normalize_kernel(blockDim, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(x.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
        tilingPtr);

    g_callCount.fetch_add(1, std::memory_order_relaxed);
    return y;
}

} // namespace ascend_kernel
