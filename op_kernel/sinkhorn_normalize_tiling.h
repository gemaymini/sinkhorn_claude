/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: CANN Open Software License Agreement Version 2.0
 */

// SinkhornNormalize Tiling - kernel 与 host 共用（纯 C/C++，不含 ASC 关键字）

#pragma once

#include <cstdint>

// ---- 问题形状 ----
constexpr uint32_t SN_MHC = 4;              // 4x4 矩阵
constexpr uint32_t SN_MAT_SIZE = 16;        // GM 中每矩阵 16 个 float
constexpr uint32_t SN_PAD_MAT = 32;         // UB 中每矩阵 32 个 float（4 行 × 8，右补 4）
constexpr uint32_t SN_REPEAT = 10;
constexpr float SN_EPS = 1e-6f;

// ---- 原 per-matrix kernel (SN_KERNEL_VARIANT=0) 的 UB 常量 ----
// S1 不使用这些，但对照组 sinkhorn_normalize_kernel.asc 需要
constexpr uint32_t SN_PADDED_SIZE = 32;   // 每矩阵补齐后 32 个 float
constexpr uint32_t SN_UB_IN = 32;
constexpr uint32_t SN_UB_OUT = 32;
constexpr uint32_t SN_UB_WORK = 64;
constexpr uint32_t SN_UB_REDUCE = 32;
constexpr uint32_t SN_UB_BCAST = 32;
constexpr uint32_t SN_UB_TRANS = 32;
constexpr uint32_t SN_UB_TMP = 32;

// ---- S1 常量 ----
// 单个 tile 最多处理多少矩阵。上限来自 AscendC level-0 API 的 repeatTimes 是 uint8 (<=255)，
// ColNormalize 里的 Add/Mul 用 repeatTimes = matsPerTile。
constexpr uint32_t SN_TILE_MAX = 128;

// ColNormalize 中避免 1/0 的极小量。列和量级 ~0.25~1，加 1e-30 在 fp32 下完全不影响有效位；
// 而填充列的和恒为 0，加完之后取倒数得到有限的 1e30，乘上恒为 0 的填充值仍是 0。
constexpr float SN_TINY = 1e-30f;

// CopyIn 的填充值。exp(-1000) 在 fp32 下下溢为**精确的 0**，
// 这样填充位在整个 10 次迭代中恒为 0，不需要任何掩码缓冲区。
constexpr float SN_PAD_FILL = -1000.0f;

// ---- eps 向量 ----
// 参考实现里 `x = softmax(x) + eps` 这一项**不能省**：当 softmax 输出小到 ~1e-11 时
// （输入幅值大于 randn×4 即可发生），加 eps=1e-6 会把它抬高 5 个数量级，是语义差异而非舍入。
// 但 eps 只能加在有效位上，填充位必须保持 0。做法是在 tiling buffer 尾部附一段全是 eps
// 的数组，kernel 用已验证的 `DataCopyPad + rightPadding=4` 把它展开成
// [eps,eps,eps,eps,0,0,0,0] 重复的向量，一次 Add 即可。
// sizeof(SinkhornTilingData) 恰好是 32 字节，尾部数组天然 32B 对齐。
constexpr uint32_t SN_EPSVEC_LEN = SN_TILE_MAX * SN_MHC;

// ---- Tiling ----
struct SinkhornTilingData {
    uint32_t blockNum;        // 实际使用的核数
    uint32_t totalMats;       // 矩阵总数
    uint32_t matPerCore;      // 每核矩阵数（末核可能不足）
    uint32_t tailMatLastCore; // 末核的矩阵数
    uint32_t repeat;          // Sinkhorn 迭代次数
    float eps;                // epsilon
    uint32_t matsPerTile;     // S1: 单 tile 的矩阵数（<= SN_TILE_MAX）
    uint32_t reserved;        // 保持 32 字节大小，使尾部 eps 数组 32B 对齐
};

// tiling buffer 总字节数 = 结构体 + eps 数组
constexpr uint32_t SN_TILING_BYTES =
    static_cast<uint32_t>(sizeof(SinkhornTilingData)) + SN_EPSVEC_LEN * sizeof(float);
// eps 数组在 buffer 中的 float 下标偏移
constexpr uint32_t SN_EPSVEC_OFFSET =
    static_cast<uint32_t>(sizeof(SinkhornTilingData) / sizeof(float));

// ---- host/kernel 共用的 tiling 计算（纯 C++，无 ASC 关键字）----
// tileTarget: 期望每个核处理多少矩阵。小 shape 下打满核心不一定更快
// （向量指令的 repeat 数是一样的），所以核数按 tileTarget 反推而不是无脑用满。
inline void SinkhornComputeTiling(SinkhornTilingData &t, uint32_t totalMats,
                                  uint32_t repeat, float eps,
                                  uint32_t availableCores, uint32_t tileTarget)
{
    if (tileTarget == 0 || tileTarget > SN_TILE_MAX) {
        tileTarget = SN_TILE_MAX;
    }
    uint32_t wanted = (totalMats + tileTarget - 1) / tileTarget;
    if (wanted == 0) {
        wanted = 1;
    }
    uint32_t cores = availableCores == 0 ? 1 : availableCores;
    if (cores > wanted) {
        cores = wanted;
    }
    if (cores > totalMats) {
        cores = totalMats;
    }
    if (cores == 0) {
        cores = 1;
    }

    const uint32_t matPerCore = (totalMats + cores - 1) / cores;
    const uint32_t blockNum = (totalMats + matPerCore - 1) / matPerCore;

    t.blockNum = blockNum;
    t.totalMats = totalMats;
    t.matPerCore = matPerCore;
    t.tailMatLastCore = totalMats - matPerCore * (blockNum - 1);
    t.repeat = repeat;
    t.eps = eps;
    t.matsPerTile = matPerCore < SN_TILE_MAX ? matPerCore : SN_TILE_MAX;
    t.reserved = 0;
}

// 把 tiling 结构 + eps 数组一起填进 host 侧缓冲区（至少 SN_TILING_BYTES 字节）
inline void SinkhornFillTilingBuffer(void *buf, const SinkhornTilingData &t)
{
    SinkhornTilingData *head = static_cast<SinkhornTilingData *>(buf);
    *head = t;
    float *eps = reinterpret_cast<float *>(head + 1);
    for (uint32_t i = 0; i < SN_EPSVEC_LEN; i++) {
        eps[i] = t.eps;
    }
}
