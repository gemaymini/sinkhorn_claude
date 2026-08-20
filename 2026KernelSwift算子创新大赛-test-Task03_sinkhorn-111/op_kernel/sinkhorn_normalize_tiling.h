/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: CANN Open Software License Agreement Version 2.0
 */

// SinkhornNormalize 共享常量（host 与 kernel 共用，纯 C/C++，不含 ASC 关键字）
//
// 问题规模固定为 [1,1024,4,4]，切分方式（16 核 x 64 矩阵）写死在 kernel 与
// host 绑定层的 FX_* 常量里，因此这里不再需要 tiling 结构体与 tiling 计算函数。

#pragma once

#include <cstdint>

// ---- 问题形状 ----
constexpr uint32_t SN_MHC = 4;              // 4x4 矩阵
constexpr uint32_t SN_MAT_SIZE = 16;        // GM 中每矩阵 16 个 float
constexpr uint32_t SN_PAD_MAT = 32;         // UB 中每矩阵 32 个 float（4 行 x 8，右补 4）


// CopyIn 的填充值。exp(-1000) 在 fp32 下下溢为精确的 0，
// 于是填充位在整个 10 次迭代中恒为 0，不需要任何掩码缓冲区。
constexpr float SN_PAD_FILL = -1000.0f;
