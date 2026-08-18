/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// SinkhornNormalize Tiling - kernel and host shared header (pure C/C++, no ASC keywords)

#pragma once

#include <cstdint>

// Problem shape constants (must match reference: n0=1, n1=1024, mhc=4)
constexpr uint32_t SN_MHC = 4;             // 4x4 matrix
constexpr uint32_t SN_MAT_SIZE = 16;      // 4*4 = 16 floats per matrix
constexpr uint32_t SN_PADDED_SIZE = 32;    // padded to 32 floats (8 elements per row group: 4 valid + 4 zero)
constexpr uint32_t SN_REPEAT = 10;        // sinkhorn iterations
constexpr float SN_EPS = 1e-6f;           // epsilon

// UB buffer element counts (per matrix)
constexpr uint32_t SN_UB_IN = 32;         // input matrix (padded)
constexpr uint32_t SN_UB_OUT = 32;        // output matrix (padded)
constexpr uint32_t SN_UB_WORK = 64;       // work buffer for reduces
constexpr uint32_t SN_UB_REDUCE = 32;     // reduce results (row/col sums/maxes)
constexpr uint32_t SN_UB_BCAST = 32;      // broadcast buffer
constexpr uint32_t SN_UB_TRANS = 32;      // transposed matrix buffer
constexpr uint32_t SN_UB_TMP = 32;        // temp buffer

// Tiling data structure - passed to kernel
struct SinkhornTilingData {
    uint32_t blockNum;        // number of AI cores used
    uint32_t totalMats;       // total matrices (1024)
    uint32_t matPerCore;      // matrices per core (except last)
    uint32_t tailMatLastCore; // matrices for last core (tail)
    uint32_t repeat;           // sinkhorn iterations (10)
    float eps;                 // epsilon (1e-6)
};
