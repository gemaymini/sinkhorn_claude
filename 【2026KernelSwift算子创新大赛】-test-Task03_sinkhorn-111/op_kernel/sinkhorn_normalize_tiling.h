/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: CANN Open Software License Agreement Version 2.0
 */


#pragma once

#include <cstdint>

constexpr uint32_t SN_MHC = 4;           
constexpr uint32_t SN_MAT_SIZE = 16;   
constexpr uint32_t SN_PAD_MAT = 32;       
constexpr uint32_t SN_REPEAT = 10;
constexpr float SN_EPS = 1e-6f;

constexpr uint32_t SN_TILE_MAX = 128;

constexpr uint32_t SN_TILE_TARGET = 64;

constexpr float SN_TINY = 1e-30f;

constexpr float SN_PAD_FILL = -1000.0f;

struct SinkhornTilingData {
    uint32_t blockNum;     
    uint32_t totalMats;       
    uint32_t matPerCore;      
    uint32_t tailMatLastCore; 
    uint32_t repeat;         
    float eps;            
    uint32_t matsPerTile;   
    uint32_t reserved;     
};

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
