/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: CANN Open Software License Agreement Version 2.0
 */


// 使用一次包含保护，避免该共享头在同一编译单元中重复定义常量和结构体。
#pragma once

// 引入定宽整数类型，确保 Host 与设备侧的 tiling 字段宽度及二进制布局一致。
#include <cstdint>

// 矩阵高和有效列宽都固定为 4，许多行数与 padding 计算由此常量派生。
constexpr uint32_t SN_MHC = 4;           
// GM 采用紧凑 4×4 布局，所以每个矩阵恰好含 16 个 FP32。
constexpr uint32_t SN_MAT_SIZE = 16;   
// UB 把每行从 4 个 FP32 补到 8 个，因此每矩阵固定占 4×8=32 个 FP32。
constexpr uint32_t SN_PAD_MAT = 32;       
// 默认执行 10 轮 Sinkhorn：首轮列缩放，加后续 9 轮行列交替缩放。
constexpr uint32_t SN_REPEAT = 10;
// 默认 eps=1e-6，用于 softmax 后偏置以及行、列归一化分母。
constexpr float SN_EPS = 1e-6f;

// 单 tile 最多容纳 128 个 padded 矩阵，该值直接决定 Kernel 各 UB 缓冲容量。
constexpr uint32_t SN_TILE_MAX = 128;

// 期望让每核处理约 64 个矩阵，在较长向量 repeat 与多核并行之间取得平衡。
constexpr uint32_t SN_TILE_TARGET = 64;

// eps 不大于该值时列分母会改用此极小正数，专门避免恒零 padding 列出现 0/0。
constexpr float SN_TINY = 1e-30f;

// CopyIn 用 -1000 填充无效 lane，使其在稳定 softmax 的 Exp 后下溢为精确 0。
constexpr float SN_PAD_FILL = -1000.0f;

// 该结构是 Host 写入、Kernel 读取的 ABI 合同，字段顺序和类型不能单边改变。
struct SinkhornTilingData {
    // 实际启动的 AIV block 数；Host 将该字段作为 Kernel launch 的 blockDim。
    uint32_t blockNum;     
    // 输入展平后 4×4 矩阵总数，等于张量 numel/16。
    uint32_t totalMats;       
    // 普通核的矩阵区间跨度，设备侧用 blockIdx×matPerCore 定位任务起点。
    uint32_t matPerCore;      
    // 最后一个有效核的真实矩阵数，用于描述总数不能整除时的尾部工作量。
    uint32_t tailMatLastCore; 
    // 本次调用的 Sinkhorn 总轮数，由 Kernel 读入 repeat_ 控制循环。
    uint32_t repeat;         
    // 本次调用的 eps，必须与参考实现的数值语义保持一致。
    float eps;            
    // 单核每次装入 UB 的矩阵数；工作量过大时 Kernel 会循环多个 tile。
    uint32_t matsPerTile;   
    // 预留 32 位扩展字段并清零，使结构保持稳定的 32B 大小与确定的二进制内容。
    uint32_t reserved;     
// 结束 Host/Kernel 共享 tiling 结构定义。
};

// 该 Host 内联函数根据问题规模和硬件核数生成负载划分。
inline void SinkhornComputeTiling(SinkhornTilingData &t, uint32_t totalMats,
                                  // repeat 与 eps 承载算子语义，并原样写入 tiling 供 Kernel 使用。
                                  uint32_t repeat, float eps,
                                  // availableCores 限制并行上限，tileTarget 决定希望每个核摊销多少矩阵。
                                  uint32_t availableCores, uint32_t tileTarget)
// 进入纯 Host 侧整数 tiling 计算；正常调用前提是 totalMats>0、repeat>=1，本函数不访问设备数据。
{
    // 检查目标 tile 是否会导致除零、循环不推进或超过设备 UB 静态容量。
    if (tileTarget == 0 || tileTarget > SN_TILE_MAX) {
        // 非法目标统一回退到最大安全 tile 大小 SN_TILE_MAX。
        tileTarget = SN_TILE_MAX;
    // 结束 tileTarget 防御性修正。
    }
    // 按每核约 tileTarget 个矩阵反推期望核数，整数上取整确保覆盖所有矩阵。
    uint32_t wanted = (totalMats + tileTarget - 1) / tileTarget;
    // 空输入时上式得到 0，需要在参与后续核数计算前提供非零下界。
    if (wanted == 0) {
        // 把 wanted 钳到 1；但空输入仍会令后面的 matPerCore=0，因此正式调用必须保证 totalMats>0。
        wanted = 1;
    // 结束期望核数下界修正。
    }
    // 硬件查询返回 0 时按查询失败处理并退化到单核，否则采用报告的可用核数。
    uint32_t cores = availableCores == 0 ? 1 : availableCores;
    // 核数超过 wanted 会让每核矩阵过少、缩短 repeat 并放大 launch 固定成本。
    if (cores > wanted) {
        // 将候选核数限制为达到目标每核工作量所需的数量。
        cores = wanted;
    // 结束期望工作粒度带来的核数上限修正。
    }
    // 核数不能多于真实矩阵数，否则会创建没有任何输入的空 block。
    if (cores > totalMats) {
        // 把核数限制到至多一核一个矩阵。
        cores = totalMats;
    // 结束矩阵数量带来的核数上限修正。
    }
    // totalMats=0 时上一分支可能令 cores 变 0，因此除法前再次检查。
    if (cores == 0) {
        // 把 cores 钳到 1；这只保护核数本身，空输入仍不受支持并应在调用 tiling 前被拒绝。
        cores = 1;
    // 结束实际核数下界修正。
    }

    // 把矩阵均分到候选核，使用上取整得到普通核的固定区间跨度。
    const uint32_t matPerCore = (totalMats + cores - 1) / cores;
    // 在 totalMats>0 因而 matPerCore>0 的前提下，重算实际 block 数以避免多余空核。
    const uint32_t blockNum = (totalMats + matPerCore - 1) / matPerCore;

    // 写入实际 launch block 数；1024 个矩阵、目标 64 时通常得到 16。
    t.blockNum = blockNum;
    // 保存完整矩阵总数，供设备侧尾核钳制终点并防止 GM 越界。
    t.totalMats = totalMats;
    // 保存每核区间跨度，Kernel 用它结合 blockIdx 计算 matStart_。
    t.matPerCore = matPerCore;
    // 总数减去前 blockNum-1 个满核工作量，得到最后一个核的真实尾部数量。
    t.tailMatLastCore = totalMats - matPerCore * (blockNum - 1);
    // 原样透传迭代次数，tiling 层不改变算法定义。
    t.repeat = repeat;
    // 原样透传 eps，保证自定义 Kernel 与参考实现数值一致。
    t.eps = eps;
    // 每核工作量能装下时只做一个 tile，否则按 128 个矩阵分批复用 UB。
    t.matsPerTile = matPerCore < SN_TILE_MAX ? matPerCore : SN_TILE_MAX;
    // 当前版本不使用扩展字段，显式清零可避免未初始化字节影响缓存或二进制比较。
    t.reserved = 0;
// 结束 tiling 生成；结构体可被复制到设备 GM 供每个 Kernel block 只读使用。
}
