/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: CANN Open Software License Agreement Version 2.0
 */

// 头文件保护宏的首次检查：仅当 OPS_H 尚未定义时编译本文件，避免重复包含导致重复声明。
#ifndef OPS_H
// 定义与上一行配对的保护宏，使后续对 ops.h 的间接或直接包含被预处理器跳过。
#define OPS_H

// 引入固定宽度整数类型，下方的调用计数接口需要 uint64_t。
#include <cstdint>
// 引入 PyTorch C++ 扩展 API，以便在声明中使用 at::Tensor。
#include <torch/extension.h>

// 把自定义算子的对外 C++ 符号放入独立命名空间，避免与其他扩展的同名函数冲突。
namespace ascend_kernel {

// 声明 NPU 实现入口：x 以常量引用传入以避免拷贝，repeat 和 eps 与 Torch 算子 schema 的 int/float 标量参数对应。
at::Tensor sinkhorn_normalize_torch(const at::Tensor& x, int64_t repeat, double eps);

// 声明进程内的 Kernel 启动计数查询接口，返回 64 位无符号值以容纳长时间累计。
uint64_t sinkhorn_normalize_call_count();

// 结束 ascend_kernel 命名空间；行尾名称用于在长文件中快速确认作用域。
} // namespace ascend_kernel

// 结束 OPS_H 头文件保护区域。
#endif // OPS_H
