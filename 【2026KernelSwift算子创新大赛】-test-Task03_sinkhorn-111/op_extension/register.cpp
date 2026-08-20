/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: CANN Open Software License Agreement Version 2.0
 */

// 引入 PyTorch C++ 扩展中的 Tensor 与类型桥接定义，供后面的函数签名和 TORCH_FN 使用。
#include <torch/extension.h>
// 引入 PyTorch dispatcher 注册宏，包括 TORCH_LIBRARY_FRAGMENT 和 TORCH_LIBRARY_IMPL。
#include <torch/library.h>
// 引入本扩展的公开函数声明，使注册层能绑定实际 NPU 实现与计数接口。
#include "ops.h"

// 打开匿名命名空间，令本文件的适配函数和注册辅助符号具有内部链接，不污染其他编译单元。
namespace {

// 定义 dispatcher 可直接装箱返回的计数适配函数；Torch schema 的 int 在 C++ 端对应 int64_t。
int64_t sinkhorn_normalize_call_count_boxed()
// 进入计数适配函数的作用域，该函数不接收 Tensor，不会启动 NPU Kernel。
{
    // 读取底层的 uint64_t 原子计数，再显式转为 schema 要求的 int64_t 返回给 Python。
    return static_cast<int64_t>(ascend_kernel::sinkhorn_normalize_call_count());
// 结束计数适配函数。
}

// 向已有的 npu 算子库追加 schema；FRAGMENT 允许同一命名空间分散在多个扩展文件中注册。
TORCH_LIBRARY_FRAGMENT(npu, m)
// 进入算子 schema 定义块，m 是 PyTorch 传入的 Library 注册器。
{
    // 声明主算子的 Python/C++ 契约：输入 Tensor、迭代次数和稳定项，返回一个同形状 Tensor。
    m.def("sinkhorn_normalize(Tensor x, int repeat, float eps) -> Tensor");
    // 声明并直接绑定调用计数辅助算子，它用于测试或确认自定义路径是否被执行。
    m.def("sinkhorn_normalize_call_count() -> int", &sinkhorn_normalize_call_count_boxed);
// 结束 npu schema 追加块。
}

// 为 PrivateUse1 调度键注册实现；torch_npu 使用该键表示真实 NPU Tensor 的执行路径。
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
// 进入 NPU 后端实现注册块，只有 dispatcher 选中 PrivateUse1 时才会进入这里绑定的函数。
{
    // 将算子名映射到真正的 Host 入口；TORCH_FN 保留函数签名信息供 dispatcher 做类型安全调用。
    m.impl("sinkhorn_normalize", TORCH_FN(ascend_kernel::sinkhorn_normalize_torch));
// 结束 PrivateUse1 实现注册块。
}

// 定义 Meta 推导函数：参数签名必须与 schema 一致，但 repeat 和 eps 不影响输出元数据。
at::Tensor sinkhorn_normalize_meta(const at::Tensor& x, int64_t repeat, double eps)
// 进入 Meta 函数作用域；该路径只做形状、步长和 dtype 推导，不读写真实 NPU 数据。
{
    // 创建与 x 具有相同尺寸、步长、dtype 和设备属性的空 Tensor，准确表达该算子的输出元数据。
    return at::empty_like(x);
// 结束 Meta 推导函数。
}

// 为 Meta 调度键注册形状推导实现，使 FakeTensor、图编译和形状传播无需启动真实 Kernel。
TORCH_LIBRARY_IMPL(npu, Meta, m)
// 进入 Meta 后端的实现注册块。
{
    // 将同一算子名绑定到上方的元数据函数，dispatcher 会根据 Tensor 的调度键选择它或 NPU 实现。
    m.impl("sinkhorn_normalize", &sinkhorn_normalize_meta);
// 结束 Meta 实现注册块。
}

// 结束匿名命名空间，上述适配函数与注册辅助符号不对其他编译单元暴露。
} // namespace
