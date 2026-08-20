/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: CANN Open Software License Agreement Version 2.0
 */

#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("sinkhorn_normalize(Tensor x, int repeat, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sinkhorn_normalize", TORCH_FN(ascend_kernel::sinkhorn_normalize_torch));
}

at::Tensor sinkhorn_normalize_meta(const at::Tensor& x, int64_t repeat, double eps)
{
    return at::empty_like(x);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("sinkhorn_normalize", &sinkhorn_normalize_meta);
}

} // namespace
