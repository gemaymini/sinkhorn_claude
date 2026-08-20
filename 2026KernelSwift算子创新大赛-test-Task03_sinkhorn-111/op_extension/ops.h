/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: CANN Open Software License Agreement Version 2.0
 */

#ifndef OPS_H
#define OPS_H

#include <cstdint>
#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor sinkhorn_normalize_torch(const at::Tensor& x, int64_t repeat, double eps);

} // namespace ascend_kernel

#endif // OPS_H
