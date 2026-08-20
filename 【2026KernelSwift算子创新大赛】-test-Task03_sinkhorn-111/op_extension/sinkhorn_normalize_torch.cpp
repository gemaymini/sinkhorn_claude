/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: CANN Open Software License Agreement Version 2.0
 */

// 引入原子类型，用于在多线程 Host 调用中无数据竞争地统计算子启动次数。
#include <atomic>
// 引入固定宽度整数类型，确保 Host 与 Device 之间的 tiling 字段宽度一致。
#include <cstdint>
// 引入 Ascend Computing Language 运行时 API，供设备查询、显存分配、拷贝和 stream 类型使用。
#include "acl/acl.h"
// 引入 PyTorch C++ 扩展 API，提供 at::Tensor、TORCH_CHECK 及张量数据指针接口。
#include <torch/extension.h>
// 引入 torch_npu 的 stream 适配层，用于取得当前 PyTorch NPU 执行流。
#include "torch_npu/csrc/core/npu/NPUStream.h"
// 共享 Kernel 侧的布局常量、SinkhornTilingData 二进制结构和 tiling 计算函数，避免 Host/Device 契约漂移。
#include "../op_kernel/sinkhorn_normalize_tiling.h"

// 保留工程原有的同名宏包装：宏展开期间内层同名标识符不会递归再展开，最终仍引用 tiling 头文件中的常量。
#define SN_TILE_TARGET (SN_TILE_TARGET)

// 以 C 链接规则声明由 AscendC 编译产物导出的 Kernel stub，防止 C++ 名字改编导致链接失败。
extern "C" void sinkhorn_normalize_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                           // 声明余下的裸字节指针参数：x/y 是设备输入输出，tiling 指向设备上的调度元数据。
                                           uint8_t *x, uint8_t *y, uint8_t *tiling);

// 打开项目的算子命名空间，与 ops.h 中声明的对外符号保持一致。
namespace ascend_kernel {

// 打开内部匿名命名空间，限制计数器、查询函数与缓存的链接可见性。
namespace {

// 从 0 开始记录 Host 入口成功发射 Kernel 的次数；原子类型允许并发读写。
std::atomic<uint64_t> g_callCount{0};

// 定义冷路径设备查询函数，它返回当前 NPU 可供调度的 Vector Core 数量。
int64_t QueryVectorCoreNum()
// 进入 Vector Core 数量查询的 Host 函数作用域。
{
    // 用 -1 初始化设备号，便于在运行时没有正确写入时保留明显的无效值。
    int32_t deviceId = -1;
    // 向 ACL Runtime 请求当前线程绑定的 NPU 设备号，供下一步查询硬件属性。
    aclrtGetDevice(&deviceId);
    // 预置核数输出为 0，便于与 ACL 返回码一起判断查询结果是否有效。
    int64_t n = 0;
    // 查询当前设备的 Vector Core 数，ret 保存 ACL 错误码，n 由运行时写入。
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &n);
    // 同时检查 API 成功且核数为正，失败时转换为带明确信息的 PyTorch 异常。
    TORCH_CHECK(ret == ACL_SUCCESS && n > 0, "failed to get NPU vector core count");
    // 返回验证后的核数，稍后 tiling 会用它限制 blockNum。
    return n;
// 结束设备查询函数。
}

// 定义轻量级核数访问器，inline 允许编译器消除额外函数调用开销。
inline int64_t GetVectorCoreNum()
// 进入核数缓存访问器的作用域。
{
    // 函数局部 static 只在首次调用时执行设备查询，后续热路径直接复用不变的核数。
    static const int64_t cached = QueryVectorCoreNum();
    // 返回已缓存的核数，避免每次 forward 都进入 ACL 设备查询。
    return cached;
// 结束核数缓存访问器。
}

// 定义单个进程级 tiling 缓存，同时保存缓存键、Host 元数据和 Device 指针。
struct TilingCache {
    // 标记缓存键和设备数据是否已完成初始化，false 会强制首次调用走冷路径。
    bool valid = false;
    // 缓存上次 tiling 对应的 4×4 矩阵总数，用于检测输入形状变化。
    uint32_t totalMats = 0;
    // 缓存上次 tiling 的 Sinkhorn 迭代次数，参数改变时需重新计算并拷贝。
    uint32_t repeat = 0;
    // 缓存上次 tiling 使用的 FP32 稳定项，按实际传给 Kernel 的精度比较。
    float eps = 0.0f;
    // 保存 Host 侧 SinkhornTilingData 副本，值初始化确保所有字段和填充位从确定状态开始。
    SinkhornTilingData meta{};
    // 指向持久化的 NPU 显存 tiling 缓冲区，为 nullptr 表示尚未进行首次分配。
    void *dev = nullptr;   
// 结束 tiling 缓存结构定义。
};

// 定义 tiling 缓存的唯一访问入口，返回引用以允许热路径就地更新状态。
TilingCache &GetTilingCache()
// 进入进程级 tiling 缓存访问函数的作用域。
{
    // 函数局部 static 在程序生命周期内只构造一次，从而跨多次 forward 保留 Host/Device tiling。
    static TilingCache c;
    // 返回单例缓存的左值引用，调用者可检查键并更新成员。
    return c;
// 结束 tiling 缓存访问函数。
}

// 定义 Host 侧 tiling 填充函数，将输入规模和数值参数转换成 Kernel 可直接消费的结构。
void FillTiling(SinkhornTilingData &t, uint32_t totalMats, uint32_t repeat, float eps)
// 进入 tiling 元数据填充函数的作用域。
{
    // 调用 Host/Kernel 共享的调度算法，前四项传入输出结构、矩阵数、迭代次数与 eps。
    SinkhornComputeTiling(t, totalMats, repeat, eps,
                          // 把已缓存的核数收窄为 uint32_t，并传入目标 tile 大小以决定 blockNum 和每核工作量。
                          static_cast<uint32_t>(GetVectorCoreNum()), SN_TILE_TARGET);
// 结束 tiling 填充函数。
}

// 结束内部匿名命名空间；以下只定义 ops.h 暴露的两个对外函数。
} // namespace

// 定义对外的调用次数查询函数，注册层会把它适配为 Torch 标量算子。
uint64_t sinkhorn_normalize_call_count()
// 进入计数查询函数的作用域。
{
    // 以 relaxed 语义原子读取计数：此值只用于统计，不承担其他内存操作的同步顺序。
    return g_callCount.load(std::memory_order_relaxed);
// 结束调用次数查询函数。
}

// 定义 PyTorch dispatcher 调用的 Host 主入口，负责校验、tiling 准备、Kernel 发射和输出返回。
at::Tensor sinkhorn_normalize_torch(const at::Tensor& x, int64_t repeat, double eps)
// 进入 Sinkhorn NPU 算子的 Host 执行作用域，此函数本身不执行矩阵数学计算。
{
    // 限定输入为 FP32，因为 Kernel 的 GM/UB 布局、向量指令和字节数都按 float 特化。
    TORCH_CHECK(x.scalar_type() == at::kFloat, "only FP32 supported");
    // 确认 x 由 PrivateUse1/torch_npu 后端持有，避免把 CPU 或其他设备指针误传给 NPU Kernel。
    TORCH_CHECK(x.is_privateuseone(), "x must be on NPU");
    // 确保至少存在两个尾维用于表示单个矩阵，前导维可以是任意 batch 组合。
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims (..., mhc, mhc)");
    // 检查最后两维都等于 SN_MHC=4，这是 Kernel 固定行列布局的基本前提。
    TORCH_CHECK(x.size(-1) == SN_MHC && x.size(-2) == SN_MHC,
        // 组装校验失败信息，附上实际 sizes 帮助用户定位形状错误。
        "last two dims must be 4x4, got ", x.sizes());

    // 读取输入张量的总元素数，后续将所有前导维统一展平为矩阵数量。
    const int64_t totalElements = x.numel();
    // 拒绝空输入，避免 tiling 生成 0 矩阵时出现无意义的分核和启动参数。
    TORCH_CHECK(totalElements > 0, "input tensor must not be empty");
    // 用每个 4×4 矩阵的 16 个元素换算矩阵总数，这是 tiling 的主要规模参数。
    const int64_t totalMats = totalElements / SN_MAT_SIZE;
    // 再次检查元素数能被 16 整除，保证展平后不存在不完整的尾部矩阵。
    TORCH_CHECK(totalMats * SN_MAT_SIZE == totalElements, "input numel must be multiple of 16");

    // 在与 x 相同设备、dtype、形状和布局上分配输出 y，内容由即将启动的 Kernel 覆盖。
    at::Tensor y = at::empty_like(x);
    // 获取 PyTorch 当前 NPU stream 的原生 ACL handle，使自定义 Kernel 与上下游 PyTorch 操作保持正确顺序。
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 将 PyTorch 使用的 64 位矩阵数收窄为 tiling ABI 规定的 32 位无符号整数。
    const uint32_t nMats = static_cast<uint32_t>(totalMats);
    // 将 schema 的 int64_t 迭代数转为 Device tiling 字段的 uint32_t 表示。
    const uint32_t nRepeat = static_cast<uint32_t>(repeat);
    // 将 dispatcher 传入的 double 标量转为 Kernel 实际运算所用的 FP32 eps。
    const float fEps = static_cast<float>(eps);

    // 取得进程级 tiling 缓存的可写引用，后续依据键决定走冷路径还是热路径。
    TilingCache &c = GetTilingCache();
    // 只要缓存未初始化或矩阵数、迭代数、FP32 eps 任一变化，就重建设备 tiling。
    if (!c.valid || c.totalMats != nMats || c.repeat != nRepeat || c.eps != fEps) {
        // 在 Host 内存中计算新的分核、每核矩阵数、尾核大小和 tile 大小等字段。
        FillTiling(c.meta, nMats, nRepeat, fEps);
        // 设备指针只需要在缓存生命周期中分配一次，后续缓存失配只覆盖其内容。
        if (c.dev == nullptr) {
            // 在 NPU 显存分配一个 SinkhornTilingData 大小的缓冲区，HUGE_FIRST 指定 ACL 的大页优先分配策略。
            auto ret = aclrtMalloc(&c.dev, sizeof(SinkhornTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
            // 同时检查 ACL 返回成功且输出指针非空，防止 Kernel 解引用无效 tiling 地址。
            TORCH_CHECK(ret == ACL_SUCCESS && c.dev != nullptr, "aclrtMalloc for tiling failed");
        // 结束仅首次缓存使用时执行的设备分配分支。
        }
        // 将 Host 上的 tiling 结构同步拷贝到持久化设备缓冲区，前两个大小参数分别是目标容量和源长度。
        auto ret = aclrtMemcpy(c.dev, sizeof(SinkhornTilingData), &c.meta,
                               // 指定完整拷贝一个 tiling 结构，方向为 Host-to-Device，使 Kernel 能从 GM 读取参数。
                               sizeof(SinkhornTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
        // 拷贝失败时立即抛出 PyTorch 异常，不让后续 Kernel 使用过期或未初始化的设备元数据。
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemcpy for tiling failed");
        // 在成功拷贝后记录当前矩阵数，作为下次缓存命中比较的第一部分。
        c.totalMats = nMats;
        // 记录当前迭代次数，确保 repeat 改变时不会错用旧 tiling。
        c.repeat = nRepeat;
        // 记录已转为 FP32 的 eps，这与设备侧真正消费的数值精度一致。
        c.eps = fEps;
        // 最后标记缓存有效，确保只有 tiling 计算、分配和 H2D 均成功后才允许走热路径。
        c.valid = true;
    // 结束 tiling 缓存失配处理；命中时会跳过计算、分配和 H2D 拷贝。
    }
    // 从 tiling 结果取出 Kernel 启动 block 数，每个 block 对应一个 Ascend Vector Core 任务。
    const uint32_t blockDim = c.meta.blockNum;
    // 把通用设备指针转为 Kernel stub ABI 所要求的字节指针，不改变其地址或内容。
    uint8_t *tilingPtr = reinterpret_cast<uint8_t *>(c.dev);

    // 异步向当前 NPU stream 发射 Sinkhorn Kernel，传入分核数、未使用的 L2 控制指针和 stream。
    sinkhorn_normalize_kernel(blockDim, nullptr, aclStream,
        // 获取 x 的设备数据地址并转为 stub 的字节指针；Kernel 逻辑只读取该缓冲区。
        reinterpret_cast<uint8_t*>(x.mutable_data_ptr()),
        // 获取新分配输出 y 的设备地址，Kernel 会将所有 Sinkhorn 结果写入该缓冲区。
        reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
        // 传入已同步到 Device 的 tiling 指针，供每个 Kernel block 解析自身工作范围。
        tilingPtr);

    // 在发射调用返回后原子加一；relaxed 足以保证计数不丢失，又不引入不必要的内存序屏障。
    g_callCount.fetch_add(1, std::memory_order_relaxed);
    // 立即返回挂在同一 stream 上的输出 Tensor，PyTorch 依靠 stream 语义保证后续消费时计算已完成。
    return y;
// 结束 Sinkhorn NPU Host 入口。
}

// 结束 ascend_kernel 命名空间，与 ops.h 的对外声明作用域配对。
} // namespace ascend_kernel
