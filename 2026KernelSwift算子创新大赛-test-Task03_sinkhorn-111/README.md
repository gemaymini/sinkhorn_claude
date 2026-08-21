# SinkhornNormalize 昇腾自定义算子

**2026 KernelSwift 算子创新大赛 · 赛道二 · Task03 sinkhorn**

| 项目 | 值 |
| --- | --- |
| 目标算子 | `sinkhorn_normalize`，输入 `[1, 1024, 4, 4]` float32 |
| 参考实现 | `model_ref.py` 中的 `Model`（赛题原版 PyTorch） |
| 提交实现 | `model_new.py` 中的 `ModelNew`（调用自研 AscendC 算子） |
| 硬件 | 华为昇腾 Atlas A2 / 910B2（`dav-2201`） |
| 软件 | CANN 9.0.0，Python 3.11，torch 2.10.0 + torch_npu 2.10.0 |
| 官方评测结果 | v0 `1252.750 us` → v1 `144.385 us`，**加速比 8.676x**，正确性 PASS |

> 上表的性能数字由 `run_auto_bench.sh` 调用**官方原版** `auto_bench.py` 产生并自动写入
> [results/performance.txt](results/performance.txt)，包含完整原始输出，非人工填写。

---

## 一、作品说明

### 1.1 算子语义

赛题要求把一批矩阵迭代归一化成双随机矩阵：

```python
x = x.softmax(-1) + eps                        # 行 softmax 后整体加 eps
x = x / (x.sum(-2, keepdim=True) + eps)        # 第一次列归一化
for _ in range(repeat - 1):                    # 再做 repeat-1 轮
    x = x / (x.sum(-1, keepdim=True) + eps)    #   行归一化
    x = x / (x.sum(-2, keepdim=True) + eps)    #   列归一化
```

### 1.2 优化核心思路

这个算子的关键特征是**规模极小、访存与调度开销占主导**：1024 个 4×4 矩阵总共只有 64 KiB 数据，
真正的浮点计算量微不足道。因此优化重心不在浮点吞吐，而在**减少 kernel 启动次数、减少 DMA 描述符数量、减少 host 侧固定开销、减少跨流水线同步**。

### 1.3 目录结构

```
.
├── README.md                       本文件
├── model_ref.py                    赛题参考实现（v0，未改动）
├── model_new.py                    提交实现（v1），forward 只调用自定义算子
├── op_kernel/
│   ├── sinkhorn_normalize_kernel.asc   device 侧 kernel（核心）
│   └── sinkhorn_normalize_tiling.h     host/kernel 共享的形状常量
├── op_extension/
│   ├── sinkhorn_normalize_torch.cpp    host 侧绑定：分配输出 + launch
│   ├── register.cpp                    注册 torch.ops.npu.sinkhorn_normalize
│   └── ops.h
├── CMakeLists.txt                  构建脚本（无需任何 -D 选项）
├── build.sh                        一键编译，产物复制到本目录
├── run_auto_bench.sh               用官方 auto_bench.py 评测并刷新 performance.txt
├── run_auto_bench_mulseed.py       多随机种子精度回归（可选，非评测必需）
├── requirements.txt                运行环境说明
├── libsinkhorn_normalize_ops.so    预编译产物（可直接评测；也可用 build.sh 重新生成）
└── results/performance.txt         官方 auto_bench.py 的评测结果与原始输出
```

---

## 二、优化方案

本实现相对参考实现的优化集中在以下八点：

| # | 优化点 | 要点 |
| --- | --- | --- |
| 2.1 | 算子融合 | 整条链路合并为单次 kernel 调用 |
| 2.2 | UB 数据布局 | padded-AoS，每行对齐一个 32B block |
| 2.3 | 填充值哨兵 | 填充位取 −1000，全程恒为精确 0 |
| 2.4 | 批量向量化 | 单核一次处理 64 个矩阵 |
| 2.5 | eps 掩码常量化 | 两条 `Duplicate`，`Init` 中构造一次 |
| 2.6 | 同步精简 | 仅保留三个跨流水线事件 |
| 2.7 | 形状与切分常量化 | 16 核 × 64 矩阵，编译期常量 |
| 2.8 | Host 路径精简 | 每次调用仅剩三步 |

### 2.1 算子融合：整条链路合并为单次 kernel 调用

参考实现中每一次 `sum` / `div` 均为一次独立的 PyTorch 算子调用，`repeat=10` 时需要 20 余次
kernel launch 与同等次数的 Global Memory 往返。

本实现将 softmax（含减行最大值）、加 eps 以及 20 次行/列归一化融合进单个 AscendC vector kernel，
执行路径为 `CopyIn → 全部计算 → CopyOut`，全部中间结果驻留 UB，不产生 Global Memory 往返。

### 2.2 UB 数据布局：padded-AoS，每行对齐一个 32B block

Global Memory 中每个矩阵为紧凑的 16 个 float，单行 4 个 float 合 16B；而昇腾一个 32B block
可容纳 8 个 FP32。`CopyIn` 阶段通过 `DataCopyPad` 的 `rightPadding` 将每行右补 4 个元素：

```
UB 中每个矩阵（4 行 × 8 float = 32 float = 128B）：
  r0: a00 a01 a02 a03 | p p p p
  r1: a10 a11 a12 a13 | p p p p
  r2: a20 a21 a22 a23 | p p p p
  r3: a30 a31 a32 a33 | p p p p
索引：z[32*m + 8*r + c]，c < 4 为有效位，c >= 4 为填充位
```

该布局使得：

- 每行正好占据一个 32B block，行归约可直接使用 `BlockReduceSum` / `BlockReduceMax`；
- 行内广播可直接使用 `Brcb`；
- 列求和与列除法可通过 block stride 在 AoS 布局上逻辑地访问同一列，无需显式转置。

五个 UB 缓冲区合计约 **27 KiB**：`zBuf` / `bcastBuf` / `epsBuf` 各 8 KiB，`rowBuf` 1 KiB，`colBuf` 2 KiB。

### 2.3 填充值哨兵：填充位在全部迭代中恒为精确 0

`CopyIn` 时填充值取 **−1000**（`SN_PAD_FILL`）。FP32 下 $e^{-1000}$ 下溢为精确的 0，因此：

| 环节 | 填充位行为 |
| --- | --- |
| `BlockReduceMax` 求行最大值 | −1000 不会成为最大值 |
| 减去行最大值后 | 仍远小于 −100 |
| `Exp` | 精确 0 |
| 行和、列和归约 | 0 为加法单位元，不污染归约结果 |
| `Div` | `0 / 正数 = 0` |

列归一化中填充列的列和恒为 0，加上 `eps` 后得到非零除数，`0 / eps = 0`，不产生 inf/NaN。

由此，kernel 中不含掩码缓冲区、`GetValue` / `SetValue`、标量端修正逻辑及相应的 S/V 同步。

### 2.4 批量向量化：单核一次处理 64 个矩阵

**行归一化**（`RowNormalize`）由四条向量指令构成，每条均以 repeat 覆盖 64 个矩阵的全部 256 行：

```
BlockReduceSum   每个 8-float block 求和，填充位为 0，结果即为行和
Adds             行和加 denomEps（softmax 一次传 0，Sinkhorn 迭代传 eps）
Brcb             将行和广播回对应的 8-lane 行
Div              整行除以广播后的分母
```

**列归一化**（`ColNormalize`）利用 UB 中 block 的排列规律——同一矩阵相邻行相差 1 个 block，
不同矩阵的同一行相差 4 个 block——以 `mask=64`、`blkStride=4`、`repStride=32` 使一次 repeat
同时覆盖 8 个矩阵的同一行：

```
colSum  = z[r0] + z[r1]        3 条 Add 求列和
colSum += z[r2]
colSum += z[r3]
colSum += eps                  1 条 Adds
z[rk] /= colSum   (k = 0..3)   4 条 Div
```

64 个矩阵的一次完整列归一化需 `64 / 8 = 8` 次 repeat。

除法全程使用 `Div` 指令，未使用精度更低的 `Reciprocal` 近似指令。

### 2.5 eps 掩码常量化

参考语义 `x = softmax(x) + eps` 要求仅对 4 个有效位加 `eps`，填充位加 0，即需要向量
`[eps eps eps eps 0 0 0 0]`。该向量与输入无关，本实现以两条 `Duplicate` 按 lane 位图直接构造，
并置于 `Init` 中，整个 kernel 生命周期内只执行一次：

```cpp
Duplicate<float>(m, 0.0f, cnt);                    // 整块清零，保证填充位为精确 0
uint64_t mask[2] = {SN_LANE_MASK, SN_LANE_MASK};   // 0x0F0F0F0F0F0F0F0F：每 block 仅选 lane 0~3
Duplicate<float>(m, eps_, mask, cnt / 64, 1, 8);   // 仅写有效位
```

由此该掩码不占用任何 Global Memory 读取与 MTE 描述符，且不进入每个 tile 的热路径。

### 2.6 同步精简：仅保留三个跨流水线事件

同一 V 流水线内的向量指令按序发射，故 kernel 中不设置任何 `PipeBarrier<PIPE_V>`。
显式同步仅用于跨流水线依赖，共三处：

- `MTE2 → V`：`CopyIn` 完成后方可计算；
- `V → MTE3`：计算完成后方可 `CopyOut`；
- `MTE3 → MTE2`：写回完成后 UB 方可被复用。

### 2.7 形状与切分常量化：16 核 × 64 矩阵

赛题规模固定为 1024 个矩阵，切分为 16 个 AI Core、每核 64 个矩阵。`1024 = 16 × 64` 整除，
因此每核恰好一个 tile，不存在尾核与尾块。

`FX_MATS = 1024`、`FX_CORES = 16`、`FX_PER_CORE = 64` 均为编译期常量，由此：

- 所有 `count` 与 `repeatTimes` 折叠为立即数，标量单元无需在向量指令前计算参数；
- UB 按实际用量分配，不按通用上限预留；
- tile 循环与尾块分支被完全消除；
- tiling 结构体与 tiling GM 缓冲不再需要，`repeat` 与 `eps` 改由 kernel 标量入参传入。

### 2.8 Host 路径精简：每次调用仅剩三步

Host 绑定层不再查询设备核数、不再计算 tiling、不再分配 tiling 张量、不再执行 H2D 拷贝。
每次调用只保留一次形状校验与三个必要动作：

```cpp
TORCH_CHECK(x.numel() == FX_MATS * SN_MAT_SIZE, ...);   // 形状校验
at::Tensor y = at::empty_like(x);                       // 1. 分配输出
sinkhorn_normalize_kernel(FX_CORES, nullptr,            // 3. launch
    c10_npu::getCurrentNPUStream().stream(true),        // 2. 取 stream
    x.mutable_data_ptr(), y.mutable_data_ptr(), repeat, eps);
```

`model_new.py` 的 `forward()` 同样只调用自定义算子一次；`torch.ops.load_library` 置于
`__init__` 中且仅加载一次，不进入热路径。

### 2.9 语义等价性

本实现未采用任何数值近似手段：保留完整的 `softmax(x) + eps` 语义，保留全部 10 轮 Sinkhorn 迭代，
全程 FP32 计算，除法使用 `Div` 而非 `Reciprocal`。减去行最大值属于 softmax 的等价变换。
因此本作品的加速比全部来自工程实现，与参考实现逐步严格一致。

## 三、复现实验要进行的操作

### 3.1 环境要求

| 项 | 要求 |
| --- | --- |
| 硬件 | 华为昇腾 Atlas A2 / 910B2（`dav-2201`） |
| CANN | 9.0.0（提供 bisheng 编译器与 ASC 语言支持） |
| Python | 3.11 |
| Python 包 | `torch==2.10.0`、`torch_npu==2.10.0`、`numpy`（见 `requirements.txt`） |
| 构建 | CMake ≥ 3.16，GCC 11 |

`torch_npu` 必须与 CANN 版本匹配，请按昇腾官方指引安装。

### 3.2 准备环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

（`build.sh` 会在 `ASCEND_HOME_PATH` 未设置时自动到 `/usr/local/Ascend/ascend-toolkit/latest`
和 `/usr/local/Ascend/cann-*` 下搜索并 source，但显式执行一次更稳妥。）

### 3.3 编译

```bash
cd <本目录>
bash build.sh
```

`build.sh` 依次做三件事：检查 CANN 环境 → **定位真正装有 torch 与 torch_npu 的解释器**
（CMake 的 `find_package(Python3)` 可能挑到没装 torch 的解释器，必须显式指定）→ 编译。
产物 `build/libsinkhorn_normalize_ops.so` 会被复制到本目录，`model_new.py` 无需任何额外配置即可找到。

> 本目录已附带预编译好的 `libsinkhorn_normalize_ops.so`，若环境完全一致可跳过本步直接评测；
> 建议仍执行一次 `build.sh` 以验证源码可从零构建。

### 3.4 评测（官方口径）

```bash
bash run_auto_bench.sh          # 不带参数 = 官方口径（自动下载 auto_bench.py 到 /tmp）
```

脚本调用**官方原版** `auto_bench.py`（不做任何修改），以 `model_ref.py` 为 v0、`model_new.py` 为 v1，
并把结果连同完整原始输出写入 `results/performance.txt`。

**所有评测参数的默认值与官方 `auto_bench.py` 的 argparse 默认值逐项一致**，
因此不带参数运行即为官方口径；亦可按需覆盖任意一项以进行自定义实验。

**评测参数**（名称与官方 `auto_bench.py` 逐字一致，脚本原样透传，不做任何改写）：

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--seed N` | `42` | 随机种子 |
| `--atol F` | `1e-2` | `torch.allclose` 绝对容差 |
| `--rtol F` | `1e-2` | `torch.allclose` 相对容差 |
| `--warmup N` | `200` | 预热次数 |
| `--repeat N` | `500` | 计时次数，取 median |
| `--fail-fast` | 关 | 第一个用例失败即停止 |
| `--full-traceback` | 关 | 加载/运行失败时打印完整 traceback |
| `--v0_file PATH` | `./model_ref.py` | 基准实现 |
| `--v1_file PATH` | `./model_new.py` | 提交实现 |

**脚本自身选项**：

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--auto-bench PATH` | `$AUTO_BENCH`，未设置则 `/tmp/auto_bench.py` | 官方脚本路径，不存在时自动下载 |
| `--no-download` | 关 | 找不到官方脚本时直接报错，不联网 |
| `--python PATH` | 自动探测（也可用 `$PYTHON`） | 指定装有 torch/torch_npu 的解释器 |
| `--output PATH` | `./results/performance.txt` | 结果文件路径 |
| `--no-report` | 关 | 只跑评测，不写结果文件 |
| `-h, --help` | — | 显示完整帮助 |

```bash
bash run_auto_bench.sh --seed 7                    # 换随机种子
bash run_auto_bench.sh --warmup 50 --repeat 100    # 快速跑一遍
bash run_auto_bench.sh --atol 1e-3 --rtol 1e-3     # 收紧精度判据
bash run_auto_bench.sh --auto-bench /path/to/auto_bench.py    # 指定本地官方脚本
bash run_auto_bench.sh -- --新选项 值                          # "--" 之后原样透传
```

`--seed 7` 与 `--seed=7` 两种写法均支持。脚本不接受位置参数，也不提供任何别名写法，
凡无法识别的 token 一律报错退出；本脚本尚未包装的官方新参数请置于 `--` 之后透传。

> **任何偏离官方默认值的参数都会在终端和 `results/performance.txt` 中被显式标注**
> （标注形式 `<== 非官方默认（官方 X）`），并且"复现方式"一栏会打印本次实际使用的完整命令，
> 以免自定义参数跑出的数字被误当成官方口径的成绩。

官方评测协议（即不带任何参数时的口径）：

- `warmup = 200`，`repeat = 500`，取 **median**；
- 每次迭代用 `time.perf_counter()` 包住整个 `forward()` 后 `sync_devices()`——**host 开销全额计入**；
- 正确性判据 `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`；
- `speedup = v0_ms / v1_ms`；默认 `seed = 42`。

预期输出：

```
PASS accuracy; v0=1.252750 ms, v1=0.144385 ms, speedup=8.676x
Summary: 1 passed, 0 failed, 1 total.
```

> 说明：绝对耗时会随机器负载、CANN 版本与进程间漂移波动（实测 v0 baseline 跨进程漂移可达 ±11%），
> 加速比一般稳定在 **8.5x ~ 9.9x** 区间。若要横向比较不同版本，建议固定 baseline 后再比。

### 3.5 可选：多随机种子精度回归

官方评测只用一个种子。为验证结果不是「刚好在 seed=42 通过」，附带了多种子回归脚本：

```bash
python3 run_auto_bench_mulseed.py                     # 从 seed 0 开始无限递增，Ctrl+C 停止
python3 run_auto_bench_mulseed.py --start-seed 1000000
python3 run_auto_bench_mulseed.py --quiet --resume    # 断点续跑
```

该脚本**复用官方 `auto_bench.py` 的模块加载、设备选择、输入克隆和 `compare_values`**，
因此 PASS/FAIL 判定与官方完全一致；每个种子只做精度 forward，不跑性能循环。
逐种子结果追加到 `results/multiseed_accuracy_cases.jsonl`，
汇总每 100 个种子原子写入 `results/multiseed_accuracy.json`，支持 `--resume` 断点续跑。

### 3.7 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `找不到 CANN` | 先 `source .../set_env.sh`，或设置 `ASCEND_HOME_PATH` |
| `找不到同时装有 torch 和 torch_npu 的 python` | 编译用 `PYTHON=/path/to/python3 bash build.sh`；评测用 `bash run_auto_bench.sh --python /path/to/python3` |
| `未找到自定义算子: ...so` | 先执行 `bash build.sh` |
| `this build is specialised for [1,1024,4,4]` | 本实现按赛题规模特化，输入形状必须是 `[1,1024,4,4]`（见 §4.3） |
| auto_bench.py 下载失败 | 手动下载后 `bash run_auto_bench.sh --auto-bench /path/to/auto_bench.py`；离线环境可加 `--no-download` 以免误联网 |
| `错误: 未知参数 xxx` | 参数名须与 `--help` 所列逐字一致（无别名、无位置参数）；未包装的官方参数请放在 `--` 之后透传 |

---

## 四、原创说明

### 4.1 原创性声明

本作品的**全部算子实现代码与工程脚本均为本队原创编写**，未复制、改写或参考任何他人的赛题提交、
开源 Sinkhorn 昇腾算子实现或第三方 kernel 代码。具体包括：

| 文件 | 原创性 |
| --- | --- |
| `op_kernel/sinkhorn_normalize_kernel.asc` | **完全原创**。padded-AoS 布局、−1000 哨兵消除 padding 污染、block-stride 列归一化、常量化 eps 掩码、跨流水线事件设计，均为本队自行设计与实测调优 |
| `op_kernel/sinkhorn_normalize_tiling.h` | **完全原创** |
| `op_extension/*.cpp`、`ops.h` | **完全原创**。torch 自定义算子注册遵循 PyTorch/torch_npu 的公开标准写法（`TORCH_LIBRARY_FRAGMENT` / `TORCH_LIBRARY_IMPL`），实现体为原创 |
| `CMakeLists.txt`、`build.sh`、`run_auto_bench.sh`、`run_auto_bench_mulseed.py` | **完全原创** |
| `model_new.py` | **原创**（仅 `get_inputs` / `get_init_inputs` 与赛题保持一致，为评测所必需） |
| `model_ref.py` | **赛题提供的参考实现，未做任何修改**，仅作为 v0 基准 |

### 4.2 第三方组件与引用

本作品依赖但未包含、未修改的第三方内容：

- **CANN / AscendC 编程接口**（`kernel_operator.h` 提供的 `DataCopyPad`、`BlockReduceSum`、
  `BlockReduceMax`、`Brcb`、`Duplicate`、`Exp`、`Div` 等原语）：华为官方 SDK，按公开文档使用；
- **PyTorch 与 torch_npu**：官方发行版，按公开 API 使用；
- **官方评测脚本 `auto_bench.py`**：来自
  `DeepLink-org/DLBlas`（`benchmarks/ks/auto_bench.py`），由 `run_auto_bench.sh` 在评测时下载到 `/tmp`，
  **本仓库不包含其副本，也不对其做任何修改**，以保证评测口径与官方完全一致。

源文件头部保留了 `SPDX-License-Identifier: CANN Open Software License Agreement Version 2.0` 许可声明。
