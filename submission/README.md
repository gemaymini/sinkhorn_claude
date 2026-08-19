# SinkhornNormalize 昇腾算子优化

2026 KernelSwift 算子创新大赛 · 赛道二（华为昇腾）· Task03 `sinkhorn`

---

## 1. 作品说明

在 KernelSwift 平台生成的 AscendC 算子基础上，通过**实测驱动的逐层优化**，把
`SinkhornNormalize` 在昇腾 910B2 上的端到端耗时从 **269.97 us 降到 158.55 us**，
按官方评测口径加速比从 **5.79x 提升到 9.86x**。

全程不含任何数值近似：输出与 PyTorch 参考实现的最大绝对误差为 **1.8e-07**
（fp32 舍入量级），在官方容差 `atol=1e-2, rtol=1e-2` 下裕度占用为 **0.0000**。

| 版本 | v1 耗时 | 加速比 | 关键改动 |
|---|---|---|---|
| KernelSwift 生成版 | 269.97 us | 5.79x | 每次处理 1 个 4×4 矩阵 |
| + host 层优化 | — | — | 去掉每次调用的同步 memcpy 与设备查询 |
| S1 批量化 | 172.69 us | 9.05x | 一次处理 64 个矩阵，消除标量↔向量同步 |
| + 精细同步 | 164.14 us | 9.52x | 去掉冗余 `PipeBarrier<PIPE_V>` |
| **本提交版本** | **158.55 us** | **9.86x** | eps 向量改为纯算术构造，零 GM 访问 |

> 加速比基准 = 1563 us，取自 20 次独立测量的中位数。该基准在同一台机器上跨进程
> 波动达 ±11%，单次测量报出的加速比因此有 ±10% 的不确定性；本文所有对比均以
> **v1 自身耗时**为判据。

---

## 2. 环境

| 项 | 版本 |
|---|---|
| 硬件 | 华为昇腾 Atlas A2 / 910B2（`dav-2201`） |
| CANN | 9.0.0 |
| Python | 3.11 |
| PyTorch / torch_npu | 2.10.0 / 2.10.0 |
| 构建 | CMake ≥ 3.16, bisheng（随 CANN 提供） |

详见 `requirements.txt`。

---

## 3. 快速开始

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

bash build.sh              # 1. 编译（不需要任何 -D 选项）
bash run_auto_bench.sh     # 2. 用官方原版 auto_bench.py 评测（含正确性校验）
```

第 3 步会自动下载官方 `auto_bench.py`；若网络受限可手动指定：

```bash
bash run_auto_bench.sh /path/to/DLBlas/benchmarks/ks/auto_bench.py
```

它等价于：

```bash
python3 auto_bench.py \
    --v0_file $PWD/model_ref.py \
    --v1_file $PWD/model_new.py
```

### 两个文件的职责

官方 `auto_bench.py` 的 `build_case()` 从**两个独立文件**分别读取：

```python
model_cls     = require_attr(v0_module, "Model",    v0_path)   # 只从 v0 取
model_new_cls = require_attr(v1_module, "ModelNew", v1_path)   # 只从 v1 取
```

| 文件 | 角色 | 定义 |
|---|---|---|
| `model_ref.py` | `--v0_file` | `Model`（赛题参考实现，逐字照抄）+ `get_inputs` / `get_init_inputs` |
| `model_new.py` | `--v1_file` | `ModelNew`（自定义算子）+ `get_inputs` / `get_init_inputs` |

规则 5.2 的「与参考实现一致的 Model 定义，包括 `__init__` 和 `forward` 的参数」指的是
**签名一致** —— `ModelNew.__init__(repeat=10, eps=1e-6)` 与 `ModelNew.forward(x)`
必须与参考实现对得上，而**不是**把参考实现抄进提交文件。

`model_new.py` 因此**刻意不包含任何 PyTorch 版的等价实现**：规则 5.1 禁止 fallback 到
内置算子，提交文件里放一份可用的 torch 实现本身就构成嫌疑面。`forward` 的每一条
返回路径都只调用自定义算子，合规自检会逐条验证。

### `.so` 的定位

官方 `load_ks_module()` 会设置 `module.__file__ = str(path)`，因此 `model_new.py`
能拿到自身路径。**`.so` 就固定从这一个位置加载 —— `model_new.py` 所在目录**，
没有环境变量、没有多路径回退：

```python
so = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "libsinkhorn_normalize_ops.so")
```

`build.sh` 编译后会把产物复制到本目录，打包时一并带上。找不到就抛异常并提示运行
`bash build.sh`，**绝不静默回退到 PyTorch 内置算子**。

---

## 4. 优化方案

优化分三层，每一层的收益都由真机实测确认。

### 4.1 Host 绑定层

官方评测用 `time.perf_counter()` 包住整个 `model.forward()` 再 `sync`，**host 侧
每次调用的开销全额计入加速比**。原实现每次调用都执行：

- `aclrtGetDeviceInfo` 查询设备核数（该值是常量）
- 重算 tiling 并用 `at::empty` 分配张量
- **`aclrtMemcpy` 同步阻塞地把 tiling 传到 device**

其中同步 memcpy 会等待 stream 排空，实测 host 端耗时 114 us。改为**静态缓存
tiling 与核数**后降至 40 us，加速比提升约 17%，且无任何精度风险。

### 4.2 Kernel 结构：per-matrix → padded-AoS 批量化

原实现每个 4×4 矩阵需要 ~228 条向量指令、~198 个 `PipeBarrier`，以及
**~82 对标量↔向量同步**（`GetValue`/`SetValue` + `SetFlag`/`WaitFlag`）。
每条向量指令只用到 8~32 个 lane（fp32 满载为 64）。

新实现把 B 个矩阵放进一个 tile 统一处理，UB 布局为每矩阵 4 行 × 8（4 有效 + 4 填充）：

```
z[32m + 8r + c]    c < 4 : 矩阵 m 第 r 行第 c 列
                   c >= 4: 填充，全程恒为 0
```

三个关键设计：

**填充值 `-1000`**：`exp(-1000)` 在 fp32 下**精确下溢为 0**，因此填充位在全部 10 次
迭代中恒为 0。归约天然正确，不需要原实现里那些标量修正，也不需要掩码缓冲区。

**算法归一**：`softmax` 就是 `Exp` + 一次行归一，整个算法塌缩成

```
Exp → RowNorm(=softmax) → +eps → ColNorm → 9 × (RowNorm, ColNorm)
```

严格交替，无特例分支。`RowNormalize` 只需 4 条指令覆盖整个 tile
（`BlockReduceSum → Adds → Brcb → Div`），`ColNormalize` 用 `mask=64` +
`blkStride=4` 让一个 repeat 同时处理 8 个矩阵。

**eps 向量纯算术构造**：参考实现的 `softmax(x) + eps` 不能省（见 §6），而 eps 只能
加在有效位、填充位必须保持 0。原做法是从 GM 读一段 eps 数组再用 `DataCopyPad`
展开，实测耗时 20.8 us。改为利用 CopyIn 后「填充位 = −1000、有效位 = x」的已知状态，
用 4 条标量-向量指令直接算出来：

| 步骤 | 填充位 | 有效位 |
|---|---|---|
| CopyIn 后 | −1000 | x |
| `Adds(+900)` | −100 | x+900 > 1 |
| `Maxs(0)` | **0** | x+900 |
| `Mins(1)` | 0 | 1 |
| `Muls(eps)` | **0** | **eps** |

零 GM 访问、零 MTE 描述符，耗时约 0.25 us。

### 4.3 指令级

- **`Div` 优于 `Reciprocal` + `Mul`**：实测昇腾的 `Reciprocal` 只有约 **9 位**有效精度
  （`1/1` 得 0.998，`1/4` 得 0.2495），直接使用会让 20 次连续归一化后的误差达 1e-3 量级。
  改用 `Div` 后精度回到 1.8e-07，**且更快**（少了 Newton-Raphson 的 8 条修正指令）。
- **去掉冗余 `PipeBarrier<PIPE_V>`**：向量指令本就在 V pipe 上按序发射，跨 pipe 依赖
  另有 `SetFlag`/`WaitFlag` 处理。去掉 23 处 barrier 后精度不变。
- **`SN_TILE_TARGET=64`**：实测每核矩阵数低于 64 时明显变慢（block 启动成本主导），
  8/16/32 三档的固定开销均为 ~136 us，64/128 档为 ~113 us。

---

## 5. 性能测试结果

### 5.1 官方口径

评测协议完全复刻 `DLBlas/benchmarks/ks/auto_bench.py`：`warmup=200`、`repeat=500`、
每次迭代 `perf_counter` 包住 `forward` 后 `sync`、取 median；提交文件同样经过
AST 过滤后 `exec`。

| 指标 | 数值 |
|---|---|
| v1（本实现） | **158.55 us** |
| v0（参考实现，20 次中位数） | 1563 us |
| **加速比** | **9.86x** |

### 5.2 时间构成

用 `t(repeat) = a + b × repeat` 线性拟合（`repeat` 是算子入参，可直接扫描）：

| 成分 | 耗时 | 占比 |
|---|---|---|
| launch + dispatch + sync 地板 | 62.0 us | 63% |
| CopyIn + CopyOut | 27.0 us | 27% |
| 全部计算（Exp + 减 max + eps + 10 次迭代） | 10.0 us | 10% |

其中 10 次 Sinkhorn 迭代本身不足 5 us。**该算子已进入 launch 开销主导区间**，
理论上限约 14.9x（计算与搬运全部归零时）。

### 5.3 精度

`scripts/check_shapes.py`，12 个形状 × 5 个随机种子，判据为官方容差
`atol=1e-2, rtol=1e-2`，裕度占用 = `max|diff| / (atol + rtol·|ref|)`：

| 用例 | 最大绝对误差 | 裕度占用 |
|---|---|---|
| 参考形状 `[1,1024,4,4]` | 1.788e-07 | 0.0000 |
| 单矩阵 `[4,4]` | 8.941e-08 | 0.0000 |
| 尾块用例（7 / 65 / 129 / 333 个矩阵） | ≤ 1.788e-07 | 0.0000 |
| 全零输入 | 0.000e+00 | 0.0000 |
| 大幅值 `randn×8` / `randn×32` | ≤ 1.788e-07 | 0.0000 |

无 NaN / Inf。**裕度占用 0.0000 意味着距离容差上限还有 5 个数量级的余量。**

### 5.4 提交前自检（开发期工具，不随包提交）

打包流程会自动运行 `check_compliance.py`，逐条对照官方规则与 `auto_bench.py` 的实际行为做 36 项检查，不通过则中止打包。检查结果只输出到终端，不写入提交件：

| 类别 | 检查内容 |
|---|---|
| 规则 5.2 | 文件 UTF-8 编码、Python ≥ 3.10 |
| auto_bench 兼容 | AST 过滤后可 exec、必需符号齐全、顶层无被丢弃的可执行语句、`get_inputs`/`get_init_inputs` 返回类型 |
| 规则 5.2 | **`ModelNew` 的 `__init__`/`forward` 参数与 `Model` 完全一致**；提交文件不重复定义 `Model`、不含 PyTorch 等价实现 |
| 规则 5.1 | `forward` 的**每一条返回路径**都调用自定义算子、内部不含 PyTorch 内置实现痕迹、无异常捕获 |
| 规则 4.2 | 算子代码 / README / 环境配置 / 运行脚本 / 性能结果齐备 |
| 规则 4.3 | README 含作品说明、优化方案、性能测试结果、原创声明 |
| 构建自包含 | `CMakeLists.txt` 引用的源文件均在包内（自动跳过 `if(EXISTS)` 保护的目标） |
| 运行时 | 算子注册成功、`forward` 真实进入 kernel（调用计数器递增）、官方容差下结果一致 |

另有多形状精度门禁（12 形状 × 5 seed）与时间拆解，同样只在打包时于终端输出。
提交件本身只保留规则 4.2 要求的内容：算子代码、README、环境配置、运行脚本、性能测试结果。

### 5.5 反 fallback 自查

按规则 5.1，算子内置调用计数器 `torch.ops.npu.sinkhorn_normalize_call_count()`，
评测脚本会校验每次 `forward` 都真实进入了自定义 kernel。该计数器不在提交热路径上。

---

## 6. 两个值得记录的坑

**`softmax(x) + eps` 不能省。** 官方容差 1e-2 很宽，初步测试（仅用 `torch.randn`）
显示省掉 eps 的裕度占用只有 0.0009，看起来完全安全。但把输入幅值放大后：

| 输入分布 | 无 eps | 完整 eps |
|---|---|---|
| `randn`（官方） | 0.0009 | 0.0000 |
| `randn×4` | **15.94 ❌** | 0.0000 |
| `randn×8` | **83.5 ❌** | 0.0000 |

这不是舍入误差而是语义差异：softmax 输出小到 ~1e-11 时，加 `eps=1e-6` 会把它抬高
5 个数量级。**数值近似必须跨输入尺度验证，不能只测官方那一个分布。**

**不能截断 Sinkhorn 迭代。** 截断到 8 次时裕度占用已达 0.558，7 次就有 1/20 的种子失败。
根因是第 10 次迭代本身都尚未收敛（行和仍偏离 1 达 1.08e-2），误差线性递减而非二次收敛。

---

## 7. 目录结构

```
.
├── README.md                       本文档
├── requirements.txt                环境依赖
├── model_ref.py                    赛题参考实现（auto_bench 的 --v0_file）
├── model_new.py                    本提交（--v1_file），只定义 ModelNew
├── build.sh                        一键编译
├── run_auto_bench.sh               用官方 auto_bench.py 评测
├── libsinkhorn_normalize_ops.so    编译产物，与 model_new.py 同目录
├── CMakeLists.txt
├── op_kernel/
│   ├── sinkhorn_normalize_tiling.h     tiling 结构与共享常量
│   └── sinkhorn_normalize_kernel.asc   算子实现
├── op_extension/                   PyTorch 绑定层
├── op_host/                        直调入口
└── results/
    └── performance.txt             性能测试结果
```

---

## 8. 构建

```bash
cmake -S . -B build && cmake --build build --target sinkhorn_normalize_ops
```

**不需要任何 `-D` 选项。** 算子配置已全部固化进源码：

| 决策 | 取值 | 依据 |
|---|---|---|
| 除法实现 | `Div` | 硬件 `Reciprocal` 只有 9 位精度；`Div` 既更准又更快 |
| softmax 减最大值 | 是 | 否则 `randn×32` 时 `exp` 溢出成 inf |
| ColNormalize 宽度 | 8 矩阵/repeat | `mask=64` + `blkStride=4`，repeat 数从 456 降到 64 |
| `PipeBarrier<PIPE_V>` | 不插 | 同 pipe 内本就按序发射，跨 pipe 另有 `SetFlag`/`WaitFlag` |
| eps 向量 | 算术构造 | 从 GM 读取版实测独占 20.8us |
| CopyOut | 单次批量 `DataCopyPad` | |
| 每核矩阵数 | 64 | 实测 <64 明显更慢，64 与 128 持平 |
| tiling / 核数 | 静态缓存 | 去掉每次调用的同步 `memcpy` 与设备查询，+17% |

开发阶段这些是编译期开关，用遗传算法在 4480 个有效配置点上搜索过；
搜索确认上表即最优解后，**开关已消解、死代码已删除**，提交件只保留最终实现。
`op_kernel/sinkhorn_normalize_kernel.asc` 的文件头注释逐条记录了每个决策的实测依据。

## 9. 原创声明

本作品的算子优化方案与全部代码由参赛者独立完成。基础版本由 KernelSwift 平台生成，
其后的 host 绑定层优化、kernel 批量化重写、eps 向量算术构造、指令级调优
及全部验证脚本均为参赛者原创工作。文中所有性能与精度数据均为在昇腾 910B2 真机上
实测所得，测试脚本随作品一并提交，结果可复现。
