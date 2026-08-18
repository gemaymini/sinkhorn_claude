# SinkhornNormalize @ 昇腾 910B A2 —— 进化搜索调优实验方案

> 赛题：2026 KernelSwift 算子创新大赛 赛道二 Task03 `sinkhorn`
> 现状：`result.json` 记录 `speedup=4.622`（237.48us vs 1097.66us，自测 event 口径）
> **本文档已按官方评测脚本 `DLBlas/benchmarks/ks/auto_bench.py` 与本地数值实测结果校准**

---

## 0. 结论先行

1. **官方容差是 `atol=1e-2, rtol=1e-2`**，比仓库自测用的 `1e-4/1e-5` 宽两三个数量级。本地 20 seed 实测（`experiments/numeric_feasibility.py`）证实：**省掉全部 eps、softmax 免减 max、全程 fp16 计算，三者叠加后裕度只占 6.1%**，可放心启用。
2. **官方计时是 `perf_counter` 包住整个 `model.forward()` 再 `sync`，warmup 200 / repeat 500 / 取 median**。这意味着 **Python 层、torch dispatch、host 端 ACL 调用、kernel launch、stream sync 全部计入**。
3. **（P0 已验证）** 因此第一块肉在 host 绑定层：当前 `sinkhorn_normalize_torch()` 每次调用都做一次 `aclrtGetDeviceInfo` 查核数、两次张量分配、以及**一次同步阻塞的 `aclrtMemcpy` 传 tiling**。这几项是纯固定开销、零风险可去除，优先级高于任何 kernel 微调。
4. kernel 本身是**小粒度指令 + 标量↔向量同步**延迟瓶颈：每个 4×4 矩阵 ~228 条向量指令、~198 个 `PipeBarrier`、**~82 对标量-向量同步**，而每条向量指令只用到 8~32 个 lane（fp32 满载是 64）。
5. **现实目标：15~21x（硬上限 22.7x）**。已由 P0 真机实测确定，见 §2.3。
6. **明确不建议截断 Sinkhorn 迭代**（见 §3），收益小、风险是直接判 0 分。

---

## 1. 官方评测口径（一切设计的基准）

来自 `auto_bench.py`：

```python
def time_forward(model, inputs, seed, warmup, repeat):
    def one_call():
        with torch.no_grad():
            model.forward(*inputs)
    for _ in range(warmup):
        one_call()
    sync_devices()
    samples = []
    for _ in range(repeat):
        set_seed(seed)
        start = time.perf_counter()
        one_call()
        sync_devices()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)
```

| 项 | 值 | 对方案的影响 |
|---|---|---|
| 容差 | `atol=1e-2, rtol=1e-2, equal_nan=True` | 打开 fp16 / 免 eps / 免 max 的空间 |
| warmup / repeat | 200 / 500，取 median | 抖动被 median 吸收；但**每次迭代都 sync**，固定开销无法摊薄 |
| 计时范围 | 整个 `forward()` + `sync` | **host 侧开销全额计入**，见 §2.2 |
| speedup | `v0_ms / v1_ms` | v0 = 赛题 reference `Model`，v1 = 提交的 `ModelNew` |
| seed | 42 | 但**不要只用 42 验精度**，官方可能换 |

### 1.1 提交形态的硬约束（AST 过滤）

`auto_bench.py` 用 AST 过滤后 `exec` 提交文件，**只保留** `Import / ImportFrom / ClassDef / FunctionDef / AsyncFunctionDef / 字面量 Assign`。

> **顶层的 `torch.ops.load_library("...")` 会被静默删掉！**

必须把 `.so` 加载放进 `ModelNew.__init__` 里，或放进一个被 `import` 的独立模块（`import` 语句会保留，其副作用会执行）。
**这一条必须在 P0 阶段就真机验证**，否则提交上去直接跑不起来。

另外规则 5.1 明令禁止 fallback 到 PyTorch 内置算子，所以门禁里要加一项：**确认自定义 kernel 真的被执行**（profiling 里能看到 kernel，或加计数器）。

---

## 2. 现状诊断

### 2.1 Kernel 侧（`sinkhorn_normalize_kernel.asc`）

每矩阵动态指令预算（repeat=10）：

| 类别 | 次数/矩阵 | 主要来源 |
|---|---|---|
| 向量指令 | ~228 | `ColNormalize`×10 = 90，`RowNormalize`×9 = 108 |
| `PipeBarrier<PIPE_V>` | ~198 | 几乎每条向量指令后都插，流水线完全串行 |
| **标量↔向量同步对** | **~82** | `RowNormalize` 每行 2 对 × 4 行 × 9 次 = 72 对 |
| 标量读写 | ~136 | `GetValue`/`SetValue`（含 `ZeroPadding` 的 16 次） |
| MTE 搬运 | 8 | CopyIn/CopyOut 各 4 次，**每次仅 16 字节** |

每核承载 1024/C ≈ 22 个矩阵，即 ~5000 条向量指令 + ~4400 barrier + ~1800 对标量同步，全部串行。
估算 ~260k cycle @ ~1.8GHz ≈ 145us，其中**标量同步占 ~55%**。

| # | 浪费点 | 位置 |
|---|---|---|
| W1 | `GetValue`→标量算 `sum_4 = sum_8 - 4*pad`→`SetValue` 回写，打断向量流水 82 次/矩阵 | `SoftmaxRows` / `RowNormalize` |
| W2 | 一次只处理 1 个 4×4 矩阵，lane 利用率 12.5%~50% | `ProcessOneMatrix` |
| W3 | `PipeBarrier` 无差别全插，无 double buffering | 全文件 |
| W4 | CopyIn/CopyOut 拆成 4 次 16B `DataCopyPad` | `CopyIn`/`CopyOut` |
| W5 | 用长延迟的 `Div`；`ZeroPadding` 用 16 次标量 `SetValue` | `ColNormalize` / `ZeroPadding` |

### 2.2 Host 侧（`sinkhorn_normalize_torch.cpp`）—— 被低估的大头

每次 `forward()` 调用都会执行：

| # | 操作 | 问题 |
|---|---|---|
| H1 | `aclrtGetDevice` + `aclrtGetDeviceInfo(VECTOR_CORE_NUM)` | 核数是常量，**每次都查** |
| H2 | tiling 的 6 个字段全部重算 | 形状不变时是常量 |
| H3 | `at::empty(sizeof(TilingData), kByte)` | 每次分配一个小张量 |
| H4 | **`aclrtMemcpy(...)` 同步 H2D** | **阻塞调用，会等 stream 排空**；这是最贵的一项 |
| H5 | `at::empty_like(x)` | 输出分配，难免，但可考虑复用 |
| H6 | `TORCH_LIBRARY` dispatch | boxing/unboxing 开销 |

**H4 尤其致命**：同步 `aclrtMemcpy` 在每次迭代里强制一次 host-device 往返，量级 10~50us。而官方计时口径把它全额计入。

修法（零精度风险）：把 tiling 改成 kernel 的**标量入参**彻底去掉 tiling 张量；或 `static` 缓存 tiling 张量 + `aclrtMemcpyAsync` 走 stream；或形状固定时编译期常量化。

> **P0 阶段必须先用 msprof + host 打点把 kernel 时间 / host 时间的比例测出来。** 如果 host 占了 100us+，那么先做 §2.2 就能白拿一大截加速比，且完全不用碰 kernel。

---

## 2.3 P0 真机实测结果（910B2，CANN 9.0.0，已完成）

`bash scripts/run_p0.sh` 的产出。**所有后续目标都以这组数为准。**

### host 侧优化的实际收益

| 指标 | `orig`（每次同步 memcpy + 查核数） | `opt`（静态缓存） | 变化 |
|---|---|---|---|
| M2 host 端耗时（不 sync） | 114.28us | **39.78us** | **−74.5us（−65%）** |
| M1 裸算子 + sync | 301.41us | 237.19us | −64.2us |
| **官方口径 speedup** | **3.834x** | **4.629x** | **+20.7%** |

> 那个每次调用的同步 `aclrtMemcpy` 一项就值 20% 的加速比。零精度风险。

### 时间地板（决定一切目标的上限）

| 测量项 | 耗时 |
|---|---|
| M3 `torch.npu.synchronize()` 空转 | 16.8us |
| M4 `torch.empty_like` + sync | 31.6us（推出分配器本身 ≈ 15us） |
| **M5 单个最简 NPU 算子往返 + sync** | **55.3us ← 硬地板** |
| M0 参考实现 forward + sync | 1240~1307us（**波动 5%，必须固定成常量**） |

**`speedup` 理论上限 = M0 / M5 ≈ 1250 / 55.3 ≈ 22.7x。** 任何 kernel 优化都无法突破。
这比方案初稿的估计（30~40x）低，原因是这台机器的 host-device 往返开销偏高
（光 sync 空转就 16.8us）。

### 当前时间构成（`opt` 变体）

```
官方口径 v1 = 267.88us
  ├─ Python wrapper (ModelNew.forward)   ≈ 30us   ← 待 M6 确认，可能含进程漂移
  ├─ host CPU (dispatch + empty_like +
  │            launch)                    ≈ 40us   ← 其中 empty_like 占 ~15us，省不掉
  └─ kernel 净计算 + launch/sync 残差      ≈ 182us  ← 唯一的大头
```

### kernel 优化目标推演

| kernel 净时间 | 预期 M1 | 预期 speedup | 备注 |
|---|---|---|---|
| 182us | 237us | 4.63x | 现状 |
| 60us | ~115us | ~10.8x | |
| **30us** | **~85us** | **~14.7x** | **性价比拐点** |
| 15us | ~70us | ~17.8x | |
| 5us | ~60us | ~20.8x | |
| 0（理论） | 55us | 22.7x | 硬上限 |

**边际收益递减非常明显：压到 30us 就能拿到 ~15x（3.2 倍提升），再往下拼到 5us 也只多拿 40%。**
这直接改变了 P1 的路线选择 —— 见 §11。

---

## 3. 数值近似实测结论（已完成，`experiments/numeric_feasibility.py`）

20 个随机种子，shape `(1,1024,4,4)`，与 fp32 reference 比对，判据 `atol=1e-2, rtol=1e-2`。
「裕度占用」= `max|diff| / (atol + rtol·|ref|)`，**<1 通过，越小越安全**。

| 变体 | 通过 | 最大绝对误差 | 裕度占用 | 结论 |
|---|---|---|---|---|
| 省掉全部 eps | 20/20 | 1.18e-05 | **0.001** | ✅ 启用 |
| softmax 免减 max | 20/20 | 1.79e-07 | **0.000** | ✅ 启用 |
| 上面两项叠加 | 20/20 | 1.18e-05 | **0.001** | ✅ 启用 |
| **fp16 全程计算** | 20/20 | 9.29e-04 | **0.061** | ✅ **启用（16x 裕度）** |
| bf16 全程计算 | 20/20 | 6.54e-03 | 0.429 | ⚠️ 可用但裕度仅 2.3x，作 fp16 的备选 |
| fp16 + 免 eps + 免 max | 20/20 | 9.32e-04 | **0.061** | ✅ **推荐组合** |
| 截断到 9 次迭代 | 20/20 | 3.94e-03 | 0.234 | ❌ 见下 |
| 截断到 8 次迭代 | 20/20 | 9.39e-03 | 0.558 | ❌ 见下 |
| 截断到 7 次迭代 | **19/20** | 1.71e-02 | 1.012 | ❌ 已翻车 |
| 截断到 6 次及以下 | ≤18/20 | ≥2.80e-02 | ≥1.657 | ❌ |

### 为什么明确不建议截断迭代

Sinkhorn 收敛轨迹（seed=42，第 k 次 vs 第 10 次）：

| k | max\|y_k − y_10\| | 行和偏离 1 | 裕度占用 |
|---|---|---|---|
| 6 | 1.92e-02 | 4.72e-02 | 1.167 |
| 7 | 1.12e-02 | 3.26e-02 | 0.716 |
| 8 | 5.96e-03 | 2.26e-02 | 0.392 |
| 9 | 2.41e-03 | 1.57e-02 | 0.162 |
| 10 | 0 | **1.08e-02** | 0 |

**第 10 次迭代本身都还没收敛**（行和仍偏离 1 达 1.08e-2），误差是线性递减而非二次收敛，所以不存在「反正已收敛、砍几次无所谓」的余地。截断 2 次只省 20% 的向量指令（而向量指令重写后总共才 ~5us），却把裕度从 0.06 推到 0.56，换个 seed 或官方收紧容差就直接判 0 分。**收益/风险比极差，不纳入基因空间。**

### fp16 的额外好处（不只是精度问题）

- 一个 vector repeat 处理 **128 个 fp16** vs 64 个 fp32 → 计算吞吐翻倍
- UB 占用减半 → 单 tile 能装下更多矩阵 → 循环次数减半
- 注意：输入输出仍是 fp32，需在 kernel 内 `Cast`，搬运量不变

---

## 4. 优化机会清单（→ 基因空间的来源）

| 层次 | 机会 | 预期 | 风险 |
|---|---|---|---|
| **Host** | 去掉每次调用的 `aclrtGetDeviceInfo` / tiling 同步 memcpy / tiling 张量分配 | **大（可能 50~150us）** | 无 |
| **Host** | tiling 编译期常量化或走 kernel 标量入参 | 中 | 无 |
| **Host** | 绕开 `TORCH_LIBRARY` boxing，改 pybind 直调 | 小~中 | 无 |
| **Python** | `ModelNew.forward` 里零多余操作（无 `.contiguous()`、无 shape 检查） | 小 | 无 |
| **Kernel 结构** | SoA 平面布局：把 `(M,4,4)` 重排为 16 个长度 M 的平面，行/列和退化成 elementwise `Add`，**归约/广播/mask/标量全部消失** | **大（kernel 145us → ~5us）** | 无（等价变换） |
| **Kernel 结构** | 批量化：单次处理 B 个矩阵而非 1 个 | 大 | 无 |
| **Kernel 指令** | `Div` → `Reciprocal` + `Mul`；消除冗余 `PipeBarrier`；double buffering；合并小 `DataCopyPad` | 中 | 无 |
| **数值** | fp16 计算 + 免 eps + 免 max | 中 | 低（已实测，裕度 16x） |
| **调度** | 用核数不必打满（小 shape 下满核未必最快）；tile 大小 | 中 | 无 |

### SoA 平面布局展开

重排为 16 个平面 `P[r][c]`（每个长度 M）后：
- 行和 = `P[r][0]+P[r][1]+P[r][2]+P[r][3]` → 3 条 `Add`，无归约、无广播、无 mask、无标量
- 归一化 = 1 条 `Reciprocal` + 4 条 `Mul`
- 每次迭代：行归一 4×(3 `Add` + 1 `Reciprocal` + 4 `Mul`) = 32 条，列归一同样 32 条
- **10 次迭代共 640 条向量指令，0 次标量同步**，加转置进出 ~50 条

转置实现是关键子问题，作为基因交给搜索：`GATHERMASK`（按 `p ≡ k (mod 16)` 的 pattern 抽取）/ `GATHER`（偏移表）/ `DATACOPY_STRIDE`（32B 粒度分块 + 块内重排）/ `NONE`（走 AoS 批量化路线兜底）。
**这几种 API 的真实吞吐必须真机实测，不能按文档想当然** —— 这是 P1 的核心任务。

---

## 5. 三条 Track

| Track | 搜索对象 | 编译通过率预期 | 定位 |
|---|---|---|---|
| **A. 参数化模板 GA**（主力） | 模板 holes 的离散取值 | >95% | 稳定产出、可归因、易写报告 |
| **B. 代码级 GA** | `.asc` 源码语句序列上的变异算子 | 40~70% | 挖模板没覆盖的微观优化 |
| **C. LLM 引导 GA** | 变异/交叉由 LLM 生成 | 60~80% | 跨结构跳跃；与 KernelSwift 形成闭环，是答辩亮点 |

执行顺序：**P0 host 侧白拿收益 → P1 人工写 2~3 个骨架并真机验证转置可行性 → Track A 全量搜 → Track B 精修 → Track C 做对照与亮点。**

---

## 6. Track A：参数化模板 GA

### 6.1 基因型（20 位，混合类型）

**Host / 提交层（新增，优先级最高）**

| # | 基因 | 取值域 |
|---|---|---|
| h1 | `tiling_mode` | `{PER_CALL_MEMCPY(现状), CACHED_STATIC, ASYNC_MEMCPY, KERNEL_SCALAR_ARG, COMPILE_CONST}` |
| h2 | `core_query` | `{PER_CALL, CACHED_ONCE}` |
| h3 | `dispatch` | `{TORCH_LIBRARY, PYBIND_DIRECT}` |

**Kernel 结构层**

| # | 基因 | 取值域 |
|---|---|---|
| g1 | `skeleton` | `{S0_CURRENT, S1_BATCH_AOS, S2_SOA_PLANE, S3_SOA_FUSED}` ← **结构基因，决定模板与岛屿** |
| g2 | `mats_per_tile` | `{1,2,4,8,16,32,64,128,256}` |
| g3 | `block_num_frac` | `{1/8,1/6,1/4,1/3,1/2,2/3,1} × C` |
| g4 | `queue_depth` | `{1,2}` |
| g5 | `transpose_impl` | `{NONE, GATHERMASK, GATHER, DATACOPY_STRIDE}` |
| g6 | `copy_in_mode` / g7 `copy_out_mode` | `{PAD_PER_ROW, SINGLE_BULK, STRIDED_BLK}` |
| g8 | `compute_dtype` | `{FP32, FP16, BF16}` ← **实测已验证 FP16 安全** |
| g9 | `div_impl` | `{DIV, RECIP_MUL, RECIP_NR1}` |
| g10 | `softmax_max` | `{REDUCEMAX, SKIP}` ← 实测裕度 0.000 |
| g11 | `eps_mode` | `{EXACT, DROP}` ← 实测裕度 0.001 |
| g12 | `sync_mode` | `{BARRIER_ALL, BARRIER_MIN, SET_WAIT_FLAG}` |
| g13 | `scalar_free` | `{0,1}` |
| g14 | `unroll_iter` | `{1,2,5,10}` |
| g15 | `rowcol_fusion` | `{SEPARATE, FUSED}` |
| g16 | `ub_reuse` | `{SEPARATE, INPLACE}` |
| g17 | `tail_mode` | `{MASKED, SEPARATE_LOOP}` |
| g18 | `compile_opt` | `{O2, O3, O3_FASTMATH}` |

> **`iter_trunc` 基因已按 §3 的实测结果剔除。**

搜索空间 ≈ 10⁹ 量级。

### 6.2 约束修复

交叉/变异后跑 `repair(genome)`：
- 按 `skeleton` 屏蔽无效基因（置哨兵值，**不参与交叉、不参与哈希**）
- `mats_per_tile × 元素宽度 × queue_depth ≤ UB 容量`（fp16 时上限翻倍）
- `block_num_frac × C ≤ totalMats`

**去重哈希用「渲染后的源码 + 编译选项」，不是基因组** —— 不同基因组常渲染出同一份代码，实测这类框架能省 20~35% 评估。

### 6.3 适应度

```
evaluate(genome):
    src = render(genome)                       # kernel.asc + host.cpp + model_new.py
    key = sha256(src + flags)
    if key in cache: return cache[key]

    if not compile(src):            return 0.0, "COMPILE_FAIL"
    if not run_ok():                return 0.0, "RUNTIME_FAIL"
    if not custom_kernel_ran():     return 0.0, "FALLBACK_DETECTED"   # 反作弊自查
    if not precision_ok():          return 0.0, "PRECISION_FAIL"

    t   = median(auto_bench 口径计时 × 3 个独立进程)
    iqr = IQR(那 3 次)
    return BASELINE_MS_FIXED / (t + 0.5 * iqr)
```

要点：
- **计时必须完全复刻 `auto_bench.py`**：`perf_counter` + 每次 `sync`，warmup 200 / repeat 500 / median。直接把 `auto_bench.py` 拉进 harness 当库用，不要自己重写。
- `BASELINE_MS_FIXED` 测一次固定成常量，否则 baseline 自身抖动会污染排序。
- **精度门禁比官方更严**：用 20 个随机 seed + 多 shape，且要求裕度占用 < 0.5（留一半安全边际），防止过拟合到 seed 42。
- 门禁里必须有 `custom_kernel_ran()`，对应规则 5.1 的反作弊条款。

### 6.4 多保真评估

| 级别 | 手段 | 耗时 | 用途 |
|---|---|---|---|
| F0 | 只编译 | ~20s | 淘汰非法个体 |
| F1 | warmup 20 / repeat 50 + 1 seed 精度 | ~10s | 粗排，留 top 30% |
| F2 | 完整 auto_bench 口径 × 3 进程 + 20 seed 精度 | ~40s | 精测入库 |

### 6.5 进化算子

- **选择**：锦标赛 `k=3` + 每岛精英保留 2
- **交叉**：均匀交叉 `p_cx=0.7`；跨骨架个体只交换共有基因
- **变异**：`p_mut=0.15/基因`；有序基因 80% 走 ±1 邻域、20% 均匀重采样；类别基因均匀重采样。**自适应**：连续 5 代无提升 → `p_mut` 翻倍至 0.4；再 5 代无提升 → 重启最差 50%
- **岛屿模型**：按 `skeleton` 分 4 岛（每岛 10 个体，总种群 40），每 5 代迁移 2 个精英
- **播种**：1 个 = 当前 KernelSwift 版本；4 个 = P0/P1 的人工最优；其余用拉丁超立方采样

### 6.6 预算

| 项 | 值 |
|---|---|
| 种群 / 代数 | 40 / 40 → 名义 1600 次评估 |
| 去重命中 ~30% → 实际 | ~1100 次 |
| 多保真后均耗时 | ~25s/个体 |
| 单卡串行 / 4 卡并行 | ~7.6h / **~2h** |
| 3 seed × 4 算法的完整对照 | **~24h 机时** |

---

## 7. Track B：代码级 GA（变异算子目录）

在 Track A 最优个体的源码上跑。**基因型是「已应用变异算子的集合 + 顺序」，表型才是源码** —— 这样可重放、可归因，交叉退化成集合交叉，比直接换代码块安全得多。

| M# | 变异算子 | 对应 |
|---|---|---|
| M1 | 删除冗余 `PipeBarrier`（读写集无冲突时） | W3 |
| M2 | `PipeBarrier` → `SetFlag/WaitFlag` 细粒度同步 | W3 |
| M3 | `Div(a,b,c)` → `Reciprocal(t,c); Mul(a,b,t)` | W5 |
| M4 | 相邻同类小调用合并（4 次 `count=8` → 1 次 `count=32`） | W2 |
| M5 | 多次 `DataCopyPad` → 单次 `DataCopy` + 块内重排 | W4 |
| M6 | 循环不变量外提 | — |
| M7 | 循环展开 / 循环交换 | — |
| M8 | buffer 原地化与合并 | W2 |
| M9 | 队列深度 1→2 | W3 |
| M10 | 独立向量指令重排（软件流水） | W3 |
| M11 | 标量 `GetValue/SetValue` → `Duplicate`/`Select`/mask | W1 |
| M12 | host 侧：同步 memcpy → static 缓存 / 标量入参 | H4 |

实现：tree-sitter 或 libclang 解析（`.asc` 本质是 C++），在语句列表上模式匹配 + 重写。编译错误信息回灌给 Track C。

---

## 8. Track C：LLM 引导进化（KernelSwift 闭环）

把 KernelSwift/Claude 当**变异与交叉算子**而非一次性生成器（FunSearch / AlphaEvolve 范式）：

- **变异 prompt** = 当前源码 + msprof 摘要 + 历史有效变异清单 + 「只改一处并说明理由」
- **交叉 prompt** = 两个高分个体 + 各自优势归因 + 「合并两者优点」
- **程序数据库**：MAP-Elites 风格，行为描述符 `(向量指令数, 标量同步数, UB 占用, host 调用数)`，按格保留精英以维持多样性
- **反馈闭环**：编译错误 / 精度失败 / profiling 热点直接回灌下一轮 prompt

单次 LLM 调用 30~60s，只在 Track A 收敛后用于跨结构跳跃，预算 200~400 次。

---

## 9. 对照实验与消融（报告核心）

同预算、同 3 个随机种子，报 mean±std。

| ID | 实验 | 目的 |
|---|---|---|
| **E-A** | Random / Local Search / **TPE(Optuna)** / GA / GA+LLM | 证明搜索算法本身有价值。**TPE 是强对手，20 维离散空间下常与 GA 相当，必须做** |
| **E-B** | 单一大种群 vs 岛屿模型 | 结构基因的处理方式 |
| **E-C** | 有/无 seeding | 人工先验的价值 |
| **E-D** | 有/无 多保真 | 同 wall-clock 下的解质量 |
| **E-E** | 有/无 源码去重缓存 | 工程手段收益 |
| **E-F** | **逐基因消融**：固定最优个体，逐个基因回退测跌幅 | **核心归因图**。预期前三：`tiling_mode` / `skeleton` / `scalar_free` |
| **E-G** | host 层基因 vs kernel 层基因的收益拆分 | 回答"钱花在哪" |

**指标**：最优 speedup（mean±std）｜best-so-far 曲线与 AUC｜达到 10x/20x/30x 所需评估次数｜编译/运行/精度通过率｜缓存命中率｜Mann-Whitney U 检验（n=3，同时报效应量）

---

## 10. 基础设施

```
ga/
├── config.yaml            基因空间、GA 超参、预算、设备列表
├── genome.py              基因定义、编解码、repair()、邻域
├── render.py              Jinja2 -> kernel.asc + host.cpp + model_new.py + CMake flags
├── templates/             S0_current / S1_batch_aos / S2_soa_plane / S3_soa_fused (.j2)
├── evaluate.py            F0/F1/F2 三级评估；子进程 + 超时 + device 清理
├── bench_adapter.py       直接复用官方 auto_bench.py 的计时与比对
├── cache.py               sha256(源码+flags) -> 结果，SQLite 持久化
├── ga.py                  岛屿模型主循环
├── baselines.py           random / local / TPE(optuna) 对照
├── llm_ops.py             Track C 变异/交叉算子
├── analyze.py             收敛曲线、消融柱状图、统计检验
└── mock_backend.py        Mac 上可跑的假评估器（解析代价模型打分）

experiments/
└── numeric_feasibility.py  ✅ 已完成，见 §3
```

**必须处理的坑**：
1. 每个 worker 独立 `build_<hash>/` 目录，避免 CMake 并发写冲突
2. 编译 180s / 运行 60s 超时即 kill；失败后 device reset，防连锁失败
3. **编译可并行，计时必须每卡串行独占**，否则数据全废
4. 每轮开跑前先测一次「金标个体」做漂移校准，漂移 >5% 该轮作废重跑
5. 1600 次编译产物及时清理，只留 top-50 的完整 build
6. 每代 checkpoint 种群 + cache，支持断点续跑

---

## 11. 里程碑

| 阶段 | 内容 | 产出 | 预计 |
|---|---|---|---|
| **P0** ✅ | ① host 打点拆出三段时间 ② 验证 AST 过滤下 `.so` 能加载 ③ 只改 host 侧 | **已完成**：3.834x → 4.629x，地板 55.3us，上限 22.7x（§2.3） | 已完成 |
| **P1** | **先只写 S1（AoS 批量化 + 消除标量同步）** —— 按 §2.3 的推演，S1 若能把 kernel 压到 30~50us 就已拿到 12~15x；S2（SoA 转置）留到 S1 落地后再按边际收益决定是否值得 | S1 骨架 + 实测 | 1.5 天 |
| **P1b** | 仅当 S1 结果显示还有明显空间时才做：S2 SoA 平面 + `GatherMask` 吞吐实测 | S2 骨架 | 1.5 天 |
| **P2** | 搭 harness（render/evaluate/cache/mock），复用官方 auto_bench | Mac mock 跑通 + 真机跑通 10 个体 | 1.5 天 |
| **P3** | Track A 全量搜索 + E-A/B/C/D 对照 | 收敛曲线、最优个体 | 2 天（含 24h 机时） |
| **P4** | Track B 精修 + E-F/E-G 消融 | 归因图 | 1.5 天 |
| **P5** | Track C LLM 闭环 | 对照结果 | 1.5 天 |
| **P6** | hold-out 复验（20 未见 seed + 多 shape）+ 按 §4.2 打包提交 | 提交件 | 0.5 天 |

---

## 12. 风险

| 风险 | 影响 | 对策 |
|---|---|---|
| **AST 过滤导致 `.so` 加载失败** | 直接 0 分 | **P0 第一件事就验证**；加载逻辑放 `__init__` 或独立模块 |
| 被判定 fallback 到内置算子 | 不计成绩 | 门禁加 `custom_kernel_ran()` 自查 |
| 官方收紧容差或换 seed | 数值近似翻车 | 自测门禁要求裕度 <0.5；已剔除截断迭代基因 |
| `GatherMask`/`Gather` 实际吞吐差 | S2/S3 打折 | 保留 S1（AoS 批量化）兜底，P1 就测出来 |
| 计时噪声淹没微小改进 | 搜索走偏 | median of 3 进程 + IQR 惩罚 + 金标漂移校准 |
| GA 早熟 | 停在局部最优 | 岛屿模型 + 自适应变异 + 50% 重启 |
| 机时不够 | 对照做不完 | 优先级 E-A > E-F > E-G > E-C > E-B > E-D > E-E |

---

## 13. 一句话总结

**host 侧的同步 memcpy 已经干掉，白拿 +21%（3.83→4.63x）。地板实测 55.3us，硬上限 22.7x。剩下 182us 全在 kernel 里：先做 S1（AoS 批量化 + 消除 82 对/矩阵的标量同步 + fp16），压到 30us 就有 ~15x；S2 完整 SoA 转置只多换 40%，按边际收益再定。最后用岛屿模型 GA 搜细节、用逐基因消融归因。截断迭代不要碰。**
