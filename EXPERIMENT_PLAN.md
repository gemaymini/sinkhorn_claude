# SinkhornNormalize @ 昇腾 910B A2 —— 进化搜索调优实验方案

> 赛题：2026 KernelSwift 算子创新大赛 赛道二 Task03 `sinkhorn`
> 起点：`result.json` 记录 `speedup=4.622`（237.48us，自测 event 口径 —— 非官方口径）
> 当前：v1 = **158.55us**，官方口径 **9.86x**（固定 baseline 1563us）
> **本文档已按官方评测脚本 `DLBlas/benchmarks/ks/auto_bench.py` 与本地数值实测结果校准**

---

## 0. 结论先行

> 本文档是**实验总纲与实测记录**。§0~§4 是已完成的实测结论（数据可直接引用），
> §5 起是路线演变、方法学取舍与进度。GA 的实现细节见 [GA_DESIGN.md](GA_DESIGN.md)，
> 未来工作见 [ROADMAP.md](ROADMAP.md)。

1. **官方容差是 `atol=1e-2, rtol=1e-2`**，比仓库自测用的 `1e-4/1e-5` 宽两三个数量级。
   但**宽容差不等于可以随便近似** —— 见第 6 条。
2. **官方计时是 `perf_counter` 包住整个 `model.forward()` 再 `sync`，warmup 200 / repeat 500 / 取 median**。
   Python 层、torch dispatch、host 端 ACL 调用、kernel launch、stream sync **全部计入**。
3. **第一块肉在 host 绑定层**：原实现每次调用都查一次设备核数、分配两次张量、
   并做**一次同步阻塞的 `aclrtMemcpy` 传 tiling**。去除后 **+17%**，零精度风险。
4. **kernel 的瓶颈不是算力，是小粒度指令 + 标量↔向量同步**：原实现每个 4×4 矩阵
   ~228 条向量指令、~198 个 `PipeBarrier`、**~82 对标量-向量同步**，每条向量指令
   只用到 8~32 个 lane（fp32 满载 64）。重写为 padded-AoS 批量化后 5.79x → 9.86x。
5. **当前进入 launch 开销主导区间**：`t(repeat)=a+b·repeat` 拟合显示
   **launch 地板 62us（63%）+ 搬运 27us（27%）+ 全部计算 10us（10%）**，
   10 次 Sinkhorn 迭代总共不到 5us。**绝对上限约 14.9x**（当前 9.86x）。
6. **两条数值红线**：`softmax(x) + eps` 不能省（只在官方 `randn` 分布下看着安全，
   `randn×4` 就崩）；Sinkhorn 迭代不能截断（第 10 次本身都尚未收敛）。
   **数值近似必须跨输入尺度验证。**
7. **平台原语要写探针实测**：`Reciprocal` 只有 9 位精度；`DataCopyPad` 每块补齐 32B
   导致 SoA 平面布局不可行。仓库原 README 对 `BlockReduceSum` 的描述就和实际代码对不上。

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

### 2.3 P0 真机实测结果（910B2，CANN 9.0.0，两次独立运行，已完成）

`bash scripts/run_p0.sh` 的产出。**所有后续目标都以这组数为准。**

### host 侧优化的实际收益

| 指标 | `orig`（每次同步 memcpy + 查核数） | `opt`（静态缓存） | 差值 |
|---|---|---|---|
| M2 host 端耗时（不 sync） | 114.28 / 119.34us | **39.78 / 40.02us** | **−76.9us（−66%）** |
| M1 裸算子 + sync | 301.41 / 309.74us | **237.19 / 240.31us** | **−66.8us（−21.9%）** |
| 官方口径 speedup | 3.834x / 4.190x | 4.629x / 4.741x | **+13~21%** |

> 那个每次调用的同步 `aclrtMemcpy` 一项就值 ~17% 的加速比（两次均值），零精度风险。
> **M1 是最稳的判据**（两次运行差 1.3%），speedup 因 baseline 漂移反而不稳。

### 测量噪声（关键工程结论）

| 量 | 观测范围 | 波动 |
|---|---|---|
| M0 参考实现（baseline） | 1173.62 ~ 1314.06us | **±6%** |
| M5 地板 | 54.90 ~ 58.97us | ±4% |
| M1（`opt`） | 237.19 ~ 240.31us | ±0.7% |

**baseline 的 ±6% 漂移直接污染 speedup 排序**，而官方口径是"v0 全测完再测 v1"，慢漂移会整体偏向一侧。
对策已实现在 `scripts/bench_official.py`：
- `--mode interleaved`（默认）：把 repeat 切成 rounds 段轮流测 v0/v1，抵消慢漂移，调优内部对比用
- `--baseline-ms`：固定 baseline 常量、跳过 v0 测量，**GA 的 fitness 用这个，评估耗时直接减半**
- `--mode official`：严格顺序口径，对外报数用

### Python wrapper 开销 = 噪声，无需优化

M6（完整 `ModelNew.forward`）− M1（裸算子调用）：`orig` 为 **−6.48us**，`opt` 为 **+8.19us**。
一正一负 ⇒ 落在噪声内。**`ModelNew.forward` 里的 `x.device.type` 检查等是免费的，基因组里的
`python_wrapper` 一项可以直接删掉。**

### 时间地板

| 测量项 | 耗时 |
|---|---|
| M3 `torch.npu.synchronize()` 空转 | 16.5~17.5us |
| M4 `torch.empty_like` + sync | 30.3~31.6us（推出分配器本身 ≈ 14us） |
| **M5 单个最简 NPU 算子往返 + sync** | **~56us ← 硬地板** |

**`speedup` 理论上限 ≈ 1250 / 56 ≈ 22x。** 任何 kernel 优化都无法突破。
比方案初稿估的 30~40x 低，原因是这台机器 host-device 往返开销偏高（光 sync 空转就 17us）。

### 当前时间构成（`opt`）

```
M1 = 238.8us（两次均值）
  ├─ host CPU（dispatch + empty_like ~14us + launch）  ≈ 40us   ← 已低于地板，不再是瓶颈
  └─ kernel 净计算 + launch/sync 残差                   ≈ 182us  ← 唯一的大头（M1 − M5，两次都是 181.5us）
```

### kernel 优化目标推演（固定 baseline = 1250us）

| kernel 净时间 | 预期 M1 | 预期 speedup |
|---|---|---|
| 182us | 239us | 5.2x | 现状 |
| 60us | ~117us | ~10.7x |
| **30us** | **~87us** | **~14.4x** ← 性价比拐点 |
| 15us | ~72us | ~17.4x |
| 5us | ~62us | ~20.2x |
| 0（理论） | 56us | 22.3x |

**边际收益递减很陡：压到 30us 拿 ~14x（2.8 倍提升），再拼到 5us 只多拿 40%。**
据此把 P1 的路线从"直接上 S2 SoA 转置"改成"先做 S1"，见 §11。

---

## 3. 数值近似实测结论（已完成，`experiments/numeric_feasibility.py`）

20 个随机种子，shape `(1,1024,4,4)`，与 fp32 reference 比对，判据 `atol=1e-2, rtol=1e-2`。
「裕度占用」= `max|diff| / (atol + rtol·|ref|)`，**<1 通过，越小越安全**。

| 变体 | 通过 | 最大绝对误差 | 裕度占用 | 结论 |
|---|---|---|---|---|
| 省掉全部 eps | 20/20 | 1.18e-05 | 0.001 | ❌ **见下方更正** |
| softmax 免减 max | 20/20 | 1.79e-07 | **0.000** | ✅ 启用 |
| 上面两项叠加 | 20/20 | 1.18e-05 | **0.001** | ✅ 启用 |
| **fp16 全程计算** | 20/20 | 9.29e-04 | **0.061** | ✅ **启用（16x 裕度）** |
| bf16 全程计算 | 20/20 | 6.54e-03 | 0.429 | ⚠️ 可用但裕度仅 2.3x，作 fp16 的备选 |
| fp16 + 免 eps + 免 max | 20/20 | 9.32e-04 | **0.061** | ✅ **推荐组合** |
| 截断到 9 次迭代 | 20/20 | 3.94e-03 | 0.234 | ❌ 见下 |
| 截断到 8 次迭代 | 20/20 | 9.39e-03 | 0.558 | ❌ 见下 |
| 截断到 7 次迭代 | **19/20** | 1.71e-02 | 1.012 | ❌ 已翻车 |
| 截断到 6 次及以下 | ≤18/20 | ≥2.80e-02 | ≥1.657 | ❌ |

### ⚠ 更正：`softmax + eps` 不能省（P1 阶段发现）

上表的「省掉 eps 安全」结论**只在官方输入分布（`torch.randn`，std=1）下成立**。
把输入幅值放大后：

| 输入分布 | 无 eps | 仅分母 eps | 完整 eps |
|---|---|---|---|
| randn（官方） | 0.0009 | 0.0009 | **0.0000** |
| randn×2 | 0.037 | 0.037 | **0.0000** |
| randn×4 | **15.94 ❌** | 15.94 ❌ | **0.0000** |
| randn×8 | **83.5 ❌** | 83.8 ❌ | **0.0000** |

这不是舍入误差而是语义差异：softmax 输出小到 ~1e-11 时，加 `eps=1e-6` 会把它抬高
5 个数量级。而分母里的 eps 无关紧要（省不省数字完全一样）。

**结论**：S1 保留完整 eps，代价仅约 +7% 指令数，换来裕度 0.0000 和跨输入尺度的鲁棒性。
基因 `eps_mode` 的取值域从 `{EXACT, DROP}` 收窄为 `{EXACT}`。
**教训**：数值近似必须跨输入尺度验证，不能只测官方那一个分布。

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
| ~~**Kernel 结构**~~ | ~~SoA 平面布局~~ | ❌ **已被探针证伪**：`DataCopyPad` GM→UB 每块补齐到 32B，平面元素步长为 8，会有 8 倍 lane 浪费 |
| **Kernel 结构** | 批量化：单次处理 B 个矩阵而非 1 个 | 大 | 无 |
| **Kernel 指令** | `Div` → `Reciprocal` + `Mul`；消除冗余 `PipeBarrier`；double buffering；合并小 `DataCopyPad` | 中 | 无 |
| **数值** | fp16 计算 + 免 eps + 免 max | 中 | 低（已实测，裕度 16x） |
| **调度** | 用核数不必打满（小 shape 下满核未必最快）；tile 大小 | 中 | 无 |

### 为什么最终没走 SoA

探针实测（`probe/`，见 §2.4）显示 `DataCopyPad` 在 `blockLen=4B` 时**每块补齐到 32B**，
抽出来的"平面"元素步长为 8 而非紧凑排列，count-based 向量 API 无法直接使用，
只能退回 level-0 带 blkStride 的形式，每个 repeat 只用 8 个 lane —— 8 倍浪费。

最终采用的是 **padded-AoS 批量化**（S1），用的全是探针确认过的原语：
`DataCopyPad + rightPadding`、`BlockReduceSum`（结果连续）、`Brcb`、`Div`。
详见 `op_kernel/sinkhorn_normalize_kernel_s1.asc` 的文件头注释。

---

## 5. 路线演变：计划 vs 实际

最初规划了三条 Track（参数化模板 GA / 代码级 GA / LLM 引导进化）。实际执行时，
P0/P1 的实测结果大幅改变了优先级，这里如实记录：

| 原计划 | 实际 | 原因 |
|---|---|---|
| Track A：20 位基因的模板 GA | **收窄为 10 位基因、直接用编译期开关** | 5 条前提被实测证伪（见 §3 更正与 §2.3） |
| 先做 S2（SoA 平面） | **改做 S1（padded-AoS）** | 探针发现 `DataCopyPad` 补齐 32B，SoA 不可行 |
| 4 算法同预算对照 | **只跑 GA，多 seed** | 机时从 10h 降到 2.5h；对照可按需补一轮随机搜索 |
| Track B / Track C | **推迟** | 见 [ROADMAP.md](ROADMAP.md) |

**最大的一条教训**：如果当初照原基因空间直接跑 GA，24 小时机时会花在一个
「80% 的基因只影响 3% 运行时间」的空间里。P1 那几轮手工调优实质上是**搜索空间的设计阶段**。

---

## 6. 遗传算法

**完整设计见 [GA_DESIGN.md](GA_DESIGN.md)**（基因空间、适应度、岛屿模型、公平性设计、局限）。
这里只记与本文档其它部分的衔接：

- 基因空间 **10 位**，去冗余后 **4480 个有效点**，每一位的取值范围都来自 §2.3 / §3 的实测
- 适应度用**固定 baseline 1563us**（§1 口径），精度是**硬门禁**而非罚分
- 搜索用快档保真度，结果必须经 `ga/apply_best.py` 全档复验 + 1.5% 噪声阈值才会采纳
- 当前只跑 GA（`bash ga/run_ga.sh`），随机搜索等对照实现保留在 `ga/search.py` 里可按需启用

---

## 7. 未来工作

紧凑布局骨架（性能上唯一剩余的大杠杆，+19%）、代码级变异 GA、LLM 引导进化 ——
全部移到 **[ROADMAP.md](ROADMAP.md)**，含技术方案与待办清单。

---

## 8. 实验方法学的实际取舍

原方案设计了 E-A ~ E-G 七组对照。实际保留下来的和放弃的：

| 原对照 | 状态 | 说明 |
|---|---|---|
| E-A 算法对照（Random/Local/TPE/GA） | **放弃** | 机时考虑。若报告需要，补一轮同预算随机搜索约 50 分钟 |
| E-B 单种群 vs 岛屿模型 | 放弃 | 岛屿模型直接采用（依据：tile<64 有实测性能台阶） |
| E-C 有/无播种 | **保留能力** | `ga/search.py --no-seed` |
| E-D 有/无多保真 | 放弃 | 多保真直接采用 |
| E-E 有/无去重缓存 | 放弃 | 去重直接采用（实测压缩 38%） |
| **E-F 逐基因消融** | **保留，核心产出** | `ga/analyze.py` 输出，另有边际最优作兜底 |
| E-G host vs kernel 收益拆分 | **已完成** | 见 §2.3，host 层 +17%，kernel 层 5.79x→9.86x |

**实际执行中新增的方法学手段**（原方案没有，但被证明更重要）：

| 手段 | 脚本 | 解决的问题 |
|---|---|---|
| 迭代次数扫描 | `scripts/s1_scaling.py` | 用 `t(repeat)=a+b·repeat` 拆开每 tile 固定开销与每次迭代开销。**零代码成本，却是唯一能证明"迭代体只占 3%"的手段** |
| 原语行为探针 | `probe/` | 实测 API 的布局与精度，不信文档 |
| CPU 算法仿真 | `scripts/s1_simulate.py` | 逐条模拟 kernel 数据流，在本地排掉索引/尾块/填充位错误 |
| 数值可行性扫描 | `experiments/numeric_feasibility.py` | 跨输入尺度验证数值近似，抓出"省 eps 只对官方分布成立"这个陷阱 |
| 诊断变体 | `SN_S1_DIAG` | 故意算错、只用于计时，直接量出 MTE 搬运与 `BuildEpsVector` 的成本 |

---

## 9. 基础设施

```
scripts/
├── check_shapes.py         精度门禁（12 形状 × N seed，官方容差 + 裕度占用）
├── bench_official.py       官方口径评测（复刻 auto_bench，含 AST 过滤预演、交替测量、固定 baseline）
├── s1_scaling.py           固定开销 vs 每次迭代开销的线性拟合
├── s1_simulate.py          S1 算法的 CPU 逐条仿真
├── p0_host_breakdown.py    M1~M6 时间拆解
├── p0_ast_check.py         提交形态的 AST 过滤验证
├── probe_report.py         探针结果解读
├── _build_lib.sh           隔离编译、失败日志落盘、torch-python 定位
└── run_p0.sh / run_s1.sh / run_probe.sh

ga/
├── genome.py / cache.py / evaluate.py / search.py
├── apply_best.py           全档复验 + 噪声阈值 + 写回 CMakeLists
├── analyze.py              收敛曲线、逐基因消融、边际最优
└── run_ga.sh / monitor.sh

probe/          AscendC 原语探针（kernel + host）
experiments/    数值可行性实验（纯 CPU）
submission/     提交件 + 34 项合规自检 + 打包脚本
```

**必须处理的工程坑**（全部已实现）：

1. 每个个体独立 `build_ga_<hash>/`，避免 CMake 并发写冲突；评估完立即删除
2. 编译 300s / 运行 240s 超时即杀
3. **编译可并行，计时必须独占**
4. baseline 钉死成常量，否则 ±11% 漂移会淹没 2% 的真实差异
5. 结果库先写 `.partial`，跑完才改名 —— 中断不会被误判为已完成
6. 换容器后 `CMakeCache.txt` 失效：配置失败自动清目录重试
7. CMake 的 `find_package(Python3)` 可能挑到没装 torch 的解释器：显式传 `-DPython3_EXECUTABLE`

---

## 10. 里程碑（实际进度）

| 阶段 | 状态 | 产出 |
|---|---|---|
| P0 host 层 + 提交形态验证 | ✅ | 同步 memcpy 与设备查询去除，+17% |
| P1 kernel 重写（S1 系列） | ✅ | 5.79x → 9.86x，时间构成完全拆清 |
| 提交件 v1 | ✅ | `submission/`，34 项合规自检 + 官方 auto_bench 复核 |
| P2' GA 框架 | ✅ | `ga/`，mock 上端到端跑通 |
| P3' GA 搜索 | ⏳ 下一步 | `bash ga/run_ga.sh 60`，约 2.5 小时 |
| P4' 紧凑布局骨架 | ❌ | 唯一剩余大杠杆（+19%），见 ROADMAP |
| P5' 代码级 GA / LLM 闭环 | ❌ | 见 ROADMAP |

---

## 11. 风险与对策

| 风险 | 状态 | 对策 |
|---|---|---|
| AST 过滤导致 `.so` 加载失败 | ✅ 已解决 | 官方确实设置 `module.__file__`，`.so` 与 `model_new.py` 同目录即可；34 项合规自检覆盖 |
| 被判定 fallback 到内置算子 | ✅ 已解决 | 内置调用计数器 + 合规检查验证**每条 return 路径**都走自定义算子 |
| 官方收紧容差或换 seed | ✅ 已缓解 | 当前裕度占用 0.0000（余量 5 个数量级）；门禁要求 <0.5 |
| 计时噪声淹没真实差异 | ✅ 已缓解 | 固定 baseline + 交替测量 + IQR 惩罚；判据用 v1/M1 而非 speedup |
| GA 采纳噪声赢家 | ✅ 已缓解 | `apply_best.py` 全档复验 + 1.5% 阈值 |
| GA 为速度牺牲鲁棒性 | ✅ 已缓解 | 精度门禁含 `randn×32`，`use_max=0` 会因 `exp` 溢出被淘汰 |
| 报告缺少算法对照 | ⚠️ 未解决 | 需要时补一轮随机搜索（约 50 分钟） |

---

## 12. 一句话总结

**host 侧同步 memcpy 值 +17%；kernel 从 per-matrix 标量同步重写成 padded-AoS 批量化值 5.79x→9.86x；
剩下 63% 的时间是任何自定义算子都躲不掉的 launch 开销，绝对上限 14.9x。
所有数值近似必须跨输入尺度验证，所有平台原语行为必须写探针实测，
动结构前必须先用 `t(repeat)=a+b·repeat` 确认收益在哪。**
