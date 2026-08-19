# P0 执行说明

P0 的目标是**在不碰 kernel 的前提下**回答三个问题，并顺手把能白拿的加速比拿掉：

1. 提交文件能不能扛住官方的 AST 过滤？（不能的话提交上去直接跑不起来）
2. host 绑定层每次调用的固定开销有多大？去掉之后能提多少？
3. 任何自定义算子的**时间地板**是多少？—— 决定了 kernel 要优化到什么程度才有意义

## 一键跑（服务器）

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
bash scripts/run_p0.sh
```

脚本会：编译 `orig` / `opt` 两个变体 → 跑 AST 检查 → 对每个变体做时间拆解 + 官方口径评测。

## 改了什么

### `op_extension/sinkhorn_normalize_torch.cpp`

原实现每次 `forward()` 都做：查设备核数、重算 tiling、`at::empty` 分配 tiling 张量、
**`aclrtMemcpy` 同步阻塞 H2D**。官方计时口径把这些全额计入。

新增两个编译期开关（`CMakeLists.txt` 里可 `-D` 覆盖）：

| 开关 | 0（对照） | 1（默认） |
|---|---|---|
| `SN_TILING_MODE` | 每次分配 + 同步 memcpy | 静态缓存，仅在 `(totalMats, repeat, eps)` 变化时拷贝一次 |
| `SN_CORE_QUERY` | 每次 `aclrtGetDeviceInfo` | 首次查询后缓存 |

稳态下 `opt` 变体**零 H2D 拷贝、零额外分配、零设备查询**。

另外注册了 `torch.ops.npu.sinkhorn_normalize_call_count()`，用于自查自定义 kernel
确实被执行（对应规则 5.1 的反 fallback 条款）。**不在提交热路径上。**

### `submission/model_new.py`

官方 `auto_bench.py` 会对提交文件做 AST 过滤后再 exec，**只保留**
`Import / ImportFrom / ClassDef / FunctionDef / AsyncFunctionDef / 字面量赋值`。
顶层的 `torch.ops.load_library(...)` 会被静默丢弃。

所以本文件顶层不含任何可执行语句，`.so` 加载全在 `ModelNew.__init__` 里完成。
加载失败会抛 `RuntimeError`，**不会静默 fallback 到 PyTorch 内置算子**。

`.so` 固定从 `model_new.py` 所在目录加载，无环境变量、无多路径回退。
调用方（`run_p0.sh` / `run_s1.sh` / `ga/evaluate.py`）会把 `model_new.py`
复制到各自的构建目录，使 `.so` 始终在它旁边。

## 脚本清单

| 脚本 | 需要 NPU | 作用 |
|---|---|---|
| `scripts/p0_ast_check.py` | 否 | 复刻 AST 过滤，验证提交文件存活 + 反 fallback 检查 |
| `scripts/bench_official.py` | 是（`--device cpu` 可冒烟） | 复刻官方口径：warmup200/repeat500/median/perf_counter + allclose(1e-2) |
| `scripts/p0_host_breakdown.py` | 是（`--device cpu` 可冒烟） | 把 forward 拆成 host / device / sync 三段，给出理论上限 |
| `scripts/p0_msprof_target.py` | 是 | msprof 采集目标，拿纯 kernel 时间 |
| `scripts/run_p0.sh` | 是 | 一键跑完上面全部 |

## 怎么读结果

`p0_host_breakdown.py` 的输出里：

- **M5（单个最简 NPU 算子 + sync）是地板**。`speedup 理论上限 ≈ M0 / M5`。
- **M2（不 sync 的 host 端耗时）** 如果明显大于 M5，说明绑定层有阻塞调用——
  这正是 `SN_TILING_MODE=0` 时的同步 memcpy。对比 `orig` 和 `opt` 两个变体的 M2 即可量化。
- **M1 − M2** 是 device + sync 的残差，这部分才需要靠 kernel 优化来压。

拿到 msprof 的纯 kernel 时间后，完整拆解就齐了：
`M1 ≈ host(M2) + kernel(msprof) + launch/sync 残差`。

## P0 实测结果（已完成）

两次独立运行的汇总：

| 指标 | `orig`（每次同步 memcpy + 查核数） | `opt`（静态缓存） | 差值 |
|---|---|---|---|
| M2 host 端耗时（不 sync） | 114.28 / 119.34us | **39.78 / 40.02us** | **−76.9us（−66%）** |
| M1 裸算子 + sync | 301.41 / 309.74us | **237.19 / 240.31us** | **−66.8us（−21.9%）** |
| 官方口径 speedup | 3.834x / 4.190x | 4.629x / 4.741x | **+13~21%** |

那个每次调用的同步 `aclrtMemcpy` 一项就值约 **17%**（两次均值），零精度风险。

**时间地板**（决定一切目标的上限）：

| 测量项 | 耗时 |
|---|---|
| M3 `torch.npu.synchronize()` 空转 | 16.5~17.5us |
| M4 `torch.empty_like` + sync | 30.3~31.6us |
| **M5 单个最简 NPU 算子往返 + sync** | **~56us ← 硬地板** |

三条关键工程结论：

1. **`M1` 是最稳的判据**（两次运行差 0.7%），而 speedup 因 baseline 跨进程漂移 ±11% 反而不稳
2. **Python wrapper 开销是噪声**：M6 − M1 在 `orig` 为 −6.48us、`opt` 为 +8.19us，一正一负
3. **`.so` 加载**：官方 `load_ks_module()` 确实设置 `module.__file__`，
   所以 `.so` 与 `model_new.py` 同目录即可被找到，无需环境变量

> P0 之后的进展（S1 kernel 重写等）见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) §2.3 与 §10。
> 当前成绩：v1 = 158.55us，官方口径 9.86x。
