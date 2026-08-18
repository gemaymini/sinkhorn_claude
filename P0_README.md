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

`.so` 路径查找：`SINKHORN_OPS_SO` → `SINKHORN_OPS_DIR` → cwd 及其 `build/` →
本文件所在目录 → `sys.path`。**最稳的做法是显式设 `SINKHORN_OPS_SO`。**

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

## 已在 Mac 上验证

- `p0_ast_check.py`：提交文件 13 个顶层节点全部保留，仅 docstring 被丢弃（无害）；
  过滤后 exec 成功、`Model/ModelNew/get_inputs/get_init_inputs` 齐全；
  缺 `.so` 时正确抛 `RuntimeError`。
- `bench_official.py --device cpu --self-test`：harness 自洽（speedup ≈ 1.0x）。
- `p0_host_breakdown.py --device cpu`：各项测量与推导逻辑正常。

C++ 侧只做了静态审查，**未编译验证**——服务器上第一次 `run_p0.sh` 会暴露编译问题。
