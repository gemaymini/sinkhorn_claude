# SinkhornNormalize 昇腾算子优化

2026 KernelSwift 算子创新大赛 · 赛道二（华为昇腾）· Task03 `sinkhorn`

在 KernelSwift 平台生成的 AscendC 算子基础上做实测驱动的优化，并用遗传算法搜索编译期配置。

| 阶段 | v1 耗时 | 加速比 |
|---|---|---|
| KernelSwift 生成版 | 269.97 us | 5.79x |
| + host 绑定层优化（P0） | — | — |
| + S1 kernel 批量化（P1） | 172.69 us | 9.05x |
| + 精细同步 | 164.14 us | 9.52x |
| **当前（eps 算术构造）** | **158.55 us** | **9.86x** |

> 加速比基准 = 1563 us（20 次实测中位数）。该基准跨进程波动 ±11%，
> 所有内部对比一律以 **v1 自身耗时**或 `M1`（裸算子 + sync）为判据。
> 精度：12 形状 × 5 seed，官方容差下裕度占用 **0.0000**（最大绝对误差 1.8e-07）。

---

## 目录

```
.
├── op_kernel/                       NPU 计算层
│   ├── sinkhorn_normalize_tiling.h      tiling 结构 + host/kernel 共享常量
│   ├── sinkhorn_normalize_kernel_s1.asc S1 kernel（当前使用）
│   └── sinkhorn_normalize_kernel.asc    原 per-matrix kernel（对照用）
├── op_extension/                    PyTorch 绑定层
├── op_host/                         直调入口
├── submission/                      提交件（含合规自检与打包脚本）
├── ga/                              遗传算法搜索框架
├── probe/                           AscendC 原语探针（开发诊断）
├── scripts/                         验证、评测与诊断脚本
├── experiments/                     数值可行性实验（纯 CPU，Mac 可跑）
└── *.md                             文档，见下
```

---

## 快速开始

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

**打包提交件**（内置编译 → 精度门禁 → 官方口径评测 → 34 项合规自检）

```bash
bash submission/package.sh "队伍名" "UID"
```

**跑 GA 搜索**（约 2.5 小时，另开终端 `bash ga/monitor.sh` 看进度）

```bash
bash ga/run_ga.sh 60
python3 ga/apply_best.py            # 全档复验候选
python3 ga/apply_best.py --apply    # 确认后写入 CMakeLists.txt 默认值
```

**变体 A/B 与诊断**

```bash
bash scripts/run_s1.sh              # kernel 变体对照
bash scripts/run_probe.sh           # AscendC 原语行为探针
python3 scripts/s1_scaling.py --so <path>   # 拆分每 tile 固定开销 vs 每次迭代开销
```

---

## 文档

| 文档 | 内容 |
|---|---|
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | 实验总纲与全部实测记录（评测口径、时间构成、数值近似边界） |
| [GA_DESIGN.md](GA_DESIGN.md) | 遗传算法设计说明（基因空间、适应度、岛屿模型、局限） |
| [ROADMAP.md](ROADMAP.md) | 后续工作路线图 |
| [P0_README.md](P0_README.md) | P0 阶段（host 层 + 提交形态）执行说明 |
| [submission/README.md](submission/README.md) | 提交件作品说明（优化方案、性能结果、原创声明） |

---

## 几条值得记住的实测结论

**时间构成**（`scripts/s1_scaling.py` 用 `t(repeat)=a+b·repeat` 拟合）

| 成分 | 耗时 | 占比 |
|---|---|---|
| launch + dispatch + sync 地板 | 62.0 us | 63% |
| CopyIn + CopyOut | 27.0 us | 27% |
| 全部计算（含 10 次 Sinkhorn 迭代） | 10.0 us | 10% |

**10 次迭代总共不到 5us。** 加速比的绝对上限约 14.9x。

**平台行为**（`probe/` 实测，不要信文档）

- `Reciprocal` 只有约 **9 位**有效精度（`1/1` 得 0.998），不是 fp32
- `DataCopyPad` GM→UB 时每块补齐到 32B，**SoA 平面布局在 910B 上不可行**
- 探针 kernel 必须用 `PipeBarrier<PIPE_ALL>`，只用 `PIPE_V` 会让 V 与 MTE2 乱序

**数值边界**

- `softmax(x) + eps` **不能省** —— 只在官方 `randn` 分布下看起来安全，`randn×4` 就崩
- **不能截断 Sinkhorn 迭代** —— 第 10 次本身都尚未收敛
- fp16 计算裕度占用仅 0.061（16 倍余量），但当前计算只占 10%，收益有限
