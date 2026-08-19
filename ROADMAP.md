# 后续工作路线图

> GA 设计详解见 [GA_DESIGN.md](GA_DESIGN.md)
>
> 记录时点：提交件 v1 已可打包（v1=158.55us，官方口径 **9.86x**，精度裕度占用 0.0000）
> 完整实验设计见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)，本文只列**尚未完成**的部分。

---

## 0. 当前位置

| 阶段 | 状态 | 产出 |
|---|---|---|
| P0 host 层 + 提交形态验证 | ✅ | 同步 memcpy 与设备查询去除，+17% |
| P1 kernel 重写（S1 系列） | ✅ | 5.79x → 9.86x，时间构成已完全拆清 |
| **提交件 v1** | ✅ | `submission/`，含 33 项合规自检 + 官方 auto_bench 复核脚本 |
| P2' GA 框架 | ✅ | `ga/`，mock 上端到端跑通 |
| **P3' GA 搜索** | ⏳ **下一步** | `bash ga/run_ga.sh 60`，3 个 seed，约 2.5 小时机时 |
| P4' 紧凑布局骨架 | ❌ | 唯一剩余的大杠杆（+19%） |
| P5' 代码级 GA / LLM 闭环 | ❌ | 方法学亮点 |
| P6' 最终复验与提交 | ❌ | |

### 剩余性能空间（实测拆解）

| 成分 | 耗时 | 能否优化 |
|---|---|---|
| launch + dispatch + sync 地板 | 62.0 us (63%) | ❌ 不可控，`M5` 已是任何算子的下界 |
| CopyIn + CopyOut | 27.0 us (27%) | ⭕ **唯一杠杆**，需改布局 |
| 全部计算 | 10.0 us (10%) | ❌ 已压到边际收益极低 |

| 场景 | 预期 v1 | 预期加速比 |
|---|---|---|
| 现在 | 158.55 us | 9.86x |
| 紧凑布局落地 | ~133 us | ~11.7x |
| 绝对上限（计算与搬运归零） | ~105 us | ~14.9x |

---

## P3'：GA 搜索（下一步）

```bash
bash ga/run_ga.sh 60      # 预算 60 × 3 seed，约 2.5 小时
```

只跑 GA，多个随机种子独立运行。单次评估 ~51s；已完成的 seed 自动跳过，
中断可续跑。跑完自动调用 `ga/analyze.py` 出汇总。

### 产出

- **收敛曲线**：best-so-far vs 真实评估次数，各 seed 一行 + 平均
- **各 seed 稳定性**：最优 fitness 的 mean/std/极差
- **逐基因消融 + 边际最优**：报告的核心图表
- **健康度**：成功率与各结局分布

### 若报告需要，补一轮随机搜索对照

只有 GA 的话，「GA 找到了 X」无法区分于「跑了 N 次评估碰到了 X」。补一轮同预算的
随机搜索即可（约 50 分钟）：

```bash
python ga/search.py --algo random --backend real --budget 60 --seed 0 \
       --db ga/runs/random_s0.sqlite
```

### 预期管理

基因空间去重后只有 **4480 个有效点**，且种子 `s1d_arith` 已接近最优。
**GA 大概率只能再挤出百分之几**。它的价值在方法学与归因，不在绝对性能。
真正能扩大空间的是 P4'（结构基因）与 P5'（代码级变异）。

### 待办

- [ ] 跑完 4×3 矩阵
- [ ] `python ga/apply_best.py --apply` —— 把最优基因写进 `CMakeLists.txt` 默认值
      （内含全档复验与 1.5% 噪声阈值，不达标不会替换）
- [ ] `analyze.py` 增加 matplotlib 出图（现在只有文本表）

---

## P4'：紧凑布局骨架（性能上的唯一大杠杆）

### 动机

当前 padded 布局为了把每矩阵 16 个 float 展开成 32 个，CopyIn 必须**逐行**下发
`DataCopyPad` 描述符：`n*4` 个 16B 块进 + `n*4` 个出，实测 27 us。

若 UB 直接采用 GM 的紧凑布局（16 float/矩阵），CopyIn/CopyOut 各退化成
**一次连续 `DataCopy`**（32B 对齐，1 个描述符），这 27 us 可降到 ~2 us。

### 技术方案

紧凑布局下矩阵 m 占 2 个 block（16 float），行 0/1 在 block 0，行 2/3 在 block 1。

- **行和**：两次 `PairReduceSum`。探针已确认其布局为「每块产出 4 个结果、连续排列」，
  两次之后正好得到每行的和（每矩阵 4 个，连续）。
- **列和**：`Add` 带 blkStride，两个 block 相加得到 (行0+行2, 行1+行3) 的 8 个值，
  再用 `PairReduceSum`… **需要探针确认**能否凑出正确的列和。
- **广播回去**：这是最难的一步。行和需要「每个值重复 4 次」，而 `Brcb` 是重复 8 次。
  候选：`Brcb` 到临时区后用带 stride 的 `Div`（src1BlkStride 取半）／
  `Copy` 带 stride ／ `GatherMask`。**必须先写探针实测**。

### 待办

- [ ] 写探针验证「每值重复 4 次」的广播手段（延续 `probe/` 的做法，用 `PipeBarrier<PIPE_ALL>`）
- [ ] 实现 `sinkhorn_normalize_kernel_s2.asc`
- [ ] 用 `scripts/s1_simulate.py` 的方式先在 CPU 上验证索引运算
- [ ] 作为 `skeleton` 结构基因加入 `ga/genome.py`，让搜索决定它值不值
- [ ] 若落地，预期 v1 ~133 us / ~11.7x

> **教训**：不要在没有实测的情况下动结构。S1b（宽 ColNormalize）把 repeat 数降了
> 7 倍却毫无收益，因为它优化的是只占 3% 的迭代体。动手前先跑
> `scripts/s1_scaling.py` 确认收益在哪。

---

## P5'：代码级 GA 与 LLM 闭环（方法学亮点）

### Track B：代码级变异

基因型是「已应用的变异算子集合 + 顺序」，表型才是源码 —— 这样可重放、可归因，
交叉退化成集合交叉，比直接换代码块安全得多。

变异算子目录（前 6 个语义保持）：

| # | 算子 |
|---|---|
| M1 | 删除冗余 `PipeBarrier`（读写集无冲突时） |
| M2 | `PipeBarrier` → `SetFlag/WaitFlag` 细粒度同步 |
| M3 | 相邻同类小调用合并（改 mask/repeat/stride） |
| M4 | 循环不变量外提 |
| M5 | 循环展开 / 循环交换 |
| M6 | buffer 原地化与合并 |
| M7 | 独立向量指令重排（软件流水） |
| M8 | double buffering（多 tile 时 MTE 与 V 重叠） |

实现：tree-sitter 或 libclang 解析 `.asc`（本质是 C++），在语句列表上模式匹配重写。

### Track C：LLM 作为变异/交叉算子

FunSearch / AlphaEvolve 范式。变异 prompt = 当前源码 + profiling 摘要 +
历史有效变异清单 + 「只改一处并说明理由」。程序数据库用 MAP-Elites，
行为描述符取 `(向量指令数, MTE 描述符数, UB 占用, 每 tile 固定开销)`。

### 待办

- [ ] Track B 的 AST 重写框架
- [ ] 变异算子实现与单测（每个算子必须保证语义等价，用 `s1_simulate.py` 验证）
- [ ] Track C 的 prompt 模板与程序数据库
- [ ] 对照：GA / GA+TrackB / GA+LLM

---

## P6'：最终复验与提交

### 从 GA 结果到提交件的完整链路

```
ga/runs/*.sqlite          GA 搜索结果（快档保真度，排名可信、数值不可直接采信）
        │
        ├─ python ga/apply_best.py            只看：列出候选、全档复验、给结论
        ├─ python ga/apply_best.py --apply    确认后写入 CMakeLists.txt 默认值
        │
CMakeLists.txt 默认值      <- build.sh 不带任何 -D，用的就是这里
        │
        └─ bash submission/package.sh "队伍名" "UID"
                打包前会打印生效配置并存进 results/build_config.txt
```

**`apply_best.py` 的两道保险**：候选必须用**全档**（精度 5 seed + warmup200/repeat500，
3 个独立进程取中位数）重新评测；且必须比当前默认值领先 **>1.5%** 才替换 ——
v1 的测量 CV 约 2%，采纳噪声赢家会让提交件比已验证的配置更差。

- [ ] 用最优配置重新 `bash submission/package.sh "队伍名" "UID"`
- [ ] hold-out 复验：20 个未在调优中出现过的 seed + 全部形状
- [ ] `bash submission/package.sh "队伍名" "UID"` —— 内置 33 项合规自检，不通过会中止打包
- [ ] `bash run_auto_bench.sh` —— 用**官方原版** auto_bench.py 复核（不是我们复刻的版本）
- [ ] 复核提交规范
  - [ ] 邮件主题 `【2026KernelSwift算子创新大赛】-<队伍名>-赛道二-<UID>`
  - [ ] 附件名 `【2026KernelSwift算子创新大赛】-<队伍名>-Task03_sinkhorn-<UID>.tar.gz`
  - [ ] 发送至 `deeplink@pjlab.org.cn`
  - [ ] PR 标题 `[KernelSwift算子优化]<队伍名>-Task03_sinkhorn-<UID>`
  - [ ] PR 内容含作品说明 / 优化方案 / 性能测试结果 / 原创声明（`submission/README.md` 已覆盖）
- [ ] 在**干净的容器**里从 tar.gz 解压后完整跑一遍，确认自包含

---

## 需要长期注意的几条

1. **判据用 `v1` 自身耗时或 `M1`，不要用 speedup。** baseline 跨进程漂移 ±11%，
   会把 2% 的真实差异淹没。固定基准 1563 us 已写进 `ga/evaluate.py`。
2. **精度门禁是唯一的鲁棒性保障。** fitness 只优化速度 —— mock 上随机搜索已经挑出
   `use_max=0`（省 2us 但丢失大幅值鲁棒性）。门禁里没有的要求等于不存在，
   所以 `check_shapes.py` 覆盖了 12 个形状含 `randn×8` / `randn×32`。
3. **动结构前先跑 `s1_scaling.py`。** 它用 `t(repeat)=a+b·repeat` 把每 tile 固定开销
   和每次迭代开销分开，零代码成本（`repeat` 是算子入参）。
4. **不要信文档，写探针。** 本仓库 README 对 `BlockReduceSum` 的描述就和实际代码对不上；
   `Reciprocal` 实测只有 9 位精度；`DataCopyPad` 会把每块补齐到 32B。
   探针 kernel 必须用 `PipeBarrier<PIPE_ALL>`，只用 `PIPE_V` 会导致 V 与 MTE2 乱序。
