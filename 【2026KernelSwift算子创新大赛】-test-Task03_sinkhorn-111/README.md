把逐个处理4x4矩阵重写成每个核批量向量化处理64个4x4矩阵。

# 基础算法流程

```python
x = x.softmax(-1) + eps
x = x / (x.sum(-2, keepdim=True) + eps)

for _ in range(repeat - 1):
    x = x / (x.sum(-1, keepdim=True) + eps)
    x = x / (x.sum(-2, keepdim=True) + eps)
```

定义行归一化和列归一化：

\[ R_\epsilon(A)_{ij} = \frac{A_{ij}}{\sum_k A_{ik}+\epsilon} \]

\[ C_\epsilon(A)_{ij} = \frac{A_{ij}}{\sum_k A_{kj}+\epsilon} \]

算法流程就是：

\[ A_0=\operatorname{softmax}_{row}(X)+\epsilon \]

\[ A_0=C_\epsilon(A_0) \]

随后执行 9 次：

\[ A_k=C_\epsilon(R_\epsilon(A_{k-1})) \]

# 第一层加速

把 softmax、+eps、第一次列归一化中的 sum、add、div，和后九次行列归一化中的 sum、add、div 融合成一个 kernel。

调用自定义算子后，将数据从 GM 拷贝到 UB，然后在 UB 中完成10次归一化，最后把结果从 UB 拷贝到 GM。

对于小矩阵运算，节省 kernel launch 的收益比节省浮点运算的收益更高。

# 第二层加速

原来的实现中，对于每个矩阵，都需要经过 

1. copyin
2. compute
3. copyout

我们可以一次读取64个矩阵，来减少读取和写回的次数。

# GM布局

GM 中每个矩阵是紧凑的 16 个 float：

```
r0: a00 a01 a02 a03
r1: a10 a11 a12 a13
r2: a20 a21 a22 a23
r3: a30 a31 a32 a33
```

但昇腾一个 32B block 能容纳 8 个 FP32，而一行只有 4 个 FP32，也就是16B。

最终版在 UB 中把每一行补到 8 个元素：

```
r0: a00 a01 a02 a03 p p p p
r1: a10 a11 a12 a13 p p p p
r2: a20 a21 a22 a23 p p p p
r3: a30 a31 a32 a33 p p p p
```

所以每个矩阵在 UB 中占：

\[ 4\text{ 行}\times8\text{ float}=32\text{ float}=128B \]

索引为：

\[ z[32m+8r+c] \]

其中 `m` 是矩阵编号，`r` 是行号，`c<4` 是有效数据，`c>=4` 是填充位。

这会浪费一半 UB 空间，但换来了三个重要收益：

- 每行正好对齐一个 32B block。
- 行归约可以直接使用 `BlockReduceSum`。
- ***广播、列求和和除法都可以通过 block stride 表达。***

最终五个 UB 缓冲区合计约 54 KiB：

- `zBuf`：16 KiB
- `bcastBuf`：16 KiB
- `epsBuf`：16 KiB
- `rowBuf`：2 KiB
- `colBuf`：4 KiB

这是典型的“用片上空间换规则布局和指令效率”。

# 直接消除填充位对规约的影响

如果填充值直接是 0，那么 softmax 的 `Exp(0)=1`，填充位会污染行和。

原实现只能：

1. 对 8 个元素归约；
2. 从向量结果中 `GetValue`；
3. 在标量端减掉 4 个填充值的贡献；
4. 再 `SetValue` 写回；
5. 做 V→S、S→V 同步。

这正是原实现大量标量—向量同步的来源。

最终版在 CopyIn 时把填充位设置为 `-1000`：

在 FP32 下：

\[ e^{-1000}=0 \]

而且是下溢到精确的 0。此后：

- 加法归约：0 是单位元；
- 行和、列和不会被污染；
- `0/divisor=0`；
- padding 在全部迭代中保持为 0。

因此不再需要 `GetValue/SetValue`、标量修正和 S/V 同步。

# 行归一化处理

最终的 `RowNormalize` 只有：

```
BlockReduceSum
Adds
Brcb
Div
```

具体过程：

1. `BlockReduceSum`

   每个 8-float block 求和。因为后四个 padding 是 0，所以结果正好是原始4个元素的行和。

2. `Adds`

   给每个行和加上 `eps。

3. `Brcb`

   把每个行和广播回对应的8元素行。

4. `Div`

   整行除以广播后的分母。有效位得到归一化结果，padding 仍为0。

对于 一次处理64个矩阵的情况，一次 `BlockReduceSum` 调用通过 repeat 覆盖 256 行，而不是逐行调用256次。

# 列归一化处理

UB block 排列为：

```
M0:r0 M0:r1 M0:r2 M0:r3
M1:r0 M1:r1 M1:r2 M1:r3
...
```

同一个矩阵相邻行之间相差1个 block；不同矩阵的同一行之间相差4个 block。

因此使用：

```
mask = 64
blkStride = 4
repeat = nE / 8
```

就可以让一次 repeat 同时选中8个矩阵的同一行。

列和计算为：

```
colSum = row0 + row1;
colSum += row2;
colSum += row3;
```

然后循环四个行组：

```
row[r] /= colSum + eps;
```

这里最值得学习的是：向量化不一定要求把数据真的转置成 SoA。只要底层 API 支持 block stride，就可以在 padded-AoS 上“逻辑地访问同一列”。

# eps张量生成

参考语义要求：

```
x = softmax(x) + eps
```

但只能给4个有效位置加 `eps`，padding 必须加0，即需要：

```
[eps eps eps eps 0 0 0 0]
```

早期方案从 GM 读取常量数组，再通过 `DataCopyPad` 展开。对于这样一个极小算子，DMA 描述符和跨 pipeline 同步成本反而非常昂贵。

最终版利用 CopyIn 后的状态：

```
有效位：x
填充位：-1000
```

执行：

```
m = z + 900
m = max(m, 0)
m = min(m, 1)
m = m * eps
```

于是：

| 状态     | 有效位 | 填充位 |
| -------- | ------ | ------ |
| 初始     | x      | -1000  |
| `+900`   | x+900  | -100   |
| `max(0)` | 正数   | 0      |
| `min(1)` | 1      | 0      |
| `*eps`   | eps    | 0      |

代码见 [BuildEpsVectorArith (line 140)](/Users/gemaymini/Desktop/sinkhorn_claude/【2026KernelSwift算子创新大赛】-test-Task03_sinkhorn-111/op_kernel/sinkhorn_normalize_kernel.asc:140)。

这相当于用输入和 padding 的数值区间“现场构造掩码”，做到：

- 零 GM 读取；
- 零 MTE 描述符；
- 不额外保存显式布尔 mask；
- 4条向量算术覆盖整个 tile。

# 同步优化

原实现几乎每条向量指令后都放：

```
PipeBarrier<PIPE_V>();
```

但同一个 V pipeline 内的向量指令本来就按序发射。无差别 barrier 会让整个流水线完全串行。

最终版删除这些 V-pipe barrier，只在跨 pipeline 时显式同步：

- `MTE2 → V`：CopyIn 完成后才能计算；
- `V → MTE3`：计算完成后才能 CopyOut；
- `MTE3 → MTE2`：写回完成后，下一 tile 才能复用同一 UB。

# tilling优化

Tiling 代码根据目标矩阵数反推核心数：

```
wanted = ceil(totalMats / tileTarget);
cores = min(availableCores, wanted);
```

官方输入有1024个矩阵，`tileTarget=64`，因此在可用核数不少于16时：

\[ wanted=\lceil1024/64\rceil=16 \]

最终通常是：

- 16个 block/核心任务；
- 每个处理64个矩阵；
- 每核只处理一个 tile。

没有使用全部可用向量核，是因为每核矩阵数太少会使：

- 向量 repeat 变短；
- 每个 block 的固定启动成本占比上升；
- DMA、事件和初始化成本难以摊薄。

真机扫描发现：

- 每核8/16/32个矩阵：固定开销约136 μs；
- 每核64/128个矩阵：约113 μs；
- 64和128基本持平。

所以选64是吞吐、并行度和固定开销之间的平衡点。

尾块则向上补齐到8个矩阵：

```
nE = (n + 7) & ~7
```

虚构的尾部矩阵不会写回，也不会和真实矩阵发生归约混合，因此可以安全参与向量计算。

# Host 优化

官方计时包住完整 `forward()`，然后同步，所以 Host 开销全部计入成绩。

原绑定层每次调用都做：

- 查询设备和向量核数；
- 重新计算 tiling；
- 分配一个很小的 tiling tensor；
- 同步执行一次 H2D `aclrtMemcpy`。

同步 memcpy 不只是复制32字节，它还可能等待 stream 排空，因此成本远高于数据本身。

最终版使用静态缓存，见 [sinkhorn_normalize_torch.cpp (line 44)](/Users/gemaymini/Desktop/sinkhorn_claude/【2026KernelSwift算子创新大赛】-test-Task03_sinkhorn-111/op_extension/sinkhorn_normalize_torch.cpp:44)：

- 核数只查询一次；
- tiling 按 `(totalMats, repeat, eps)` 缓存；
- device tiling buffer 只分配一次；
- 只有缓存未命中时才 H2D；
- warmup 后的稳态路径零 tiling 分配、零 tiling H2D、零设备查询。

实测 Host 耗时曾从约114 μs降到40 μs，对加速比贡献约17%。