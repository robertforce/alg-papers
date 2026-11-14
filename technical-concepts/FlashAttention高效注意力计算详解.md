# FlashAttention：高效注意力计算的概念、原理与计算过程

## 1. FlashAttention 是什么？

FlashAttention 是一种 **IO-aware（感知内存读写成本）的精确注意力算法及 GPU 实现方法**。

> 它不改变标准注意力的数学目标，而是改变计算顺序：将 Q/K/V 分块，在片上存储中完成分数、softmax 和 Value 加权，避免把完整的注意力矩阵写入显存；反向传播时再按需重计算中间量。

它主要优化的是 **HBM（GPU 高带宽显存）与片上存储之间的数据搬运，以及中间结果的存储成本**。

需要先记住三个边界：

1. FlashAttention 不是一种新的注意力语义，通常仍计算标准 scaled dot-product attention。
2. 它不是通过只看部分历史来提速，因此不等于稀疏或近似注意力。
3. 对一般稠密注意力，计算量仍是二次量级；主要减少的是中间矩阵存储和内存读写，并改善硬件执行效率。

本文首先解释 FlashAttention 的核心原理，再简述后续版本的优化方向。公式以无 dropout 的注意力为主，省略 batch 和头维度。

## 2. 标准 Attention 为什么昂贵？

设：

$$
Q\in\mathbb R^{N_q\times d},\qquad
K,V\in\mathbb R^{N_k\times d}
$$

为简化记号，假设 Value 的头维度也为 $d$。标准计算为：

$$
S=\frac{QK^\top}{\sqrt d}+M
$$

$$
P=\operatorname{softmax}_{\text{row}}(S)
$$

$$
O=PV
$$

其中 $M$ 为 mask 或可表示为分数偏置的项。

朴素实现通常按多个算子执行：

```text
Q、K 从显存读取
    ↓ 矩阵乘
完整 S 写入显存
    ↓ 读取 S，执行 mask/softmax
完整 P 写入显存
    ↓ 读取 P、V，执行矩阵乘
O 写入显存
```

问题不只是计算 $QK^\top$ 很多，而是 $S$ 和 $P$ 的逻辑尺寸都是：

$$
[N_q,N_k]
$$

在自注意力 $N_q=N_k=N$ 时，尺寸随 $N^2$ 增长。

### 一个容量例子

假设：

```text
序列长度 N = 8192
batch B = 1
Query 头数 H = 32
每个元素 2 字节
```

仅一份完整分数或概率矩阵就需要：

$$
BH N^2\times2
=1\times32\times8192^2\times2
=4\ \mathrm{GiB}
$$

这是“如果以两字节存储一份完整矩阵”的容量估算，不是所有框架固定的实际峰值。中间精度、内存复用、反向保存策略等都会改变实际占用。

即使 GPU 能容纳，反复写入、读取如此大的矩阵，也会带来显著的时间和带宽成本。

## 3. 核心思路：矩阵存在于数学上，不必完整存在于显存中

FlashAttention 使用三个相互配合的设计：

1. **Tiling：**将 Q/K/V 拆成能高效处理的小块；
2. **Online Softmax：**跨块维护精确的 softmax 归一化统计量；
3. **融合与重计算：**前向不保存完整概率矩阵，反向按块重新计算它。

```text
读取一块 Q
  ├── 读取一块 K/V
  │      ↓ 分数块 → 在线 softmax 更新 → 输出累加
  ├── 读取下一块 K/V
  │      ↓ 分数块 → 在线 softmax 更新 → 输出累加
  └── ...
最终只写回输出和少量归一化统计量
```

这里“在片上计算”是算法层面的描述。实际 GPU kernel 会结合共享内存、寄存器、Tensor Core 操作数布局等安排数据，不应理解成所有临时结果都放在一种固定存储中。

## 4. 第一个难点：softmax 如何分块？

普通 softmax 的分母依赖整行：

$$
p_i=\frac{e^{s_i}}{\sum_j e^{s_j}}
$$

如果只看到一部分 Key，就不知道完整分母。

因此不能这样做：

```text
第一块单独 softmax → 输出 o1
第二块单独 softmax → 输出 o2
把 o1、o2 直接相加或平均
```

这样通常不等于对所有 Key 一起做 softmax。

FlashAttention 的关键是：**不保留整行分数，但保留足以合并所有块的统计量。**

## 5. 在线 softmax：维护最大值、分母和加权和

先只看一个 Query，其与全部 Key 的分数是 $s_1,\ldots,s_{N_k}$。

为了数值稳定，标准 softmax 会减去最大值：

$$
m=\max_i s_i
$$

最终输出为：

$$
o=\frac{\sum_i e^{s_i-m}v_i}{\sum_i e^{s_i-m}}
$$

对于已经处理过的 Key，维护三个量：

- $m$：已处理分数的最大值；
- $l$：以 $m$ 为基准的指数和；
- $u$：以 $m$ 为基准的未归一化 Value 加权和。

即：

$$
l=\sum_{i\in\text{processed}}e^{s_i-m}
$$

$$
u=\sum_{i\in\text{processed}}e^{s_i-m}v_i
$$

### 5.1 来了一个新分数块

记新块的最大值为：

$$
m_b=\max_{i\in\text{block}}s_i
$$

统一基准为：

$$
m'=\max(m,m_b)
$$

旧统计量原来使用 $m$，现在要改成 $m'$，所以乘：

$$
\alpha=e^{m-m'}
$$

更新分母：

$$
l'=\alpha l+\sum_{i\in\text{block}}e^{s_i-m'}
$$

更新分子：

$$
u'=\alpha u+\sum_{i\in\text{block}}e^{s_i-m'}v_i
$$

最后令：

$$
m\leftarrow m',\quad l\leftarrow l',\quad u\leftarrow u'
$$

全部块处理完后：

$$
\boxed{o=u/l}
$$

数学上，这与对整行一次性 softmax 完全一致；浮点运算下可能因求和顺序不同产生微小差异。

### 5.2 为什么这个合并成立？

旧分子中的每一项：

$$
e^{s_i-m}v_i
$$

乘以 $e^{m-m'}$ 后变成：

$$
e^{s_i-m'}v_i
$$

因此旧块与新块都转换成了同一个指数基准。将它们的分子和分母分别相加，就恢复了全体 Key 的加权和。

初始化可以理解为 $m=-\infty,l=0,u=0$。具体实现需要正确处理全屏蔽块，避免出现 $-\infty-(-\infty)$ 导致 NaN；可跳过这些块或使用专门的数值分支。

## 6. 一个分两块计算的数值例子

设一个 Query 对四个 Key 的分数为：

$$
s=[1,2,3,4]
$$

为便于计算，Value 使用标量：

$$
v=[10,20,30,40]
$$

### 第一块：分数 [1,2]

$$
m=2
$$

$$
l=e^{-1}+1\approx1.367879
$$

$$
u=10e^{-1}+20\approx23.678794
$$

只看这一块，归一化输出约为 17.31，但这还不是最终答案。

### 第二块：分数 [3,4]

新最大值：

$$
m'=4
$$

旧统计量重标定因子：

$$
\alpha=e^{2-4}=e^{-2}\approx0.135335
$$

新分母：

$$
l'=e^{-2}(e^{-1}+1)+e^{-1}+1
=e^{-3}+e^{-2}+e^{-1}+1
\approx1.553002
$$

新分子：

$$
u'=e^{-2}(10e^{-1}+20)+30e^{-1}+40
\approx54.240960
$$

最终：

$$
o=\frac{54.240960}{1.553002}\approx34.9265
$$

直接对完整 `[1,2,3,4]` 做 softmax 再加权，也得到同一结果。

**处理完一个块后，只需保留 $m,l,u$，不用保留这个块的全部分数或概率。**

## 7. 从单 Query 推广到矩阵分块

假设：

```text
Q 总长度 Nq = 8192
K/V 总长度 Nk = 8192
头维度 d = 128
每块 Query 行数 Bq = 64
每块 Key 行数 Bk = 128
```

这里只是说明形状，真实块大小依赖硬件、dtype、头维度和 kernel 实现。

### 第一步：取一个 Query 块

$$
Q_b\in\mathbb R^{64\times128}
$$

为 64 个 Query 分别维护最大值、指数和和输出累加器：

```text
m：[64]
l：[64]
u：[64, 128]
```

### 第二步：遍历 Key/Value 块

当前：

$$
K_b,V_b\in\mathbb R^{128\times128}
$$

计算分数块：

$$
S_b=Q_bK_b^\top/\sqrt{128}+M_b
$$

```text
S_b：[64, 128]
```

只对这块执行逐行最大值和指数计算，再更新 64 行的统计量及 Value 累加结果。

### 第三步：处理所有 Key/Value 块

同一 Query 块遍历所有允许读取的 K/V 块后，计算：

$$
O_b=u/l
$$

除法按行广播，最终将 $O_b$ 写回显存。

### 第四步：完成所有 Query 块

最终得到完整输出：

$$
O\in\mathbb R^{8192\times128}
$$

概念伪代码如下。它用于解释数学过程，不代表某个版本的实际线程调度：

```python
for Q_block in Q_blocks:
    m = negative_infinity_per_row()
    l = zeros_per_row()
    u = zeros_like_output_block()

    for K_block, V_block in visible_KV_blocks:
        s = Q_block @ K_block.T / sqrt(d)
        s = apply_mask_and_bias(s)
        # 全屏蔽行/块需要特殊处理，此处省略工程分支
        m_new = maximum(m, rowmax(s))
        alpha = exp(m - m_new)
        p = exp(s - m_new[:, None])

        l = alpha * l + rowsum(p)
        u = alpha[:, None] * u + p @ V_block
        m = m_new

    O_block = u / l[:, None]
    LSE_block = m + log(l)
    write_output_and_lse(O_block, LSE_block)
```

注意：不同算法版本可能调整循环顺序、维护归一化输出还是未归一化累加器，以及统计量保存形式；不能将这个解释性循环视为所有实现的统一代码。

## 8. 反向传播：不保存 P，梯度从哪里来？

普通实现为了反向传播，可能保留完整 $P$。FlashAttention 选择保存较小的中间信息，如输出 $O$ 和每行 log-sum-exp：

$$
L_i=\log\sum_j e^{S_{ij}}
$$

反向时按块重算：

$$
S_{ij}=q_i^\top k_j/\sqrt d+M_{ij}
$$

$$
P_{ij}=e^{S_{ij}-L_i}
$$

因此不需要把完整 $P$ 存在显存中等待反向。

### 8.1 梯度公式

设上游梯度为 $dO$，无 dropout 时：

$$
dV=P^\top dO
$$

$$
dP=dO\,V^\top
$$

softmax 反向为：

$$
dS_{ij}=P_{ij}\left(dP_{ij}-\sum_t P_{it}dP_{it}\right)
$$

其中行内归约项可写为：

$$
\sum_t P_{it}dP_{it}=dO_i\cdot O_i
$$

所以它也能通过每行较小的统计量计算，不需要完整概率矩阵。

最后：

$$
dQ=dS\,K/\sqrt d
$$

$$
dK=dS^\top Q/\sqrt d
$$

这些操作按块执行、累加梯度。

### 8.2 为什么多算一遍反而可能更快？

因为重新做一些高吞吐矩阵计算，可能比将巨大中间矩阵写入显存、长期保存、再读回来更便宜。

这是 **以重计算换显存与 IO**，与 activation checkpointing 思想相近，但针对 Attention 的结构做了专门设计。

这不代表整个网络无需保存激活；Q/K/V、输出以及其他层的训练状态仍要按具体反向策略管理。

若启用 dropout，还需要可重现对应随机掩码，并使用匹配的梯度公式。上面的无 dropout 推导不能不加修改地用于所有 dropout 实现。

## 9. 时间复杂度、空间复杂度和 IO：三者不要混淆

### 9.1 算术复杂度

一般稠密自注意力仍为：

$$
O(N^2d)
$$

FlashAttention 没有把标准全注意力变成线性计算。

### 9.2 中间存储

朴素实现显式物化的分数/概率矩阵是 $O(N^2)$。

FlashAttention 避免在 HBM 保存这些完整矩阵；除块内临时空间外，输出及归一化统计量随序列长度线性增长。常说的“线性内存”针对这类注意力中间存储，不表示整个模型训练显存只有 $O(N)$。

### 9.3 HBM 读写

通过块内融合，避免了完整 $S/P$ 的反复显存读写。它不意味着所有 Q/K/V 数据只从 HBM 读一次：不同 Query 块仍可能重新读取 K/V 块，具体取决于调度与缓存复用。

### 9.4 实际延迟

会受到序列长度、头维度、batch、因果 mask、硬件、精度、占用率和 kernel 版本影响。

因此不能给出与配置无关的固定“加速几倍”。短序列或单 token 解码也不一定适合与长序列训练完全相同的 kernel。

## 10. 因果 mask 和特殊 mask 怎么处理？

### 因果注意力

未来 Key 的位置设为不可见，相应分数为 $-\infty$。分块实现可以：

- 跳过全部属于未来位置的块；
- 对跨越因果边界的块做局部 mask；
- 对完全可见的块正常计算。

跳过本来就被 mask 的位置不改变结果，也不意味着改成另一种注意力语义。

### 更复杂的业务 mask

例如 OneTrans-V2 中“曝光只读请求时可见历史、不同曝光和阶段相互隔离”的 mask，可以由定制算子根据锚点和任务布局生成，不一定要显式保存一个完整布尔矩阵。

但不是任意 FlashAttention API 都直接支持任意业务 mask。若用户显式构造并保存一个 $N\times N$ 的 mask，它自身仍有二次空间成本；要维持整体收益，mask 的表示和生成方式也需优化。

[OneTrans-V2 原文](../../OneTransV2.2609.28589v1.pdf) §5.2 报告使用内置阶段可见性 mask 的 FlashAttention 风格算子。这是论文的定制使用场景，不等于通用接口天然支持该布局。

## 11. 与 GQA、KV Cache、SwiGLU、MoE 的区别

| 技术 | 改变什么 | 主要收益 |
|---|---|---|
| FlashAttention | Attention 的计算调度、中间存储与 IO | 减少大中间矩阵读写与保存 |
| GQA | Query 头与 KV 头的对应关系 | 减少 K/V 投影、缓存和读取 |
| KV Cache | 复用过去 token 的 K/V | 自回归解码时避免重复计算历史 K/V |
| SwiGLU | FFN 内部的非线性与门控结构 | 增强逐 token 特征变换 |
| MoE | 当前 token 选择哪些专家 FFN | 用稀疏激活扩展总参数容量 |

### FlashAttention 不自动缩小 KV Cache

标准 KV Cache 主要尺寸为：

$$
2\times N_{layers}\times B\times N_k\times H_{kv}\times d
$$

FlashAttention 优化的是如何读取它并计算 Attention，通常不改变这些历史 K/V 本身的数量。GQA 才是通过减少 $H_{kv}$ 直接压缩缓存的设计。

### FlashAttention 与 GQA 可以叠加

先用 GQA 让每组 Query 共享一套 K/V，再由支持 GQA 的高效 Attention kernel 分块读取和计算。应避免为了套用普通接口而物理展开全部 KV 头，否则可能抵消部分带宽与临时内存收益。

相关文档：

- [GQA 分组查询注意力详解](./GQA分组查询注意力详解.md)
- [SwiGLU 门控前馈网络详解](./SwiGLU门控前馈网络详解.md)
- [MoE 混合专家模型详解](./MoE混合专家模型详解.md)

## 12. Prefill 与 Decode：为什么受益方式不同？

### Prefill / 训练

许多 Query 同时读取很长的 Key 序列，逻辑注意力矩阵较大，避免物化矩阵的收益突出；块内矩阵乘也更容易利用高吞吐计算单元。

### 单 token Decode

每条序列每步只有一个新 Query，此时逻辑分数矩阵是 $1\times N_k$，不是 $N_k\times N_k$。

主要瓶颈经常转向读取长 KV Cache 的带宽，以及如何让足够多 GPU 单元并行工作。可能需要面向解码的专用调度，如按 KV 序列分割计算再合并统计量。

在线 softmax 的统计量合并也适用于这种分割，但不能据此认定训练用 FlashAttention kernel 在所有解码场景都最优。

## 13. FlashAttention 不同版本如何理解？

核心思想不变，后续版本继续优化 GPU 执行方式：

- **FlashAttention 初始设计：**IO-aware 分块、在线 softmax、避免物化完整注意力矩阵，以及反向重计算。
- **FlashAttention-2：**进一步优化工作划分和并行度，减少部分非矩阵乘操作与线程块内通信，提高 GPU 利用率。
- **FlashAttention-3：**面向 Hopper 等对应硬件路径，进一步利用异步执行、计算/数据搬运重叠和低精度能力。

这些是版本设计方向的概述，不代表所有 GPU、dtype 和功能组合都支持同样能力。具体接口、支持的 mask、头维度和硬件要求应以所用实现版本为准。

## 14. 常见误解

1. **“FlashAttention 是近似 Attention。”**核心算法针对标准注意力给出精确数学结果，允许常见浮点舍入差异；某些低精度路径还会引入对应精度误差。
2. **“完全不计算注意力矩阵。”**仍计算分数块，只是不在 HBM 物化完整矩阵。
3. **“把二次 Attention 变成线性时间。”**通常没有，线性的是部分中间存储规模。
4. **“K/V 只读取一次。”**不保证；分块调度可能多次读取 K/V，收益来自更好的总体 IO 安排。
5. **“用了就不用 KV Cache。”**两者解决不同问题，可以同时使用。
6. **“需要完整返回 Attention 权重也一样省显存。”**若显式返回所有权重，就必须承担这份二次规模输出的存储成本。
7. **“任意 mask 都无需额外代价。”**是否可高效支持，取决于 mask 结构及 kernel 实现。
8. **“理论 FLOPs 少才会更快。”**减少数据搬运、融合算子与改善并行度，也能在 FLOPs 不降甚至重计算增加时获得加速。

## 总结

**FlashAttention 不改变“Query 对所有可见 Key 做 softmax，再加权 Value”的目标；它把大计算拆成小块，通过可合并的 softmax 统计量及时累加输出，避免让巨大的分数和概率矩阵往返显存。训练时再用按块重计算换取更低的中间存储成本。**
