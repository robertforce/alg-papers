# GQA：分组查询注意力的概念、原理与计算过程

GQA（**Grouped-Query Attention，分组查询注意力**）是多头注意力的一种变体：

> **保留多个独立的 Query 头，但让一组 Query 头共享同一套 Key 和 Value。**

它主要减少 **K/V 投影、KV Cache 容量和解码时的缓存读取量**，尤其适合长序列、自回归生成，以及 OneTrans-V2 中“多个任务/候选读取同一份用户历史”的场景。

## 1. 从普通注意力开始：Q、K、V 分别做什么？

对一个输入 token 的表示 $x$，通过三个可训练投影得到：

$$
q=xW_Q,\qquad k=xW_K,\qquad v=xW_V
$$

可以直观理解为：

- **Query：**当前 token 想找什么信息；
- **Key：**每个被读取 token 提供什么匹配线索；
- **Value：**匹配成功后，实际取回什么信息。

假设当前 Query 要读取 $L$ 个历史 token：

$$
K=\begin{bmatrix}k_1\\\vdots\\k_L\end{bmatrix},
\qquad
V=\begin{bmatrix}v_1\\\vdots\\v_L\end{bmatrix}
$$

先计算匹配分数：

$$
s_i=\frac{q^\top k_i}{\sqrt{d_h}}
$$

再用 softmax 得到权重：

$$
a_i=\frac{\exp(s_i)}{\sum_{j=1}^{L}\exp(s_j)}
$$

最后加权汇总：

$$
o=\sum_{i=1}^{L}a_i v_i
$$

矩阵形式就是：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
$$

其中 $M$ 是因果或其他可见性 mask。

**GQA 不改变这个注意力公式，改变的是多个头如何配置和共享 K/V。**

## 2. MHA、MQA、GQA 的区别

### 2.1 MHA：每个 Query 头都有自己的 K/V 头

普通多头注意力 MHA（Multi-Head Attention）：

```text
Q0 → K0、V0
Q1 → K1、V1
Q2 → K2、V2
Q3 → K3、V3
Q4 → K4、V4
Q5 → K5、V5
Q6 → K6、V6
Q7 → K7、V7
```

如果有 8 个 Query 头，就有 8 个 Key 头和 8 个 Value 头。

每个头可以学习自己的查询方式、匹配空间和信息提取方式。表达灵活，但历史序列需要为每层存储 8 套 K/V。

### 2.2 MQA：所有 Query 头共享一套 K/V

MQA（Multi-Query Attention）：

```text
Q0 ─┐
Q1  │
Q2  │
Q3  ├──→ K0、V0
Q4  │
Q5  │
Q6  │
Q7 ─┘
```

仍有 8 个 Query 头，但只有 1 个 KV 头。

缓存很小，但所有头共用同一种 Key/Value 表示，对表达能力的约束更强。

### 2.3 GQA：分组共享 K/V

例如 8 个 Query 头、2 个 KV 头：

```text
Q0 ─┐
Q1  │
Q2  ├──→ K0、V0
Q3 ─┘

Q4 ─┐
Q5  │
Q6  ├──→ K1、V1
Q7 ─┘
```

每组 4 个 Query 头共享一套 K/V。

设 Query 头数为 $H_q$，KV 头数为 $H_{kv}$，每个 KV 头服务的 Query 头数为 $r$。通常要求：

$$
r=\frac{H_q}{H_{kv}}
$$

为整数。

| 机制 | Query 头数 | KV 头数 | 特点 |
|---|---:|---:|---|
| MHA | $H_q$ | $H_q$ | 完全不共享 |
| GQA | $H_q$ | $1<H_{kv}<H_q$ | 分组共享 |
| MQA | $H_q$ | 1 | 全部共享 |

因此，**MHA 和 MQA 可以看作 GQA 的两个边界情况。**

注意：GQA 分的是**注意力头**，不是把 token 或历史序列分组。

## 3. GQA 的完整计算过程

使用一套具体维度：

```text
batch size B = 2
序列长度 L = 100
模型维度 D = 512
Query 头数 Hq = 8
KV 头数 Hkv = 2
每个头维度 dh = 64
每组 Query 数 r = 4
```

输入：

$$
X\in\mathbb R^{2\times100\times512}
$$

### 3.1 计算 Q、K、V 投影

投影矩阵尺寸：

$$
W_Q\in\mathbb R^{512\times(8\times64)}
=\mathbb R^{512\times512}
$$

$$
W_K,W_V\in\mathbb R^{512\times(2\times64)}
=\mathbb R^{512\times128}
$$

计算：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

得到：

```text
Q：[2, 100, 512]
K：[2, 100, 128]
V：[2, 100, 128]
```

**K/V 不是先生成 8 个头再取平均，而是通常直接投影出 2 个头。**

### 3.2 拆分注意力头

经过 reshape 和维度交换：

```text
Q：[2, 8, 100, 64]
K：[2, 2, 100, 64]
V：[2, 2, 100, 64]
```

如果是普通 MHA，K、V 都是 `[2, 8, 100, 64]`。这就是缓存缩小的来源。

### 3.3 将 Query 头映射到 KV 组

对从 0 开始编号的 Query 头 $h$，定义：

$$
g(h)=\left\lfloor\frac{h}{r}\right\rfloor
$$

这里 $r=4$，所以：

```text
Query 头 0、1、2、3 → KV 头 0
Query 头 4、5、6、7 → KV 头 1
```

第 $h$ 个 Query 头计算：

$$
A_h=\operatorname{softmax}\left(
\frac{Q_hK_{g(h)}^\top}{\sqrt{64}}+M
\right)
$$

由于 $\sqrt{64}=8$，就是：

$$
A_h=\operatorname{softmax}\left(
\frac{Q_hK_{g(h)}^\top}{8}+M
\right)
$$

每个头的分数矩阵尺寸：

$$
[100,64]\times[64,100]=[100,100]
$$

所有 Query 头的注意力权重，逻辑尺寸为：

```text
A：[2, 8, 100, 100]
```

实际高效算子可以不显式保存整个矩阵。

### 3.4 对 Value 加权汇总

每个 Query 头分别计算：

$$
O_h=A_hV_{g(h)}
$$

尺寸：

$$
[100,100]\times[100,64]=[100,64]
$$

全部头输出：

```text
O：[2, 8, 100, 64]
```

### 3.5 拼接多个头并做输出投影

将 8 个头拼接：

$$
O_{\text{concat}}=\operatorname{Concat}(O_0,\ldots,O_7)
$$

尺寸：

```text
O_concat：[2, 100, 512]
```

再计算：

$$
Y=O_{\text{concat}}W_O,
\qquad W_O\in\mathbb R^{512\times512}
$$

最终：

```text
Y：[2, 100, 512]
```

**虽然 K/V 只有 2 个头，最终仍有 8 个 Query 头的输出，并没有缩成 2 个头。**

## 4. 同组共享 K/V，为什么输出还能不同？

> **共享的是被检索的信息表示，不是 Query，也不是注意力权重。**

用一个简化数值例子说明。取头维度 $d_h=2$，同组两个 Query：

$$
q_0=[2,0],\qquad q_1=[0,2]
$$

共享的历史 Key：

$$
K=\begin{bmatrix}1&0\\0&1\end{bmatrix}
$$

共享的历史 Value：

$$
V=\begin{bmatrix}10&0\\0&20\end{bmatrix}
$$

### Query 0 的计算

$$
\frac{q_0K^\top}{\sqrt2}=[\sqrt2,0]
$$

softmax 后约为：

$$
a_0=[0.804,0.196]
$$

输出：

$$
o_0=0.804[10,0]+0.196[0,20]=[8.04,3.92]
$$

### Query 1 的计算

$$
\frac{q_1K^\top}{\sqrt2}=[0,\sqrt2]
$$

softmax 后约为：

$$
a_1=[0.196,0.804]
$$

输出：

$$
o_1=0.196[10,0]+0.804[0,20]=[1.96,16.08]
$$

因此：

```text
同一套 K/V
    + 不同 Query
    → 不同注意力权重
    → 不同输出
```

不过，相较 MHA，同组头不能各自学习完全独立的 K/V 投影。这正是 GQA 用来交换效率的表达约束。

## 5. 自回归生成时，GQA 如何配合 KV Cache？

以生成式召回为例，模型需要逐步生成：

```text
决策前缀 → SID0 → SID1 → SID2
```

### 5.1 为什么要缓存 K/V？

假设已有 $T$ 个历史 token。生成下一个 token 时，它需要读取前面所有位置的信息。

历史 token 的 K/V 在标准因果解码中不会因为后面追加 token 而变化，所以可以缓存。

当前步只需：

1. 计算新 token 的 Q/K/V；
2. 将新 K/V 追加到缓存；
3. 用当前 Query 读取全部缓存；
4. 输出当前位置表示并预测下一个 token。

### 5.2 GQA 的缓存尺寸

单层历史缓存为：

$$
K_{\text{cache}},V_{\text{cache}}
\in\mathbb R^{B\times H_{kv}\times T\times d_h}
$$

当前一个 token 的 Query：

$$
Q_{\text{new}}\in\mathbb R^{B\times H_q\times1\times d_h}
$$

按照头分组读取：

$$
O_h=\operatorname{softmax}\left(
\frac{q_hK_{\text{cache},g(h)}^\top}{\sqrt{d_h}}
\right)V_{\text{cache},g(h)}
$$

历史缓存只需要保留 $H_{kv}$ 个头，而不是 $H_q$ 个头。

### 5.3 缓存到底节省多少？

设层数为 $N$，每个数值占 $b$ 字节。不考虑额外管理开销，KV Cache 容量为：

$$
\boxed{2NBTH_{kv}d_hb}
$$

开头的 2 表示 K、V 各一份。

与同样 Query 头数和头维度的 MHA 相比：

$$
\frac{\text{GQA Cache}}{\text{MHA Cache}}
=\frac{H_{kv}}{H_q}=\frac1r
$$

例如：

```text
层数 N = 32
batch B = 1
历史长度 T = 8192
Query 头 Hq = 32
KV 头 Hkv = 8
头维度 dh = 128
BF16，每个数 2 字节
```

则：

- MHA KV Cache：**4 GiB**；
- GQA KV Cache：**1 GiB**。

即缓存缩小至四分之一。

## 6. GQA 节省的是哪些计算？哪些没有节省？

不能把“KV 头减少 4 倍”理解成“整个 Attention 快 4 倍”。

### 6.1 明显减少的部分

#### K/V 投影

普通 MHA：

$$
W_K,W_V\in\mathbb R^{D\times H_qd_h}
$$

GQA：

$$
W_K,W_V\in\mathbb R^{D\times H_{kv}d_h}
$$

所以 K/V 投影的参数量和乘加量，按 $H_{kv}/H_q$ 缩减。

#### KV Cache 容量

同样按 $H_{kv}/H_q$ 缩减。

#### 解码时的 K/V 内存读取

高效分组算子可以将同一套 K/V 复用于一组 Query 头，降低内存流量。

长上下文、小 batch 解码经常受到内存带宽限制，因此这部分非常重要。实际收益取决于算子和硬件，不能直接等同于理论缓存压缩倍数。

### 6.2 基本没有按分组比例减少的部分

仍然有 $H_q$ 个 Query 头，所以每个头仍要独立计算：

$$
Q_hK_{g(h)}^\top
$$

以及：

$$
A_hV_{g(h)}
$$

这两部分的主要算术量仍分别为：

$$
O(BH_qL_qL_kd_h)
$$

其中 $L_q$ 为 Query 序列长度，$L_k$ 为 Key/Value 序列长度。

**注意力分数的数量也仍按 Query 头数计算。**

另外：

- Q 投影不因此缩小；
- 输出投影不因此缩小；
- FFN 不因 GQA 自动缩小。

| 项目 | GQA 的直接影响 |
|---|---|
| K/V 投影参数与计算 | 减少 |
| KV Cache 容量 | 减少 |
| 解码 K/V 读取量 | 高效实现下减少 |
| Query 头数 | 不变 |
| 注意力权重的逻辑尺寸 | 不变 |
| QK、AV 的主要 FLOPs | 通常不变 |
| FFN 计算 | 不变 |

## 7. 实现时为什么经常看到 repeat_kv？

有些代码会先将 KV 头重复到 Query 头数：

```python
# 形状说明，非完整可执行代码
# 原始形状
q: [B, Hq,  Lq, dh]
k: [B, Hkv, Lk, dh]
v: [B, Hkv, Lk, dh]

# 每个 KV 头重复 r 次
k_expanded = repeat_kv(k, r)
v_expanded = repeat_kv(v, r)

# 逻辑上得到
k_expanded: [B, Hq, Lk, dh]
v_expanded: [B, Hq, Lk, dh]
```

这样可以套用普通多头注意力计算。

但要区分：

- **逻辑重复：**多个 Query 头引用同一套 K/V；
- **物理复制：**真的分配扩大后的 K/V 张量。

高效实现应尽量避免显式物理复制，并让算子直接支持分组读取。否则可能抵消一部分运行时内存和带宽收益。

**持久 KV Cache 应保留紧凑的 $H_{kv}$ 头布局。**

## 8. 放回 OneTrans-V2：GQA 为什么有价值？

结合 OneTrans-V2 的架构，用户长历史需要供以下计算反复读取：

```text
共享用户历史 K/V
   ├── 召回：多个决策/SID beam
   ├── 粗排：大量候选的 Query
   └── 精排：更少候选的多个特征 token Query
```

这里有两个不同层面的共享。

### 第一层：历史计算复用

同一份历史 K/V 被多个任务、候选或 beam 使用。

这是跨任务/候选的共享，关联 SNT 和线上共享上下文。

### 第二层：GQA 的头内共享

每次读取历史时，一组 Query 头共享一个 KV 头。

这是注意力头之间的共享。

二者可以叠加：

> **先避免为每个候选重复编码历史，再用 GQA 缩小那份共享历史缓存。**

另外，只要 KV 头数和头维度兼容，不同阶段可以使用不同数量的 Query 头读取同一份历史 K/V。

例如下面只是维度示意，**不是论文披露的配置**：

```text
共享历史：8 个 KV 头，每头 64 维

粗排： 8 个 Query 头 → 每个 KV 头服务 1 个 Query 头
精排：32 个 Query 头 → 每个 KV 头服务 4 个 Query 头
```

这样可以在共享历史表示的同时，让不同阶段拥有不同的查询计算预算。

论文上下文参见本地 [OneTrans-V2 原文](../../OneTransV2.2609.28589v1.pdf) §4.3。本文中的通用维度与数值例子用于解释原理，不代表论文的实际生产配置。

## 9. 最容易混淆的几个点

1. **GQA 不对 token 分组，而是对 Query 头分组。**
2. **共享 K/V 不等于共享注意力权重。**不同 Query 仍产生不同权重。
3. **GQA 不改变可见范围。**能看哪些历史，由因果或阶段 mask 决定。
4. **GQA 不等于稀疏注意力。**每个 Query 仍可读取所有未屏蔽的历史。
5. **GQA 与 FlashAttention 可以同时使用。**前者改变头结构，后者优化注意力计算和内存访问。
6. **GQA 不能保证效果完全不下降。**它在表达灵活度与效率之间折中，具体效果需要实验验证。

## 总结

**GQA 让多个不同的“查询者”共用一套“索引和内容”，查询结果仍各自独立；主要省的是历史 K/V 的生产、存储与搬运。**
