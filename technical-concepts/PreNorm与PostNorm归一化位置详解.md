# Pre-Norm 与 Post-Norm：Transformer 归一化位置详解

## 1. 二者究竟区别在哪里？

Pre-Norm 和 Post-Norm 描述的是：**归一化层放在子层变换之前，还是放在子层输出与残差相加之后。**

设输入为 $x$，子层变换为 $F$，归一化操作为 $\mathcal N$。

### Pre-Norm

$$
\boxed{y=x+F(\mathcal N(x))}
$$

```text
x ────────────────────────┐
  └── Norm ── F ──────────┤ 加法 → y
```

归一化位于残差分支的内部，先归一化输入，再送入 Attention 或 FFN。

### Post-Norm

$$
\boxed{y=\mathcal N(x+F(x))}
$$

```text
x ─────────────────┐
  └── F ───────────┤ 加法 ── Norm → y
```

先完成子层变换和残差相加，再归一化整个结果。

**Pre/Post 是相对于残差子层结构而言，不是只看 Norm 是否出现在某个线性层的前面或后面。**

本文讨论经典 Pre-Norm/Post-Norm；实际架构还可能使用额外归一化、残差缩放或混合布局。

## 2. 放入完整 Transformer 块

一个 Transformer 块通常包含 Attention 和 FFN 两个残差子层，二者一般各有自己的 Norm 参数。

### 2.1 Pre-Norm 块

$$
h=x+\operatorname{Attention}(\mathcal N_1(x))
$$

$$
y=h+\operatorname{FFN}(\mathcal N_2(h))
$$

```text
输入 x
  ├────────────── 残差 ──────────────┐
  └── Norm1 → Attention ──────────── 加法 → h
                                      ├──────── 残差 ────────┐
                                      └── Norm2 → FFN ───── 加法 → y
```

深层堆叠后通常还会在预测头之前设置一个最终 Norm，用来控制主干最终输出的尺度。这是常见做法，而非由 Pre-Norm 定义强制规定。

### 2.2 Post-Norm 块

$$
h=\mathcal N_1\left(x+\operatorname{Attention}(x)\right)
$$

$$
y=\mathcal N_2\left(h+\operatorname{FFN}(h)\right)
$$

每个残差子层相加后的结果都经过归一化。最初的 Transformer 使用这种典型布局。

## 3. 先理解 Norm：它通常沿哪个维度计算？

设输入：

$$
X\in\mathbb R^{B\times L\times D}
$$

典型 Transformer 的 LayerNorm 或 RMSNorm 沿最后的特征维 $D$ 计算：

```text
对每个 batch、每个 token，独立归一化它的 D 维表示。
```

它通常不跨 batch 统计，也不将整个序列混在一起归一化。因此不要将这里的 Norm 与 BatchNorm 混为一谈。

### 3.1 LayerNorm

对单个 token 的向量 $x\in\mathbb R^D$：

$$
\mu=\frac1D\sum_{i=1}^D x_i
$$

$$
\sigma^2=\frac1D\sum_{i=1}^D(x_i-\mu)^2
$$

$$
\operatorname{LN}(x)_i
=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i
$$

归一化后再施加可训练缩放与偏置。所以“LayerNorm 输出始终严格零均值、单位方差”只适用于忽略 $\epsilon$ 且未施加一般仿射变换的理想化情况。

### 3.2 RMSNorm

$$
\operatorname{RMSNorm}(x)_i
=\gamma_i\frac{x_i}{\sqrt{\frac1D\sum_jx_j^2+\epsilon}}
$$

常见形式不减均值，也不带加性偏置。

### 3.3 两个相互独立的设计轴

- Pre-Norm/Post-Norm：归一化放在哪里；
- LayerNorm/RMSNorm：采用什么归一化公式。

因此 Pre-Norm 不等于 RMSNorm，Post-Norm 也不等于 LayerNorm。它们可以按架构需求组合。

## 4. 一个简单前向计算例子

为只观察位置差异，设：

$$
x=[1,3]
$$

取 $F(z)=z$，Norm 使用无仿射变换的 LayerNorm，并暂时忽略 $\epsilon$。

这只是教学例子，不代表真实 Attention/FFN 等于恒等映射。

输入的均值为 2，方差为 1，因此：

$$
\mathcal N(x)=[-1,1]
$$

### Pre-Norm

$$
y=x+F(\mathcal N(x))
=[1,3]+[-1,1]=[0,4]
$$

### Post-Norm

$$
y=\mathcal N(x+F(x))
=\mathcal N([2,6])=[-1,1]
$$

二者显然不是交换一下执行顺序而结果不变：

- Pre-Norm 归一化的是送入分支的输入，残差主干本身没有被该 Norm 直接归一化；
- Post-Norm 对残差相加后的整体结果归一化。

## 5. 为什么归一化位置会影响深层训练？

关键在于**残差主干的前向尺度和反向梯度路径**。

下面使用雅可比矩阵表示局部导数。为简洁起见，省略参数下标以及 dropout 等操作。

### 5.1 Pre-Norm：有一条不经过当前 Norm 的恒等路径

$$
x_{l+1}=x_l+F_l(\mathcal N_l(x_l))
$$

局部雅可比为：

$$
\frac{\partial x_{l+1}}{\partial x_l}
=I+J_{F_l}J_{\mathcal N_l}
$$

这里有明确的 $I$ 项。梯度可以经由直接残差路径传递，而不必通过当前子层的归一化变换。

从多层前向看：

$$
x_L=x_0+\sum_{l=0}^{L-1}F_l(\mathcal N_l(x_l))
$$

主干保留原始状态，并累加各层更新。

**这有利于深层优化，但不保证梯度永不爆炸或消失。**多个分支的雅可比、输出尺度、损失与优化器仍可能造成不稳定；最终 Norm 或输出头也仍会影响整体梯度。

### 5.2 Post-Norm：相加后的主干也经过 Norm

$$
x_{l+1}=\mathcal N_l(x_l+F_l(x_l))
$$

局部雅可比为：

$$
\frac{\partial x_{l+1}}{\partial x_l}
=J_{\mathcal N_l}(I+J_{F_l})
$$

注意 $J_{\mathcal N_l}$ 乘在整个残差结果外侧：即使走残差路径，也要经过这一层 Norm 的导数。

多层堆叠时，梯度反复受到这些归一化雅可比影响。相比典型 Pre-Norm，经典 Post-Norm 在深层或较大学习率下往往更依赖合适的初始化、warmup 和残差控制。

但不能将其简单说成“Post-Norm 的梯度必然消失”。归一化的导数会随方向和输入尺度变化，也可能放大某些分量；具体训练行为依赖完整配置。

## 6. Norm 的导数为什么不等于恒等映射？

LayerNorm 会消除输入整体平移的影响。忽略仿射参数时：

$$
\operatorname{LN}(x+c\mathbf 1)=\operatorname{LN}(x)
$$

所以沿全 1 向量的变化不改变输出，该方向不可能原样传递。

对输入尺度的变化，LayerNorm/RMSNorm 在忽略 $\epsilon$、正比例缩放时也具有尺度不变性。这意味着 Norm 会重新分配或压制某些变化方向，而不是仅做一个固定标量缩放。

这解释了为什么：

```text
I + 分支导数
```

与：

```text
Norm 导数 × (I + 分支导数)
```

是不同的梯度结构。

现实中还要考虑 $\epsilon$ 和可训练缩放参数，不能直接把所有方向都描述为严格投影。

## 7. Pre-Norm 的代价：主干尺度可能随深度增长

Pre-Norm 规范化的是分支输入，不是相加后的整个主干：

$$
x_{l+1}=x_l+\Delta_l
$$

若为说明而假设各层增量近似不相关、零均值且每层方差约为 $c$，则：

$$
\operatorname{Var}(x_L)\approx\operatorname{Var}(x_0)+Lc
$$

因此标准差可能按约 $\sqrt L$ 增长。这是简化分析，不是对任意训练模型的严格增长定律。

它可能带来：

- 深层主干尺度变大；
- 单层更新相对主干的比例变小；
- 后续层有时表现得更像小幅修正；
- 深度增加未必自动转化为等比例的有效表征增益。

应把这些看作可能出现的现象，而不是断言“Pre-Norm 后面的层都无用”。常见应对方式包括最终 Norm、残差缩放、合适的初始化，以及宽度/深度相关的参数化。

## 8. 残差缩放如何与 Pre-Norm 配合？

可在分支前添加缩放系数：

$$
x_{l+1}=x_l+\alpha_lF_l(\mathcal N_l(x_l))
$$

此时梯度为：

$$
\frac{\partial x_{l+1}}{\partial x_l}
=I+\alpha_lJ_{F_l}J_{\mathcal N_l}
$$

保留恒等路径，同时控制分支增量。

例如，一个包含 $N$ 个 Transformer 块、每块两个残差分支的架构，可以使用：

$$
\alpha=\frac1{\sqrt{2N}}
$$

在理想化的独立增量分析下，该缩放有助于限制累积方差。实际需要与初始化、学习率和其他稳定化设计配合；这不是所有 Pre-Norm 模型的必选公式，也不能单独保证训练稳定。

[OneTrans-V2 原文](../../OneTransV2.2609.28589v1.pdf) §4.3 的 TransBlock 使用 pre-norm 布局、RMSNorm，以及该形式的分支残差乘数。其完整稳定化方案还包括 QKNorm、门控注意力和参数化/优化器设计，不能把效果全部归因于 Pre-Norm。

## 9. 哪一种更好？

| 维度 | 典型 Pre-Norm | 典型 Post-Norm |
|---|---|---|
| 子层公式 | $x+F(\mathcal N(x))$ | $\mathcal N(x+F(x))$ |
| Norm 作用对象 | 分支输入 | 残差相加后的整体结果 |
| 残差直通路径 | 存在不经过当前 Norm 的路径 | 相加后仍经过当前 Norm |
| 深层优化 | 通常更容易稳定训练 | 往往更依赖初始化、warmup 和残差设计 |
| 主干尺度 | 可能随深度增长 | 每个子层后受 Norm 调整 |
| 最终 Norm | 常见 | 取决于架构，一般末尾子层已有 Norm |
| 效果判断 | 依赖完整训练与模型设置 | 依赖完整训练与模型设置 |

实践上，深层大模型常用 Pre-Norm 作为稳定训练的基础，但不是“Post-Norm 永远较差”的定理。

对具体任务，应在合理匹配参数、训练预算和超参数的条件下比较。直接把一个布局换成另一个，沿用所有学习率、初始化和最终 Norm 设置，可能得到失真的结论。

## 10. 与 QKNorm、Gated Attention 的区别

### Pre-Norm / Post-Norm

描述 Transformer 残差子层中 Norm 的位置。

### QKNorm

对投影得到的 Query、Key 再做归一化，通常按头操作，以控制注意力 logit 的尺度。

即使分支输入已经经过 RMSNorm，Q/K 投影的范数仍会影响注意力分数，所以 QKNorm 不因 Pre-Norm 而自动多余。

### Gated Attention

对注意力输出施加输入相关门控，控制写回残差流的内容。这不是 Norm，也不是残差布局。

### SwiGLU

是 FFN 的内部门控结构，可以放入 Pre-Norm 或 Post-Norm 块。详见 [SwiGLU 门控前馈网络详解](./SwiGLU门控前馈网络详解.md)。

## 11. 概念代码

```python
# 概念伪代码：省略 dropout、位置编码、mask 和残差缩放

def pre_norm_block(x):
    h = x + attention(norm1(x))
    y = h + ffn(norm2(h))
    return y


def post_norm_block(x):
    h = norm1(x + attention(x))
    y = norm2(h + ffn(h))
    return y
```

实际识别时不要只搜索 `norm` 的调用位置，要检查它与加法、残差分支、最终输出归一化的完整数据流。

## 12. 常见误解

1. **“Pre-Norm 是在加残差之前做 Norm。”**这不够精确；经典 Pre-Norm 是对子层输入先做 Norm，随后运行子层，最后才加残差。
2. **“Pre-Norm 不会有梯度问题。”**它保留较直接的路径，但整体稳定性仍受分支和优化配置影响。
3. **“Post-Norm 的梯度一定消失。”**不是；关键是各层 Norm 改变梯度传递结构，实际可能放大也可能压缩不同方向。
4. **“Pre-Norm 就是 RMSNorm。”**一个是位置，一个是公式。
5. **“有 Pre-Norm 就不需要 QKNorm。”**两者作用在不同位置，控制不同尺度。
6. **“Norm 会把所有 token 混起来。”**典型 LayerNorm/RMSNorm 沿单 token 的特征维归一化。
7. **“Post-Norm 每层输出范数完全固定。”**可训练仿射参数、$\epsilon$ 和所用 Norm 形式都会影响这一说法。
8. **“换个位置，参数一样，所以训练配置也可以完全不变。”**布局改变了前向尺度和梯度结构，可能需要重新匹配训练配置。

## 总结

**Pre-Norm 先规范化分支输入，再把更新加回未直接归一化的残差主干；Post-Norm 则规范化整个残差相加结果。前者保留更直接的恒等梯度路径，通常更利于深层优化，但也需要处理主干尺度增长；后者逐层调整主干尺度，却让残差梯度同样经过 Norm。**
