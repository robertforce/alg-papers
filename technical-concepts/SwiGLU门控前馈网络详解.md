# SwiGLU：门控前馈网络的概念、原理与计算过程

## 1. SwiGLU 是什么？

SwiGLU 是一种用于 Transformer 前馈网络（FFN）的门控结构，名称来自 **Swish + Gated Linear Unit**。

> 同一输入经过两个独立投影，一个分支生成内容，另一个分支经过 SiLU 激活生成门控信号；二者逐元素相乘，再投影回模型维度。

典型公式为：

$$
\operatorname{SwiGLU}(x)
=\left[\operatorname{SiLU}(xW_g)\odot(xW_u)\right]W_d
$$

其中：

- $W_g$：gate projection，门控投影；
- $W_u$：up projection，内容分支的升维投影；
- $W_d$：down projection，降维投影；
- $\odot$：逐元素相乘；
- $\operatorname{SiLU}(z)=z\sigma(z)$。

本文采用行向量右乘权重的记法。具体实现中的权重存储方向可能相反，但计算等价。为突出结构，公式默认省略偏置。

## 2. 它在 Transformer 中的位置

典型 pre-norm Transformer 块可概念化为：

$$
h=x+\operatorname{Attention}(\operatorname{Norm}(x))
$$

$$
y=h+\operatorname{FFN}(\operatorname{Norm}(h))
$$

SwiGLU 通常用来实现第二行的 FFN。

- Attention 负责跨 token 读取和汇总信息；
- FFN 对每个 token 的表示进行非线性变换；
- SwiGLU 是 FFN 的一种具体结构。

**SwiGLU 自身不让一个 token 读取其他 token。**序列中的各位置独立应用同一套前馈变换。若架构使用 token 专属参数或 MoE，参数选择方式会变化，但这一逐 token 计算性质仍成立。

## 3. 从普通 FFN 到 GLU，再到 SwiGLU

### 3.1 普通 FFN

普通 Transformer FFN 使用两次线性投影：

$$
\operatorname{FFN}(x)=\phi(xW_1)W_2
$$

其中 $\phi$ 可以是 ReLU、GELU 等激活函数。

```text
输入 x
  ↓ W1
中间表示
  ↓ 激活函数
非线性表示
  ↓ W2
输出
```

例如模型维度为 $D$，中间维度为 $F$：

$$
W_1\in\mathbb R^{D\times F},\qquad W_2\in\mathbb R^{F\times D}
$$

### 3.2 GLU：增加一个门控分支

GLU 的核心是把变换分成“内容”和“门”两个分支：

$$
\operatorname{GLU}(x)=(xW_u)\odot\sigma(xW_g)
$$

其中 sigmoid 门的取值在 $(0,1)$，可对每个中间特征进行输入相关的缩放。

### 3.3 SwiGLU：把门激活换成 SiLU

SwiGLU 将 sigmoid 门替换为 SiLU：

$$
\operatorname{SiLU}(z)=z\sigma(z)=\frac{z}{1+e^{-z}}
$$

于是：

$$
a=\operatorname{SiLU}(xW_g),\qquad b=xW_u
$$

$$
\operatorname{SwiGLU}(x)=(a\odot b)W_d
$$

```text
                  ┌── Wg ── SiLU ──┐
输入 x ───────────┤                × ── Wd ── 输出
                  └── Wu ──────────┘
```

严格来说，Swish 可带参数 $\beta$：$z\sigma(\beta z)$。常见 SwiGLU 使用 $\beta=1$，即 SiLU。

## 4. “门控”到底控制了什么？

中间维度的第 $j$ 个分量是：

$$
u_j=\operatorname{SiLU}(xw_{g,j})\,(xw_{u,j})
$$

内容分支学习一个输入特征，门分支再根据同一个输入调整该特征的贡献。

这比“先线性变换，再逐元素激活”多了一种乘性交互：两个不同的输入投影共同决定最终中间特征。

需要注意：**SiLU 门不是概率，也不是严格的 0/1 开关。**

- 正输入很大时，$\operatorname{SiLU}(z)\approx z$，可以大于 1；
- 负输入时可以得到负值；
- 负输入绝对值很大时，输出趋近于 0；
- 因此它可以抑制、放大，也可以改变内容分支分量的符号。

SwiGLU 通常仍计算全部中间维度，因此门控并不自动等于稀疏计算或 FLOPs 节省。

## 5. 完整张量计算过程

以下是说明性配置，不代表任何论文的实际生产配置：

```text
batch B = 2
序列长度 L = 100
模型维度 D = 512
SwiGLU 中间维度 F = 1408
```

输入：

$$
X\in\mathbb R^{2\times100\times512}
$$

### 第一步：门控投影

$$
G=XW_g,\qquad W_g\in\mathbb R^{512\times1408}
$$

得到：

```text
G：[2, 100, 1408]
```

### 第二步：内容投影

$$
U=XW_u,\qquad W_u\in\mathbb R^{512\times1408}
$$

得到：

```text
U：[2, 100, 1408]
```

### 第三步：激活并逐元素相乘

$$
Z=\operatorname{SiLU}(G)\odot U
$$

```text
Z：[2, 100, 1408]
```

不是矩阵乘法，也不是两个向量做点积；对应位置分别相乘，形状不变。

### 第四步：投影回模型维度

$$
Y=ZW_d,\qquad W_d\in\mathbb R^{1408\times512}
$$

```text
Y：[2, 100, 512]
```

输出恢复模型维度，便于与残差流相加。

## 6. 一个数值例子

为突出门控作用，假设两个投影分支已经算出：

$$
g=[2,-1,0],\qquad u=[3,4,5]
$$

则：

$$
\operatorname{SiLU}(g)\approx[1.7616,-0.2689,0]
$$

逐元素相乘：

$$
z\approx[5.2848,-1.0756,0]
$$

可以看到：

- 第一维内容 3 被放大到约 5.28；
- 第二维内容 4 被变成约 −1.08；
- 第三维被压到 0。

若降维矩阵取：

$$
W_d=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}
$$

则最终输出为：

$$
y=zW_d\approx[5.2848,-1.0756]
$$

真实网络的 $W_d$ 会学习如何混合这些中间特征，而不只是原样取出。

## 7. 参数量与计算量：为什么常见中间维度不是 4D？

省略偏置，普通 FFN 的参数量为：

$$
P_{\text{FFN}}=2DF_{\text{plain}}
$$

SwiGLU 有三块矩阵：

$$
P_{\text{SwiGLU}}=3DF_{\text{swiglu}}
$$

如果保持中间维度完全相同，SwiGLU 的投影参数和主要矩阵乘计算量都是普通 FFN 的约 1.5 倍。

若希望匹配普通 $F_{\text{plain}}=4D$ 的 FFN 参数预算：

$$
3DF_{\text{swiglu}}=2D(4D)
$$

因此：

$$
F_{\text{swiglu}}=\frac83D
$$

实际实现通常会再按硬件友好的倍数取整。因此前面的 $D=512,F=1408$ 是接近该比例、便于对齐的示例，并非精确相等。

若将一次乘加记作 2 FLOPs，单个 token 的主要投影成本约为：

$$
\operatorname{FLOPs}_{\text{SwiGLU}}\approx6DF
$$

另有 SiLU 和逐元素乘法成本。上述比较只针对 FFN 子层，不包含 Attention、归一化或路由。

## 8. 训练与实现

### 8.1 两个分支都会学习

设 $g=xW_g$、$u=xW_u$、$z=\operatorname{SiLU}(g)\odot u$，则：

$$
\frac{\partial z}{\partial u}=\operatorname{SiLU}(g)
$$

$$
\frac{\partial z}{\partial g}=u\odot\operatorname{SiLU}'(g)
$$

其中：

$$
\operatorname{SiLU}'(g)=\sigma(g)+g\sigma(g)(1-\sigma(g))
$$

所以内容分支和门分支都通过下游任务损失训练，不需要额外的门标签。

### 8.2 gate/up 投影可以融合

两个分支输入相同，可以拼接权重：

$$
[G,U]=X[W_g,W_u]
$$

把两次矩阵乘变成一次较大的矩阵乘，再切分输出。这样不改变数学结果，也不减少理论矩阵乘 FLOPs，但可能减少调度开销、改善硬件利用率。

概念伪代码：

```python
def swiglu(x, W_gate_up, W_down):
    gate_up = x @ W_gate_up
    gate, up = split_last_dim_in_half(gate_up)
    hidden = silu(gate) * up
    return hidden @ W_down
```

## 9. 与 MoE、Gated Attention 的关系

### SwiGLU 与 MoE

- SwiGLU 定义“一个前馈网络内部怎么算”；
- MoE 定义“当前 token 选择哪些前馈网络来算”。

一个 MoE 专家可以直接采用 SwiGLU：

$$
E_i(x)=\left[\operatorname{SiLU}(xW_{g,i})\odot(xW_{u,i})\right]W_{d,i}
$$

二者是可组合的不同层次设计，详见 [MoE 混合专家模型详解](./MoE混合专家模型详解.md)。

### SwiGLU 与 Gated Attention

二者都使用门控思想，但作用位置不同：

- SwiGLU：在 FFN 内部调制中间特征；
- Gated Attention：在注意力路径上调制注意力输出。

不能将两者当成同一个模块。

## 10. OneTrans-V2 中的对应设计

[OneTrans-V2 原文](../../OneTransV2.2609.28589v1.pdf) §4.3 将 FFN 设计为稀疏 MoE，各专家使用中间层较窄的 SwiGLU；§5.2 描述 gate/up 投影融合等算子优化。

本文中的具体维度、数值和通用梯度推导用于解释 SwiGLU 原理，不代表论文披露了这些生产参数。

## 总结

**SwiGLU 用一个输入相关的 SiLU 分支，逐维调制另一个内容分支，再映射回模型空间。它增强了 FFN 的非线性和乘性交互，但不会自动跳过计算；比较效率时应匹配中间维度或参数预算。**
