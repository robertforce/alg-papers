# SiLU：Sigmoid 线性单元的计算原理与激活函数对比

## 1. SiLU 是什么？

SiLU（Sigmoid Linear Unit，Sigmoid 加权线性单元）定义为：

$$
\operatorname{SiLU}(x)=x\cdot\sigma(x)=\frac{x}{1+e^{-x}}
$$

其中 $\sigma(x)=\dfrac{1}{1+e^{-x}}$ 是标准 sigmoid 函数。

**命名与出处（论文事实）：**

- "SiLU" 一词来自 Elfwing、Uchibe、Doya 的论文 *Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning*（2017 年 arXiv，2018 年刊于 Neural Networks）。
- 同一函数族的带参形式 $f(x)=x\sigma(\beta x)$ 由 Ramachandran、Zoph、Le 在 *Searching for Activation Functions*（Google Brain，2017）中通过自动搜索发现，命名为 **Swish**；论文报告 $\beta=1$（即 SiLU）在多数任务上已足够好。
- 因此日常语境下 SiLU ≈ Swish（β=1）。"Swish" 是 Google 的商标命名，开源框架（如 PyTorch 的 `nn.SiLU`）多采用 SiLU 这个名字。

与[SwiGLU 门控前馈网络详解](./SwiGLU门控前馈网络详解.md)的关系：SiLU 是 SwiGLU 门控分支的激活函数。理解了 SiLU 本身，SwiGLU 只是在此之上加了一个内容分支。

## 2. 计算原理

### 2.1 逐元素计算

SiLU 是**逐元素（elementwise）**函数：对张量中每个标量独立计算，不改变形状，没有可学习参数（$\beta=1$ 时）。一个元素的完整计算只有三步：

```text
1. e = exp(-x)          # 算一次指数
2. s = 1 / (1 + e)      # 算 sigmoid，s ∈ (0, 1)
3. y = x * s            # 用 s 缩放输入本身
```

可以把它读成"**自门控**"：sigmoid 以 $x$ 自己为门控信号，生成一个 $(0,1)$ 区间的软开关 $s$，再用它缩放 $x$。

- $x$ 很大（如 $x=5$）：$\sigma(x)\to 1$，输出 $\approx x$，退化为恒等映射，信息无衰减通过；
- $x$ 很小（如 $x=-5$）：$\sigma(x)\to 0$，输出趋近 $0$，但不是硬零——是"乘性压制"而非"截断"；
- $x=0$：输出恰为 $0$。

### 2.2 数值例子

手算 $\operatorname{SiLU}(-1.5)$：

$$
\sigma(-1.5)=\frac{1}{1+e^{1.5}}=\frac{1}{1+4.4817}=0.1824,\qquad
\operatorname{SiLU}(-1.5)=-1.5\times0.1824\approx-0.2736
$$

手算 $\operatorname{SiLU}(2)$：

$$
\sigma(2)=\frac{1}{1+e^{-2}}=\frac{1}{1.1353}=0.8808,\qquad
\operatorname{SiLU}(2)=2\times0.8808\approx1.7616
$$

若干典型取值（说明性示例）：

| $x$ | $\sigma(x)$ | $\operatorname{SiLU}(x)$ |
|---:|---:|---:|
| −3 | 0.0474 | −0.1423 |
| −2 | 0.1192 | −0.2384 |
| −1 | 0.2689 | −0.2689 |
| −0.5 | 0.3775 | −0.1888 |
| 0 | 0.5000 | 0 |
| 0.5 | 0.6225 | 0.3112 |
| 1 | 0.7311 | 0.7311 |
| 2 | 0.8808 | 1.7616 |
| 3 | 0.9526 | 2.8577 |

### 2.3 形状特性（通用原理）

- **光滑**：处处无穷阶可导（$C^\infty$），没有 ReLU 那种 $x=0$ 处的导数间断点；
- **非单调**：负区间存在一个"小凹坑"——在 $x^*\approx-1.2785$ 处取全局最小值 $\operatorname{SiLU}(x^*)\approx-0.2785$。凹坑方程可解析写出：令导数为零得 $\sigma(x)=1+1/x$，数值求解即得 $x^*$；
- **上无界、下有界**：输出不封顶（大正数原样通过），但下界 $-0.2785$，不会像 ELU 那样负向饱和到 $-\alpha$；
- **非零中心**：输出均值为正（正输入近似原值、负输入只贡献小的负数），这点与 ReLU 相同，不是优点。

非单调的"凹坑"是 SiLU 与单调函数家族（ReLU、ELU、tanh）的本质区别：绝对值不大的负输入不会被一刀切成零，而是以小幅度、甚至改变符号后参与后续计算。

## 3. 导数与反向传播

### 3.1 解析导数

由乘积法则：

$$
\frac{d}{dx}\big[x\sigma(x)\big]=\sigma(x)+x\sigma'(x)
=\sigma(x)+x\sigma(x)\big(1-\sigma(x)\big)
$$

整理成便于实现的形式：

$$
\operatorname{SiLU}'(x)=\sigma(x)\big(1+x\big(1-\sigma(x)\big)\big)
$$

关键点：

- $\operatorname{SiLU}'(0)=\sigma(0)=0.5$，即原点处梯度是 $0.5$，而不是 ReLU 在原点的"0 或 1 取决于约定"；
- $x>0$ 时导数大于 1（输出被放大），如 $\operatorname{SiLU}'(2)\approx1.0908$；
- $-1.2785<x<0$ 区间导数为正但小于 1；$x<-1.2785$ 区间导数为负但幅值很小（如 $\operatorname{SiLU}'(-2)\approx-0.0908$）；
- $x\to-\infty$ 时导数按 $e^x$ 速率趋于 0：深负输入梯度极小但严格非零，"死亡神经元"问题只以渐近形式存在，没有 ReLU 那种梯度严格为 0 的硬死亡。

### 3.2 反向传播实现

设前向保存了输入 $x$ 和输出 $y=\operatorname{SiLU}(x)$，反向可用恒等式 $\sigma(x)=y/x$（$x\neq0$）免掉一次 exp：

$$
\operatorname{SiLU}'(x)=\frac{y}{x}+\Big(1-\frac{y}{x}\Big)\,y
$$

一个实现注意点：**只存 $y$、不存 $x$ 是不可行的**。因为 SiLU 非单调，$( -0.2785,\ 0)$ 区间的每个 $y$ 值对应两个不同的 $x$，无法从 $y$ 唯一恢复 $x$ 再求梯度。做激活重计算（activation checkpointing）时，要么保存 $x$，要么重算整段前向。

## 4. 参数 β：SiLU 是 Swish 家族的特例

Swish 的完整形式是：

$$
f_\beta(x)=x\cdot\sigma(\beta x)
$$

$\beta$ 是缩放斜率（Ramachandran et al. 论文中可固定也可学习）：

- $\beta\to0$：$\sigma(\beta x)\to0.5$，退化为线性函数 $x/2$；
- $\beta\to\infty$：$\sigma(\beta x)\to$ 阶跃函数 $\mathbb 1[x>0]$，退化为 **ReLU**；
- $\beta=1$ 即 SiLU。

所以可以把 Swish 家族理解为"线性函数与 ReLU 之间的连续插值"，SiLU 是论文实证中性价比最高的折中点。后续大规模实践（Shazeer, 2020）基本固定 $\beta=1$ 不再学习，这也是 SiLU 与 Swish 两个名字趋同的原因。

## 5. 为什么它有效：论文事实与假说

需要区分两类内容：

**论文实证事实：**

- Ramachandran et al.（2017）：用自动搜索发现 Swish，在 ImageNet 分类、COCO 检测、机器翻译、语音识别上稳定小幅优于 ReLU，且对 $\beta$ 不敏感；
- Shazeer（2020, *GLU Variants Improve Transformer*）：在 Transformer FFN 上系统对比多种激活与门控组合，SiLU 门控（SwiGLU）表现最好；该文同时推测这些差异部分属于噪声、最优选择与任务相关；
- LLaMA 及其后主流开源大模型（含 Qwen、DeepSeek 系列）的 FFN 均采用 SiLU 门控（SwiGLU），使其成为当前大模型的事实标准。

**常见解释（属于合理假说，非定理）：**

1. **光滑性利于优化**：处处可导，二阶信息连续，梯度行为更平滑；损失面对参数扰动更不敏感；
2. **非单调凹坑保留负信息**：小幅负输入以小权重通过而非清零，网络保留更多判别信息，也缓解"大面积神经元死亡"；
3. **自门控的乘性交互**：sigmoid 分支随输入自适应缩放，效果上类似于给每个神经元装了一个软门（这也是它在 SwiGLU 中当门激活的天然契合点）；
4. **大正数区间恒等**：不像 tanh/sigmoid 有饱和区，深网络中梯度不易衰减。

这些假说在文献中均缺乏严格的因果证明，选择 SiLU 更多是工程实证驱动。

## 6. 与其他激活函数的对比

### 6.1 特性对照表

| 激活函数 | 定义 | 负区间行为 | 光滑 | 单调 | 每元素大致开销 | 典型使用 |
|---|---|---|---|---|---|---|
| ReLU | $\max(0,x)$ | 硬截断为 0 | 否（0 处不可导） | 是 | 1 次比较 | 经典 CNN 默认 |
| LeakyReLU | $\max(\alpha x,x)$ | $\alpha x$，小斜率保留 | 否 | 是 | 比较 | GAN 判别器等 |
| PReLU | 同上，$\alpha$ 可学习 | $\alpha x$ | 否 | 是 | 比较+乘 | ResNet 变体 |
| ELU | $x$（$x>0$）；$\alpha(e^x-1)$（否则） | 负向饱和到 $-\alpha$ | 是 | 是 | exp | 自编码器 |
| SELU | 缩放 ELU（$\lambda\approx1.0507,\alpha\approx1.6733$） | 负饱和 | 是 | 是 | exp | 自归一化网络（需特定初始化/归一化条件） |
| tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | 双向饱和到 $[-1,1]$ | 是 | 是 | exp | RNN、门控内部 |
| Softplus | $\ln(1+e^x)$ | 平滑趋 0 | 是 | 是 | exp | ReLU 的光滑替代 |
| **SiLU/Swish** | $x\sigma(x)$ | 平滑小凹坑（min ≈ −0.2785） | 是 | **否** | exp+除法+乘 | 大模型 FFN（SwiGLU） |
| GELU | $x\Phi(x)$ | 平滑小凹坑（min ≈ −0.170） | 是 | **否** | erf 或 tanh 近似 | BERT/GPT 系 FFN |
| Mish | $x\tanh(\operatorname{softplus}(x))$ | 平滑凹坑（min ≈ −0.309） | 是 | 否 | exp+tanh | YOLOv4/v5 |
| HardSwish | $x\cdot\frac{\operatorname{ReLU6}(x+3)}{6}$ | 分段折线近似凹坑 | 否 | 否 | 整数运算 | MobileNetV3（端侧） |

### 6.2 数值对照（说明性示例）

同一组输入下三个常用非单调激活的输出：

| $x$ | ReLU | GELU | SiLU |
|---:|---:|---:|---:|
| −3 | 0 | −0.0040 | −0.1423 |
| −2 | 0 | −0.0455 | −0.2384 |
| −1 | 0 | −0.1587 | −0.2689 |
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0.8413 | 0.7311 |
| 2 | 2 | 1.9545 | 1.7616 |
| 3 | 3 | 2.9960 | 2.8577 |

三个关键观察：

- **正区间**：GELU/SiLU 都略低于 ReLU（因为乘的因子 $<1$），大 $x$ 时三者趋同；
- **负区间**：GELU 的凹坑更浅更靠原点（最低 ≈ −0.170，位置 ≈ −0.75），SiLU 的凹坑更深更靠左（最低 ≈ −0.2785，位置 ≈ −1.28）。所以 GELU 更接近"带软边缘的 ReLU"，SiLU 对负输入更宽容；
- **原点斜率**：SiLU'(0)=0.5；GELU'(0)=0.5（$\Phi(0)=0.5$）。两者在原点附近几乎重合，这解释了实践中 GELU 与 SiLU 常可互换且结果接近。

### 6.3 与 ReLU：光滑换一点计算

ReLU 的优势是极简（一次比较）、稀疏激活（严格零）且长期被验证；缺点是硬零截断导致神经元死亡、原点不可导。SiLU 用一次 exp+乘法的代价换取：处处光滑、负输入不完全清零、无硬死亡。实证上 SiLU 通常带来小幅稳定收益（千分之一量级的精度提升常见），但不是量变。

### 6.4 与 GELU：哲学不同，形状相近

- GELU（Hendrycks & Gimpel, 2016）的动机是概率性的：$x\Phi(x)$ 相当于把输入乘上"它在标准正态噪声扰动下仍为正的概率"，依赖"输入近似零均值单位方差"的假设，因此总与 LayerNorm 搭配（BERT/GPT 系）；
- SiLU 没有分布假设，纯粹是"自身 sigmoid 缩放自身"的自门控；
- 两者在原点附近几乎重合、整体形状高度相似，实证差异通常很小。工程上 Transformer 世界的现状是：传统 BERT/GPT 用 GELU，LLaMA 及之后的新模型几乎都用 SiLU（SwiGLU）。

### 6.5 与 Mish：同一家族的更平滑变体

Mish（Misra, 2019）$=x\tanh(\ln(1+e^x))$，比 SiLU 更光滑、凹坑略深，计算也更贵（exp+tanh 两次超越函数）。实证收益不一致，未成为主流。

### 6.6 与 HardSwish：算力受限时的替身

MobileNetV3（Howard et al., 2019）用 HardSwish 以分段线性近似 SiLU，避免浮点 exp，适合端侧整数推理。它在 $x=\pm3$ 处不可导、形状略粗糙，但实验表明精度损失可忽略——这也侧面说明这类激活函数的具体形状并非十分敏感。

### 6.7 计算成本视角

- **单元素算子成本**：ReLU 一次比较 ≪ SiLU（exp + 除法 + 乘法）；
- **但现代 GPU 上激活函数不是瓶颈**：Transformer FFN 的开销集中在矩阵乘（张量核心），SiLU 属于逐元素访存受限操作，exp 由 SFU 硬件单元承担，实际占比很小。工程上的正确姿势是**融合**：把 SiLU 与逐元素乘（SwiGLU）、或与相邻的归一化/残差合并成一个 CUDA/Triton kernel，只读一次显存；
- **显存与反向**：反向传播需要 $\sigma(x)$，要么前向保存 $x$ 反向重算 exp，要么用 §3.2 的 $y/x$ 恒等式省一次 exp（仍需保存 $x$，见该节说明）；
- **端侧/量化场景**：浮点 exp 不友好，HardSwish 或查表 sigmoid 是常见替代。

## 7. 工程实现

PyTorch：

```python
import torch
import torch.nn.functional as F

y = F.silu(x)          # 标准实现
y = x * torch.sigmoid(x)   # 等价手写
```

数值稳定 sigmoid：朴素写法 $1/(1+e^{-x})$ 在 $x$ 很负时 $e^{-x}$ 上溢。稳定实现分支处理：

```python
def stable_sigmoid(x):
    pos = x >= 0
    z = torch.exp(torch.where(pos, -x, x))       # 指数恒为负数
    return torch.where(pos, 1 / (1 + z), z / (1 + z))
```

说明性伪代码——SwiGLU 融合算子中的 SiLU（现代推理引擎的典型做法）：

```python
# gate_up = x @ W_gate_up 一次矩阵乘，切出两支
gate, up = split_last_dim_in_half(gate_up)
hidden = silu(gate) * up        # SiLU 与逐元素乘融合在同一 kernel
out = hidden @ W_down
```

## 8. 适用条件与注意事项

- **无超参**（β=1 固定），无分布假设，可即插即用替换 ReLU/GELU；
- **输出非零中心**：与前代激活相同的缺陷，依赖归一化层（LayerNorm/BatchNorm）消化；
- **替换 ReLU 不保证涨点**：收益通常是小幅且任务相关的，大规模实验（Shazeer, 2020）也承认部分差异在噪声范围内；
- **纯门控语义时注意**：SiLU 门不是概率也不是 0/1 开关——正输入大时输出大于 1（放大），负输入时输出可为负（翻转符号），详见 [SwiGLU 门控前馈网络详解](./SwiGLU门控前馈网络详解.md) §4；
- **反向传播与重计算**：保存 $x$ 而非 $y$（§3.2 的不可逆性）。

## 参考

- Elfwing, Uchibe, Doya. *Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning*. arXiv:1702.03118; Neural Networks, 2018.
- Ramachandran, Zoph, Le. *Searching for Activation Functions*. arXiv:1710.05941, 2017.
- Hendrycks, Gimpel. *Gaussian Error Linear Units (GELU)*. arXiv:1606.08415, 2016.
- Shazeer. *GLU Variants Improve Transformer*. arXiv:2002.05202, 2020.
- Misra. *Mish: A Self Regularized Non-Monotonic Neural Activation Function*. arXiv:1908.08681, 2019.
- Howard et al. *Searching for MobileNetV3*. arXiv:1905.02244, 2019.
- Clevert, Unterthiner, Hochreiter. *Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)*. arXiv:1511.07289, 2015.
- Klambauer et al. *Self-Normalizing Neural Networks*. NeurIPS 2017.

## 总结

**SiLU(x) = x·σ(x) 是"用输入自身的 sigmoid 给输入加软门"的逐元素激活：正区间近似恒等，负区间以平滑小凹坑保留而非截断信息，处处光滑、无参数。它的导数 σ(x)(1+x(1−σ(x))) 在原点为 0.5、深负区仅渐近为零。与 ReLU 相比用一次 exp 换光滑与非单调性，与 GELU 形状几乎重合而哲学不同；在大模型时代它的主流用法是作为 SwiGLU 的门控激活。**
