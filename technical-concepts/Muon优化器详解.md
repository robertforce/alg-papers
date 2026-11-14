# Muon 优化器：动量正交化的计算原理与优化器对比

## 1. Muon 是什么？

Muon（**Mo**ment**u**m Orthogonalized by **N**ewton-Schulz，牛顿-舒尔茨正交化动量）是 Keller Jordan 于 2024 年 10 月提出的神经网络隐藏层优化器，专用于 **2D 矩阵参数**。

一句话概括：**先按标准 SGD-momentum 算出动量更新矩阵，再用 Newton-Schulz 迭代把它"正交化"，然后才应用到权重上。**

**定位（论文事实）：**

- 它刷新了 NanoGPT speedrun（在 FineWeb 上训到 3.28 val loss）与 CIFAR-10 speedrun 的训练速度纪录：NanoGPT 速度提升约 1.35x，CIFAR-10 达 94% 精度的时间从 3.3 降到 2.6 A100-秒；
- 此后 12 次新的 NanoGPT speedrun 纪录（7 位不同研究者）全部沿用 Muon；
- Moonshot AI 的技术报告《Muon is Scalable for LLM Training》（arXiv:2502.16982）证明其可扩展到十亿级参数：加入权重衰减与更新尺度校准后，计算最优训练下比 AdamW 快约 2 倍（达到相同性能只需约 52% FLOPs），并据此训出 Moonlight（3B/16B [MoE 混合专家模型](./MoE混合专家模型详解.md)，5.7T tokens）；
- Kimi K2（1T 参数 MoE）使用 MuonClip（Muon + QK-Clip）完成万亿级训练，进一步佐证其可扩展性。

注意：Muon 不是全局优化器——标量、向量参数以及输入/输出层（embedding、最终分类头）仍用 AdamW 等标准方法优化，Muon 只接管隐藏层的 2D 权重矩阵。

## 2. 计算原理

### 2.1 总体更新规则

对每个 2D 参数矩阵，Moonlight 报告中的形式为：

$$
M_t=\mu M_{t-1}+\nabla\mathcal{L}_t(W_{t-1})
$$

$$
O_t=\operatorname{Newton\text{-}Schulz}(M_t)
$$

$$
W_t=W_{t-1}-\eta_t\,O_t
$$

其中 $M_t$ 是动量缓冲（初始为零矩阵），$\mu$ 是动量系数（实践中 0.95），$\eta_t$ 是学习率。实际实现使用 **Nesterov 风格动量**：送入 Newton-Schulz 的是 $\mu M_t+\nabla\mathcal{L}_t$ 而非 $M_t$，Keller 报告这在所有测试中都略优。

Keller 原始实现还带一个按形状的尺度因子 $\sqrt{\max(1, A/B)}$（$A\times B$ 为矩阵形状），Moonlight 将其改为 $\sqrt{\max(A,B)}$（见 §3）。

### 2.2 Newton-Schulz 迭代：正交化的核心

正交化操作的数学定义是：

$$
\operatorname{Ortho}(G)=\arg\min_O\{\lVert O-G\rVert_F:\ O^\top O=I\ \text{或}\ OO^\top=I\}
$$

即找 $G$ 在 Frobenius 范数下最近的半正交矩阵。设 $G=USV^\top$ 是其 SVD，则该解恰为 $UV^\top$——**把所有奇异值替换为 1，只保留奇异向量**。

直接算 SVD 太慢，Shampoo 用的耦合牛顿迭代又必须 float32 才数值稳定。Muon 采用 Newton-Schulz 五次多项式迭代：先归一化 $X_0=M_t/\lVert M_t\rVert_F$（保证奇异值落在 $[0,1]$；此缩放是安全的，因为 $\operatorname{Ortho}(cG)=\operatorname{Ortho}(G)$），然后反复执行：

$$
X_k=aX_{k-1}+b(X_{k-1}X_{k-1}^\top)X_{k-1}+c(X_{k-1}X_{k-1}^\top)^2X_{k-1}
$$

**原理推导**：由 SVD 可知一步迭代把奇异值逐个映射为 $\varphi(s)=as+bs^3+cs^5$（$\varphi$ 是奇多项式，不改变奇异向量，只改奇异值）。$N$ 步后结果为 $U\varphi^N(S)V^\top$。所以只要系数满足 $\varphi^N(x)\to1$（$x\in[0,1]$），迭代就收敛到 $UV^\top$。

**系数选择**：Keller 用约束优化 $\max a$（$\varphi'(0)=a$ 控制小奇异值的收敛速度）subject to 迭代极限落在 $[0.7,1.3]$ 内——实证发现允许 $\varepsilon\approx0.3$ 的正交化误差不损害训练，由此得到：

$$
(a,b,c)=(3.4445,\ -4.7750,\ 2.0315)
$$

整套 Newton-Schulz 可以稳定地跑在 **bfloat16** 上，通常只迭代 5 步（$N=5$；Moonlight 验证 $N=10$ 正交化更准但性能不变）。

原始实现（Keller 的仓库）：

```python
def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)          # 奇异值压进 [0,1]
    if G.size(0) > G.size(1):      # 保证行 >= 列，减少计算量
        X = X.T
    for _ in range(steps):
        A = X @ X.T                # Gram 矩阵
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
```

### 2.3 数值例子：奇异值如何被推向 1

说明性示例：设动量矩阵的某个奇异值为 $s=0.2$，反复套用 $\varphi(x)=3.4445x-4.7750x^3+2.0315x^5$：

| 迭代步 | 奇异值 |
|---:|---:|
| 0 | 0.200 |
| 1 | 0.651 |
| 2 | 1.162 |
| 3 | 0.815 |
| 4 | 0.953 |
| 5 | 0.747 |

可见：小奇异值（如 0.2）一步就被拉到 0.65，之后在 1 附近以 $\pm0.3$ 的带内振荡——这正是"快速拉升 + 允许近似"的设计权衡。若奇异值初始为 1，则 $\varphi(1)=3.4445-4.7750+2.0315=0.7010$，同样落在带内。最终所有奇异值都被"压平"到接近 1，等价于把更新矩阵变成了 $UV^\top$。

### 2.4 为什么正交化有效

三个层面的解释，注意区分假说与理论：

**经验观察（论文事实，Keller 的博客）**：手工检查发现，SGD-momentum 和 Adam 在 Transformer 2D 参数上产生的更新矩阵**条件数极高、近乎低秩**——几乎所有神经元的更新被少数几个方向主导。正交化等效于把那些"幅度小但重要"的稀有方向放大到与主导方向同量级，避免更新沿少数方向塌缩。

**理论视角（Bernstein & Newhouse, 2024）**：把优化看作"范数约束下的最速下降"，不同优化器只是选了不同的范数（见 §5.1）。Muon 的更新方向恰是**谱范数约束**下的最速下降方向。由于权重矩阵的本质是对输入/隐空间做线性算子，算子范数（谱范数）是比 Adam 的逐坐标范数更贴合参数几何的约束。

**门控/等贡献直觉（假说）**：正交化保证每个奇异方向获得同样的更新幅度，等效于对各"学习通道"做均衡，防止单一方向吃掉全部步长。严格的因果证明在文献中仍缺失，属于解释性假说。

另一个重要性质：$\operatorname{Ortho}(cG)=\operatorname{Ortho}(G)$ 是**尺度不变**的——梯度的大小信息被完全丢弃，更新幅度只剩学习率决定。这既是简化（天然尺度鲁棒），也埋下了 §3 的更新尺度校准问题。

## 3. 更新尺度、权重衰减与超参（Moonlight 的两项扩展）

原始 Muon 从小模型扩展到大模型时增益缩水。Moonshot 找到两个关键修复：

**(1) 权重衰减**。无衰减时权重与层输出 RMS 持续增长，超出 bf16 高精度范围。引入 AdamW 式解耦衰减：

$$
W_t=W_{t-1}-\eta_t(O_t+\lambda W_{t-1})
$$

**(2) 一致的更新 RMS**。AdamW 的更新 RMS 恒定在约 $0.2\sim0.4$，而 Muon 的理论更新 RMS 依赖矩阵形状：形状 $[A,B]$ 的满秩矩阵，更新 RMS 为 $\sqrt{1/\max(A,B)}$。形状不同则步长不同——$\max(A,B)$ 大的矩阵更新过小（限制容量），过小的矩阵更新过大（训练不稳，例如把 GQA/MLA 的每个 KV 头当独立参数时）。修复：每个矩阵的更新乘以 $0.2\cdot\sqrt{\max(A,B)}$：

$$
W_t=W_{t-1}-\eta_t\big(0.2\cdot O_t\cdot\sqrt{\max(A,B)}+\lambda W_{t-1}\big)
$$

这一调整把 Muon 的更新 RMS 拉回 AdamW 的区间，**使 Muon 可以直接复用为 AdamW 调好的学习率和权重衰减**，大规模训练"开箱即用"。

**其余超参**：动量 $\mu=0.95$（调参无一致收益）；NS 步数 $N=5$；Nesterov 动量默认开启。可选细化：把融合的 QKV 投影拆开，对 Q、K、V 分别正交化（实证更好）。

## 4. 开销分析（论文事实）

- **内存**：Muon 只需 1 份动量缓冲（AdamW 需 2 份一/二阶矩），NS 迭代只需临时 Gram 矩阵，无额外持久状态；
- **FLOPs**：对 $n\times m$ 矩阵（设 $m\le n$），NS 每步 $2(2nm^2+m^3)$ 矩阵乘 FLOPs；相对一次前向+反向（$6nmB$，$B$ 为 token 数）的**总开销不超过 $Tm/B$**（$T=5$）；
- 两个具体数字：NanoGPT speedrun（$m=768$，$B=524288$）开销 0.7%；Llama 405B 规模（$m=16384$，$B=16\text{M}$）开销 0.5%——**典型 LM 训练场景下 Muon 的 FLOPs 开销低于 1%**，2 倍样本效率远大于此。

分布式实现（Moonlight 开源）：ZeRO-1 式分片优化器状态，各 rank 计算本地分片的正交化，通信高效且数学性质不变。2026 年 Tri Dao 进一步提出硬件感知的 Gram Newton-Schulz 算法，优化系数与重启策略以压低 NS 的实际墙钟开销。

## 5. 与其他优化器的对比

### 5.1 统一视角：范数约束下的最速下降

Bernstein & Newhouse (2024) 的框架：每步优化器都在"选一个范数，然后在单位范数球上找最陡下降方向"。

| 优化器 | 隐含范数约束 | 更新方向的几何含义 |
|---|---|---|
| SGD / SGD-M | $\ell_2$（Frobenius） | 均匀下降，大梯度方向步长大 |
| Adam / AdamW | 逐坐标 max-of-max（近 $\ell_\infty$/RMS） | 每个坐标的更新 RMS 一致 |
| Muon | **谱范数**（Schatten-$p$，$p$ 大时趋近） | 每个奇异方向等量更新 |

权重矩阵作为线性算子，其自然度量是算子范数（谱范数），这是 Muon 相对逐坐标方法的几何优势（Moonlight 报告引用的观点）。

### 5.2 特性对比表

| | SGD-M | AdamW | Shampoo | SOAP | Sophia | **Muon** |
|---|---|---|---|---|---|---|
| 核心思想 | 动量加速 | 自适应逐坐标缩放 | 用 $G G^\top$、$G^\top G$ 的 $-1/4$ 次幂预条件 | 在 Shampoo 特征基中跑 Adam | 对角 Hessian 估计 + 裁剪 | 动量更新经 NS 迭代正交化 |
| 捕捉的二阶信息 | 无 | 对角曲率（一阶矩/二阶矩比） | 全矩阵曲率（两个 Gram 矩阵） | 全矩阵曲率（特征基） | 对角 Hessian | 奇异向量结构（丢弃奇异值） |
| 优化器状态/参数 | 1x | 2x | 高（累积 Gram 矩阵） | 高（含特征基） | ~2x | **1x** |
| 每步额外计算 | ~0 | 低 | 高（矩阵求根） | 高 | 中（周期性 Hessian） | 低（NS 5 步，<1% 训练 FLOPs） |
| 适用参数 | 任意 | 任意 | 任意张量 | 任意张量 | 白盒 LM 矩阵 | 仅 2D 矩阵（卷积可展平）；embedding/输出层仍需 AdamW |
| 关键超参 | lr, μ | lr, β₁, β₂, wd | lr, 预条件动量, 频率 | lr, 周期等 | lr, ρ, 更新频率 | lr, μ=0.95, N=5, 尺度因子 |
| 大规模证据 | 经典 | 事实标准 | 分布式 Shampoo 训过数十亿级 | 数十亿级（报告） | 数十亿级（报告） | Moonlight 16B MoE/5.7T tokens；Kimi K2 1T |

### 5.3 与 Shampoo：理论基础相同，实现哲学相反

Shampoo（Gupta et al., 2018）的预条件更新为 $(GG^\top)^{-1/4}G(G^\top G)^{-1/4}$。Bernstein & Newhouse 指出：**若去掉 Shampoo 的预条件子累积，该式恰好退化为正交化梯度 $UV^\top$**——即"无累积的瞬时 Shampoo"。Muon 再把动量移到正交化之前。二者理论同源，工程取舍相反：

- Shampoo 累积历史 Gram 矩阵、用耦合牛顿迭代求逆四次方根（需 float32），捕捉更完整的曲率信息，但状态重、实现复杂；
- Muon 只用当前动量、NS 迭代跑 bf16，**故意丢弃奇异值（曲率幅度）、只保留奇异向量（方向结构）**，换来极低的开销和实现复杂度，实证效率反而更优。

### 5.4 与更早的正交化方法

- **Orthogonal-SGDM**（Tuddenham et al., 2022）：先对梯度做 SVD 正交化再加动量。Muon 的差异：动量在正交化**之前**（实证更好）+ NS 代替 SVD（更快）。该方法当年甚至输给调好参的 SGD-M，可见"动量与正交化的顺序"和"正交化的实现成本"都是关键；
- **Stochastic spectral descent / RMSspectral**（Carlson et al., 2015）：更早的 SVD 正交化 + 谱范数缩放方案，缺乏动量且依赖（随机）SVD。Muon 的实证优势说明动量是必要的。

### 5.5 实证对比（论文事实）

- **小规模**：NanoGPT speedrun 上 Muon 对比多个强优化器（AdamW 调优版等）样本效率全面领先；1.5B 模型训到 GPT-2 XL 的 HellaSwag 水平，Muon 用 10.8 个 8xH100-小时，AdamW 需 13.3 小时；
- **大规模**：Moonshot 的 scaling law 实验显示计算最优训练下 Muon 达到 AdamW 约 2 倍计算效率；Moonlight（16B MoE，5.7T tokens）以更少 FLOPs 推进了性能-FLOPs 帕累托前沿；
- **万亿级**：Kimi K2 用 MuonClip（Muon + QK-Clip；后者对注意力 logit 超过阈值的 Q、K 投影行做收缩，抑制 Muon 大更新幅度带来的注意力 logits 激增）完成 1T MoE 训练，兼得 Muon 的 token 效率与训练稳定。这类"样本效率提升"最终要落到墙钟时间上，可结合 [MFU 模型浮点运算利用率详解](./MFU模型浮点运算利用率详解.md) 评估：Muon 的 2 倍计算效率优势远大于其 NS 迭代的 <1% FLOPs 开销。

### 5.6 证据标准的方法论提醒

Keller 特别讨论了"欠调优基线问题"：优化器文献里大量"大幅超越 AdamW"的方法死于 AdamW 基线没调好。他主张以**竞技任务**（如 NanoGPT speedrun，基线是公开可复现的纪录）作为证据标准——纪录可被任何人用标准方法刷新，自发纠错。评价 Muon 或任何新优化器时，应检查对比基线的调优强度与可复现性。

## 6. 工程实践要点

```python
# 概念伪代码：混合优化器分工
muon_params   = [p for p in model.parameters() if p.ndim == 2 and not is_embedding_or_head(p)]
adamw_params  = [p for p in model.parameters() if p.ndim < 2 or is_embedding_or_head(p)]
# Muon（隐藏层 2D 矩阵）：动量 + NS 正交化 + 尺度校准 + 解耦权重衰减
# AdamW（embedding、lm_head、norm、bias、标量）
```

- **参数分工**：隐藏层 2D 矩阵用 Muon；embedding、最终输出层、所有 1D/0D 参数用 AdamW。输入输出层用 AdamW 有 modular norm 理论支持（embedding）也有纯经验成分（输出层）；
- **卷积**：4D 卷积核把后三个维度展平成 2D 后可用 Muon；
- **超参迁移**：采用 Moonlight 的 $0.2\cdot\sqrt{\max(A,B)}$ 尺度后，可直接复用 AdamW 的 lr 与 wd 起点调参；
- **稳定性**：大模型上注意权重 RMS 增长（开权重衰减）、注意力 logits 激增（参考 QK-Clip 思路）；
- **QKV 拆分**：对 Q、K、V 分别正交化，不要作为单一融合矩阵处理。

## 7. 适用条件与注意事项

- **只管 2D 矩阵**：替换 AdamW 时是"部分替换"，需要双优化器管线；
- **正交化丢弃梯度尺度**：天然对梯度噪声尺度鲁棒，但更新 RMS 随形状漂移，必须做 §3 的校准；
- **NS 精度是近似的**（带内 $\pm0.3$）：这是刻意设计，追求精确正交化（$N=10$+）无性能收益；
- **收益与任务相关**：实证集中在 LM 预训练与视觉分类 speedrun；RL、推荐排序等场景的证据较少，迁移前应自建对齐良好的基线对比；
- **开源生态**：KellerJordan/Muon（参考实现）、MoonshotAI/Moonlight（分布式内存最优实现）、各训练框架已陆续集成。

## 参考

- Jordan, K. *Muon: An optimizer for hidden layers in neural networks*. 博客, 2024-10. kellerjordan.github.io/posts/muon/
- Liu, J., Su, J., et al. (Moonshot AI). *Muon is Scalable for LLM Training*. arXiv:2502.16982, 2025.（含分布式实现与 Moonlight 模型）
- Kimi Team. *Kimi K2: Open Agentic Intelligence*. arXiv:2507.20534, 2025.（MuonClip / QK-Clip）
- Bernstein, J., Newhouse, J. *Old Optimizer, New Norm: An Anthology*. arXiv:2409.20325, 2024.（范数约束视角，含 Shampoo 退化为正交化梯度的分析）
- Gupta, V., Koren, T., Singer, Y. *Shampoo: Preconditioned Stochastic Tensor Optimization*. ICML 2018.
- Tuddenham, M., et al. *Orthogonalising Gradients to Speed Up Neural Network Optimization*. arXiv:2202.07052, 2022.
- Vyas, N., et al. *SOAP: Improving and Stabilizing Shampoo using Adam*. ICLR 2025.
- Liu, H., et al. *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training*. arXiv:2305.14342, 2023.
- Grishina, E., et al. *Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type polynomials*. 2025.
- Tri Dao. *A Fast, Hardware-Aware Newton-Schulz Algorithm for Muon*. 博客, 2026. tridao.me/blog/2026/gram-newton-schulz/

## 总结

**Muon = 动量 + Newton-Schulz 正交化：先用标准动量聚合梯度历史，再用五次多项式迭代把更新矩阵的奇异值压平到 1（只留奇异方向），最后按 $\sqrt{\max(A,B)}$ 校准步长。它以谱范数为隐含约束、以低于 1% 的 FLOPs 开销换取约 1.3~2 倍的样本效率，靠"只保留方向、丢弃幅度"的激进简化，把矩阵自适应方法做到了内存 1x、实现简单、可扩展到万亿参数。与 Shampoo 理论同源而工程相反，与 AdamW 几何不同而互补——embedding 与输出层仍交给 AdamW。**
