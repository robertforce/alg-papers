# Transformer 三大架构：Encoder-Decoder、Encoder-only、Decoder-only 详解

## 1. 总览

Transformer 原始论文（Vaswani et al., 2017, *Attention Is All You Need*）为机器翻译设计的是**完整的 encoder-decoder** 结构。后续工作沿两条路径各自做"减法"，形成今天的三大架构：

| 架构 | 出现 | 结构 = 原始 Transformer 的 | 代表模型 |
|---|---|---|---|
| Encoder-Decoder | 2017（原始形态） | 编码器 + 解码器（含交叉注意力） | 原始 Transformer、T5、BART、Whisper |
| Encoder-only | 2018（BERT） | 只留编码器 | BERT、RoBERTa、DeBERTa、各类嵌入模型 |
| Decoder-only | 2018（GPT-1） | 只留解码器，**去掉交叉注意力** | GPT 系、LLaMA、Qwen、DeepSeek |

三者共享同一地基——Transformer 块（多头自注意力 + 前馈网络 + 残差与归一化，归一化位置的取舍见 [PreNorm 与 PostNorm 归一化位置详解](./PreNorm与PostNorm归一化位置详解.md)，FFN 的现代形态见 [SwiGLU 门控前馈网络详解](./SwiGLU门控前馈网络详解.md)）——**本质区别只在注意力的可见模式（掩码）与子层组合方式**。

核心统一公式：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}+M\Big)V
$$

三种架构的区别，就是矩阵 $M$（掩码）和 $(Q,K,V)$ 来源的不同。

## 2. Encoder-Decoder：原始完整版

### 2.1 结构

原始 Transformer base 配置：编码器、解码器各 6 层，$d_{model}=512$，8 头，FFN 隐层 2048。

**编码器层**（$N$ 层堆叠），自注意力是**双向**的——每个源 token 可见全部源 token：

$$
H=\operatorname{Encoder}(X): \quad
\begin{aligned}
h' &= h+\operatorname{MHA}(h,h,h)\\
h'' &= h'+\operatorname{FFN}(h')
\end{aligned}
$$

**解码器层**（$N$ 层堆叠），每层**三个子层**：

$$
\begin{aligned}
d' &= d+\operatorname{MaskedMHA}(d,d,d) && \text{因果自注意力：只看已生成的目标前缀}\\
d'' &= d'+\operatorname{CrossMHA}(d',H,H) && \text{交叉注意力：}Q\text{ 来自解码器，}K,V\text{ 来自编码器输出}\\
d''' &= d''+\operatorname{FFN}(d'') &&
\end{aligned}
$$

交叉注意力是 encoder-decoder 独有的部件：它是"**目标对源的全连接**"，解码的每一步都能查询源序列的全部信息，且不需要因果掩码（源在解码前已完整给定）。

### 2.2 三种注意力可见模式

掩码矩阵示意（✓ = 可见）：

```text
编码器自注意力（双向，全连接）      解码器自注意力（因果，下三角）      交叉注意力（目标→源全连接）
     t1  t2  t3                          t1  t2  t3                     s1  s2  s3  s4   (源)
t1   ✓   ✓   ✓                    t1   ✓                          d1   ✓   ✓   ✓   ✓
t2   ✓   ✓   ✓                    t2   ✓   ✓                      d2   ✓   ✓   ✓   ✓
t3   ✓   ✓   ✓                    t3   ✓   ✓   ✓                  d3   ✓   ✓   ✓   ✓
```

### 2.3 训练与推理

**训练**：teacher forcing 的条件语言建模。目标序列整体右移一位作解码器输入，一次前向并行计算所有位置：

$$
\mathcal{L}=-\sum_{t=1}^{T}\log P\big(y_t \mid y_{<t},\, x\big)
$$

**推理**：两阶段。编码器对源序列**只算一次**并缓存其输出 $H$（以及交叉注意力要用的 $K,V$）；解码器从起始符开始逐 token 自回归生成，每步因果自注意力可用 KV cache，交叉注意力的 $K,V$ 反复复用同一份。

**代表（论文事实）**：原始 Transformer（翻译）；T5（Raffel et al., 2019，把一切任务统一成 text-to-text）；BART（Lewis et al., 2019，降噪自编码器预训练）；Whisper（Radford et al., 2023，语音识别——音频 log 特征进编码器，文本自回归解码）。

## 3. Encoder-only：BERT 路线

### 3.1 结构与训练

**只保留编码器栈**，双向自注意力，无解码器、无生成头。预训练目标是**掩码语言建模（MLM）**：随机遮住部分输入 token，让模型还原它们（BERT 选 15% 的 token，其中 80% 换成 `[MASK]`、10% 换成随机词、10% 保持不变），另加被 RoBERTa 弃用的下一句预测：

$$
\mathcal{L}_{\text{MLM}}=-\sum_{i\in\text{被遮位置}}\log P\big(x_i \mid x_{\setminus i}\big)
$$

关键点：**预测发生在输入序列中间**，所以注意力必须双向（因果掩码下无法看到右侧上下文）。损失只落在约 15% 的位置上，其余位置不产生监督信号。

### 3.2 使用方式

encoder-only **不生成文本**，输出的是每个位置的上下文表示。用法是接任务头：

- 句子级：取 `[CLS]` 表示（或 mean pooling）接分类头 → 情感分类、CTR/相关性预估；
- token 级：逐位置接标注头 → NER、序列标注；
- 向量检索：句向量 + 余弦相似度 → 语义召回、embedding 模型（E5、BGE、GTE 等 encoder 系）。

**代表**：BERT（Devlin et al., 2018）、RoBERTa（Liu et al., 2019，去 NSP、更多数据）、DeBERTa（He et al., 2020，解耦注意力）。TIGER、Sentence-T5 用于产生物品语义嵌入的文本编码器（如 Sentence-T5 encoder）也属此类用法。

## 4. Decoder-only：GPT 路线

### 4.1 结构与训练

**只保留解码器栈，并把交叉注意力整个去掉**。所以 decoder-only 模型只有两个子层：因果自注意力 + FFN。它与原始 Transformer 解码器不同——没有任何编码器可供查询，每层只有：

$$
\begin{aligned}
h' &= h+\operatorname{MaskedMHA}(h,h,h)\\
h'' &= h'+\operatorname{FFN}(h')
\end{aligned}
$$

预训练目标是**因果语言建模（下一 token 预测）**：

$$
\mathcal{L}=-\sum_{t=2}^{T}\log P\big(x_t \mid x_{<t}\big)
$$

因果掩码让一次前向**并行**算出全部位置的"预测下一词"损失——训练时看着像 encoder 的并行效率，推理时像解码器的自回归。

### 4.2 推理：KV cache

自回归生成第 $t$ 个 token 时，前 $t-1$ 个 token 的 $K,V$ 不变，缓存复用，每步只算新 token 的 $Q,K,V$。单步开销从 $O(t^2 d)$ 降到 $O(td)$。上下文越长 KV cache 越大，这正是 [GQA 分组查询注意力详解](./GQA分组查询注意力详解.md) 与 [FlashAttention 高效注意力计算详解](./FlashAttention高效注意力计算详解.md) 所优化的对象。

### 4.3 使用方式

一切任务（包括"理解"类）都转成**文本进、文本出**的自回归：把任务说明与输入拼成 prompt，让模型续写答案（zero-shot / few-shot）。**代表**：GPT-1/2/3（Radford et al. 2018; Brown et al., 2020）起确立，今天的 LLaMA、Qwen、DeepSeek 均为 decoder-only。

## 5. 关键机制对比：信息流与监督信号

| 维度 | Encoder-only | Decoder-only | Encoder-Decoder |
|---|---|---|---|
| 注意力可见性 | 双向（全连接） | 因果（下三角） | 源双向 + 目标因果 + 目标→源全连接 |
| 交叉注意力 | 无 | 无 | **有**（独有部件） |
| 预训练目标 | MLM（还原被遮 token） | 下一 token 预测 | 条件语言建模（teacher forcing） |
| 产生损失的 token 占比 | ~15%（被遮位置） | ~100%（每个位置） | 目标侧 ~100%，源侧 0% |
| 输出形态 | 上下文表示（需接任务头） | 自回归生成的文本 | 自回归生成的文本 |
| 生成能力 | 无（本质非生成模型） | 强 | 强（且显式条件于源） |
| 推理缓存 | 不适用 | 因果自注意力 KV cache | 因果 KV cache + 交叉注意力 K,V（源侧只算一次） |
| 训练-推理一致 | 天然一致（非自回归） | 一致（同左到右） | 存在 teacher forcing 暴露偏差问题 |
| 典型任务 | 分类/标注/嵌入 | 开放生成、通用对话 | 翻译、摘要、语音转写、条件生成 |

**监督信号密度**是最容易被忽视的差异：BERT 每个 batch 只有 15% 的位置产生损失；causal LM 每个位置都在预测下一个词，同样的数据量下监督信号密度高得多——这是 decoder-only 在海量无标注数据上 scaling 更有效的常见解释之一（属解释性假说，非定理）。

## 6. 为什么 LLM 时代 decoder-only 成为主流

**论文/工程事实：**

- GPT-3 展示了 decoder-only + 大规模数据 + in-context learning 的能力跃迁，此后主流开源大模型（LLaMA、Qwen、DeepSeek 等）全部采用 decoder-only；
- 工程上因果掩码 + KV cache 的组合在推理侧极其友好，且训练一次前向即可对全部位置并行计算损失；
- 但 encoder-decoder 在**小数据条件生成**上仍占优：T5、BART 时代的对比实验表明翻译、摘要等任务 enc-dec 参数效率更高；Whisper（2023）选 enc-dec 做语音识别。

**常见解释（假说）：**

1. 目标简单统一（下一 token），与推理时的使用方式完全一致，无 train/test 形式差异；
2. 监督密度高（每 token 都有损失），数据效率在大数据下转化为 scaling 优势；
3. 双向注意力与自回归生成存在目标冲突，encoder 的双向编码能力可以由足够大的因果模型在上下文中隐式获得（UL2、PaLM 相关分析持类似观点）；
4. 生态与累积工程优化（KV cache、推理框架、并行策略）都围绕 decoder-only 迭代。

**来自推荐系统的重要反例视角**：架构优劣不是绝对的，取决于"计算花在哪里"。OneRec-V2 的分析（见 §7）显示：参数量相同时 encoder-decoder 比经典 decoder-only 还省近一半计算，但其 FLOPs 大部分消耗在不直接产生损失的上下文 token 上——上下文越长该比例越悬殊（上下文 512 时不足 3% 的 FLOPs 贡献损失）。**把计算集中到"有监督"的 token 上**，是架构选择的深层原则，而非"decoder-only 万能"。

## 7. 在生成式推荐中的体现（本地论文）

三种架构在本仓库的生成式推荐论文中全部登场：

- **TIGER（encoder-decoder）**：[TIGER 译稿](../tiger-2305.05065v3/translation/paper.zh-CN.md) 用 T5X 框架实现标准 encoder-decoder（4 层编码器 + 4 层解码器，约 13M 参数）：编码器读用户 ID token 与历史物品的 Semantic ID 序列，解码器自回归生成下一物品的 Semantic ID，beam search 解码即"生成式检索"；
- **OneRec-V1（encoder-decoder + MoE）**：[OneRec 译稿](../onerec-2502.18965v1/translation/paper.zh-CN.md) 采用 T5 式 encoder-decoder：编码器建模用户历史行为，解码器以 MoE FFN（24 专家取 2）扩展容量、会话级生成推荐列表；在线部署 1B 激活参数仅 13%。MoE 结构详见 [MoE 混合专家模型详解](./MoE混合专家模型详解.md)；
- **OneRec-V2（Lazy Decoder-only）**：[OneRec-V2 译稿](../onerec-v2-2508.20900v4/translation/paper.zh-CN.md) 发现 V1 的计算大头消耗在编码上下文 token 上，改为"懒 decoder-only"：用 Context Processor 把用户特征直接变换成**静态 K/V 对**（跨层共享、$v=k$、去掉交叉注意力的 $K/V$ 投影并叠加 [GQA](./GQA分组查询注意力详解.md)），解码器只处理目标 token。1B 参数下计算量从 296.4 降到 18.9 GFLOPs（**-94%**），收敛损失持平（3.27 vs 3.28），并把模型扩展到 8B。它说明：保留"目标→上下文全连接"的交叉注意力（encoder-decoder 的精髓）与 decoder-only 的计算效率可以兼得——**架构边界是可以被工程重新划分的**；
- **HSTU（decoder 风格的自回归序列转导器）**：[HSTU 译稿](../hstu-2402.17152v3/translation/paper.zh-CN.md) 不用标准 Transformer 块，以点式聚合注意力构建序列转导器，用下一 token 预测自回归训练——属于 decoder 风格的生成式建模范式，而非标准的 GPT 式 decoder-only 块。

## 8. 如何选择

- **只需表示/打分/检索，不生成**：encoder-only（省算力、双向上下文、任务头轻量）；
- **开放生成、通用任务、大数据预训练**：decoder-only（目标简单统一、KV cache、生态成熟）；
- **条件生成（输入与输出异构或输出远短于输入）**：encoder-decoder 值得优先评估——源只需编码一次，输出侧自回归（翻译、摘要、ASR、TIGER/OneRec 式生成式检索）；
- **工业扩展遇到编码侧计算瓶颈**：可参考 OneRec-V2 思路，把上下文压缩成静态 K/V、让计算集中在有监督的解码 token 上。

## 参考

- Vaswani, A., et al. *Attention Is All You Need*. NeurIPS 2017. arXiv:1706.03762.
- Devlin, J., et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019. arXiv:1810.04805.
- Radford, A., et al. *Improving Language Understanding by Generative Pre-Training* (GPT-1). 2018.
- Brown, T., et al. *Language Models are Few-Shot Learners* (GPT-3). NeurIPS 2020. arXiv:2005.14165.
- Raffel, C., et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5). JMLR 2020. arXiv:1910.10683.
- Lewis, M., et al. *BART: Denoising Sequence-to-Sequence Pre-training*. ACL 2020. arXiv:1910.13461.
- Tay, Y., et al. *Unifying Language Learning Paradigms* (UL2). 2022. arXiv:2205.05131.
- Radford, A., et al. *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). ICML 2023. arXiv:2212.04356.
- 本地：TIGER (arXiv:2305.05065)、OneRec (arXiv:2502.18965)、OneRec-V2 (arXiv:2508.20900)、HSTU (arXiv:2402.17152) 译稿与阅读记录，见上方链接。

## 总结

**三种架构共享同一个 Transformer 地基，差异只在注意力掩码与子层组合：encoder-only 双向看、输出表示、配任务头，长于理解；decoder-only 因果看、自回归生成，长于开放生成与大数据 scaling；encoder-decoder 两者兼有，用交叉注意力把"编码好的源"接到"自回归的目标"上，长于条件生成。没有普适最优——监督信号密度与计算分布才是本质权衡，OneRec 从 V1 的 encoder-decoder 演进到 V2 的 lazy decoder-only 正是这一权衡的工业级实证。**
