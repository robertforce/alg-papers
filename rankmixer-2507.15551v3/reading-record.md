---
title: "RankMixer: Scaling Up Ranking Models in Industrial Recommenders"
version: arXiv:2507.15551v3
source_file: ./source/paper.pdf
translation_file: ./translation/paper.zh-CN.md
status: evidence_verified
---

# 阅读记录

## 问题与已有方法不足

推荐排序稠密交互网络既要处理异构特征，又受高 QPS、低延迟预算约束。CPU 时代多模块拼装带来大量内存访问与小算子，参数扩张未有效转成 GPU 算力利用。论文聚焦稠密特征交互骨干；序列模块作为输入上游，不是自回归生成物品 ID 的生成式召回模型。[E01,E02]

## 关键贡献

- 语义分组并按固定长度切分、投影到统一 D 维；最终采用少量较宽 token。未报告分组清单及 tokenization 消融。[E03]
- 将 [T,D] 拆为 [T,H,D/H]，交换 token/head 维，再重组为 [H,TD/H]；H=T 时支持残差。无可学习权重，不等于无数据搬运成本。[E04]
- 每个 token 对应独立两层 FFN，不共享权重。相对共享 FFN 同形状前向 FLOPs 近似不变，参数增至 T 倍。语义子空间说法需注意：混合后 token 已包含所有初始 token 的切片，并非一直只代表某个原始字段。[E05]
- ReLU 路由与 L1 门控正则提供动态专家数；DTSI 使用两个路由器、推理分支正则，缓解专家训练不足。正文未给双分支任务损失完整组合、lambda 更新规则或训练伪代码。[E06]

## 数据、训练、损失与评估

- 抖音私有数据，两周，300+ 特征，数十亿用户 ID、数亿视频 ID；作者称每日万亿条记录，不可改写成精确的去重训练样本数。[E07]
- 数百 GPU，稀疏参数异步、稠密参数同步；RMSProp lr=0.01，稀疏 Adagrad。GPU 型号、精确卡数、训练步数、时序切分和去重规则均未报告。[E07]
- 任务含 Finish/Skip，输出还有 Like 等；正文未明确多任务 task loss 的具体函数、权重和负采样机制。不能自行补成“已报告 BCE”。[E02,E07]
- AUC/UAUC 越高越好，Skip AUC 提升表示更好地区分快速划走，不表示划走率上升。UAUC 的用户加权方式未交代。[E07]

## 实验与收益归因

表 1：RankMixer-100M 107M/233G；基线 8.7M/52G；Wukong 122M/442G；HiFormer 116M/326G。RankMixer 的 Finish AUC 增益 +0.64%，1B 为 +0.95%。百分数增益计算公式未明确，不据此反推精确绝对 AUC。[E08]

表 2：去混合 -0.50%、共享 FFN -0.31%、去残差 -0.07%、去 LN -0.05%。共享 FFN 会同时缩减参数，不能把 -0.31% 全部归因于参数隔离；其余消融也无方差或重复试验信息。[E09]

表 3：All-Concat-MLP -0.18%、All-Share -0.25%，表内参数/FLOPs 变化均 0；Self-Attention -0.03%，参数 +16%、FLOPs +71.8%。这是当前配置下的效果/效率证据，非 Attention 不适合推荐的普遍证明。[E10]

图 2 支持有限参数范围的扩模趋势，无幂律指数、置信区间、数据规模对照和计算最优配比研究。正文称 D/T/L 等参数下效果近似相同，但没有单独的充分曲线或表。[E11]

图 3：1B dense AUC 约 0.8645，DTSI+ReLU 在 1/8 激活时约 0.8643，普通稀疏约 0.8615，100M dense 约 0.8614；均为读图近似值，不作为精确表格数据。正文称吞吐 +50%，未给硬件/batch/延迟设置。没有 DTSI-only 与 ReLU-only 独立消融。[E12]

图 4 是不同 token 的激活比例轨迹，不是完整专家级均衡分布；支持动态差异，不足以单独证明所有专家充分训练或激活数等同于信息量。[E13]

表 4/5：抖音总体 Active Days +0.2908%，Duration +1.0836%，抖音极速版 +0.1968%/+0.9869%；广告 AUC +0.73%，ADVV +3.90%。抖音低活跃 Active Days +1.7412%。作者称显著，8 个月长期观察；未报告分流比例、样本量、p 值、置信区间和反向实验精确定义。[E14]

表 6：15.8M→1.1B 约 69.62 倍；107G→2106G 为约 19.68 倍，原文表记 20.7 倍。MFU 4.47%→44.57% 约 9.97 倍；fp32→fp16，时延 14.5ms→14.3ms。基线与表 1 的 8.7M 不同；不能将全部上线收益仅归于架构，因容量、MFU、精度也变化。摘要/引言“100倍/两个数量级”与具体70倍口径不同，以表6为具体部署依据。[E15]

## 图片与表格资产

PDF 全 9 页渲染，关键页和裁剪图均视觉核对。4 幅图完整导出。6 张表已在译稿转录；表 4 改成长表，保留全部 40 个增益值。没有附录或单独补充材料。

| Asset | 来源 | 状态 | 支持结论 | 本地文件 |
|---|---|---|---|---|
| fig-01 | Figure 1，p3 | ready | E02–E06 | source/assets/fig-01-p03.png |
| fig-02 | Figure 2，p5 | ready | E11 | source/assets/fig-02-p05.png |
| fig-03 | Figure 3，p6 | ready | E12 | source/assets/fig-03-p06.png |
| fig-04 | Figure 4，p6 | ready | E13 | source/assets/fig-04-p06.png |
| table-01 | Table 1，p5 | ready | E08 | source/assets/page-05.png；译稿表1 |
| table-02 | Table 2，p6 | ready | E09 | source/assets/page-06.png；译稿表2 |
| table-03 | Table 3，p6 | ready | E10 | source/assets/page-06.png；译稿表3 |
| table-04 | Table 4，p7 | ready | E14 | source/assets/page-07.png；译稿表4 |
| table-05 | Table 5，p7 | ready | E14 | source/assets/page-07.png；译稿表5 |
| table-06 | Table 6，p7 | ready | E15 | source/assets/page-07.png；译稿表6 |

许可：用户本地提供论文，PDF 可见页未明确图片复用许可。本地翻译/阅读草稿使用；任何公开转载前需人工确认授权与引用要求。evidence_verified 表示本地证据已核对，不表示独立复现实验或认可所有作者主张。

## 工程判断

1. 主要价值是以规则的数据流、足够大的矩阵和位置专属 FFN，让模型容量与 GPU 执行更适配，而不是某个复杂显式交叉公式。[I01]
2. 固定 T,D 时 per-token FFN 用同样计算量调用更多不同参数；在固定结构下扩 D/L/T，公式12仍使参数和FLOPs同比增长。不能宣称任意扩模都不加算力。[I02]
3. 无参数置换本身只重新布线；交互函数由后续可学习 FFN 形成。置换可能发生真实 copy，需融合/布局优化。[I03]
4. 同等吞吐成本不是全部 TCO：1B 权重存储、加载、训练、优化器状态、跨设备通信及 tail latency 均需额外测量。[I04]
5. 建议先复现 dense 100M 和 token mixing，再扩大宽度，最后做 MoE；MoE 非 1B dense 上线低延迟的必要解释。[I05]

## 局限与未回答问题

- U01：AUC 增益 % 是相对增益还是百分点，缺计算式；不从表1推绝对值。
- U02：式12与训练FLOPs/Batch、线上表6的计量范围不统一；batch、前后向、上游序列模块是否计入不清。
- U03：上线参数增幅两种说法，表6 FLOPs 倍数算术不符；保留原文并显示复算。
- U04：ReLU 与 DTSI 具体训练流程、专家数、lambda 更新规则、极端负载与零专家回退缺失。
- U05：A/B 统计量、分流设置、样本量和分人群定义未披露。
- U06：基础损失、任务权重、划分方式、GPU型号/固定服务约束、模型配置的 k 未报告。
- U07：没有统一容量/统一FLOPs/统一墙钟预算三种维度的完整对照，不能作普遍最优结论。
- U08：没有公开代码、公开数据结果在本 PDF 中给出；未外部搜索核查，不能断言作者从未公开。

## 证据索引

所有位置均对应本地 v3 PDF 页码。

| ID | 类型 | 结论 | 原文位置 |
|---|---|---|---|
| E01 | author_claim | 异构特征与低MFU是扩模障碍 | p1–2，§1 |
| E02 | paper_fact | 输入→混合/PFFN→均值池化→多任务输出 | p2 §3.1，p3 Fig1 |
| E03 | paper_fact | 语义分组、固定切片和投影 | p2–3 §3.2，式2 |
| E04 | paper_fact | 无参数 head/token 重排，H=T | p3–4 §3.3.1，式3–5 |
| E05 | paper_fact | token独立FFN及MMoE区别 | p4 §3.3.2，式6–9 |
| E06 | paper_fact | ReLU门控、正则及双router DTSI | p4 §3.4，式10–11 |
| E07 | paper_fact | 数据、优化器、评估定义 | p5 §4.1 |
| E08 | paper_fact | 离线模型效果/参数/FLOPs | p5 Table1 |
| E09 | paper_fact | 组件消融结果 | p6 Table2，§4.4 |
| E10 | paper_fact | 路由消融与Attention额外成本 | p6 Table3，§4.4 |
| E11 | author_claim | 扩展趋势与最终D/T/L配置 | p5 Fig2，p5–6 §4.3 |
| E12 | paper_fact | 稀疏激活下效果及吞吐主张 | p6 Fig3，§4.5 |
| E13 | paper_fact | token激活比例轨迹 | p6 Fig4，p7 §4.5 |
| E14 | paper_fact | 两应用Feed和广告线上增益 | p7 Tables4–5，p7–8 §4.7及脚注1 |
| E15 | paper_fact | dense 1B部署成本与优化拆解 | p7 Table6，§4.6；p1摘要 |
| E16 | paper_fact | 参数/单样本前向FLOPs近似式 | p4 §3.5，式12 |
| I01 | agent_inference | 硬件适配是主要工程意义 | E01,E04,E05,E15 |
| I02 | agent_inference | 共享与扩模两个比较口径不可混同 | E05,E16 |
| I03 | agent_inference | 置换是路由，交互由FFN完成 | E04,E05 |
| I04 | agent_inference | 线上成本结论不能覆盖所有TCO | E15及部署细节缺失 |
| I05 | agent_inference | 推荐先dense再MoE的复现顺序 | E08,E09,E12,E15 |
