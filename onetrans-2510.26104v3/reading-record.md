---
title: "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender"
version: arXiv:2510.26104v3
source_file: ./source/paper.pdf
translation_file: ./translation/paper.zh-CN.md
status: evidence_verified
---

# 阅读记录

> evidence_verified 指本地论文中的事实、图表与所引用结果已核对，不代表作者的全部解释已被证明。原文本身的歧义保留在下面的证据边界中。

## 问题与已有方法不足

排序阶段常将历史序列压缩后才交给特征交互模块。作者希望统一两个模块的建模、扩展和执行，同时跨候选复用用户历史计算。任务为候选级 CTR/CVR，非物品 ID 生成。证据 E01—E03。

## 关键贡献

- 统一输入为 `[S;NS]`；S 可多行为、按时间融合或按意图拼接；NS 使用按组或 Auto-Split tokenizer。
- 每层 S 侧共享 QKV/FFN，NS 侧按 token 槽位独立参数。
- 因果掩码：S 不依赖 NS，NS 读取全部 S 和更早 NS。该依赖结构使 S 可跨候选复用。
- 金字塔：当前层对尾部 query 计算注意力，但保留当前完整输入的 K/V；输出才缩短，顶层保留 NS。
- 跨候选与跨请求 KV 缓存，结合 FlashAttention-2、混合精度及训练激活重计算。

## 数据、样本、训练与评估

29.1B 曝光、27.9M 用户、10.2M 物品；曝光时刻特征快照，标签固定窗口。按时间处理，每 batch 先预测后训练，日指标宏平均。UAUC 按曝光加权。具体训练损失、任务权重、CVR 负样本处理、标签窗口长度未报告。

S 模型 6 层、d=256、4 头、91M 非稀疏嵌入参数；L 模型 8 层、d=384、330M。每卡训练 batch=2048，16 H100，推理每卡 batch=100。原文 Adagrad 的 beta 参数不符合常见 API，不能视作可直接复现配置。

## 实验与收益归因

- E08：OneTransS 为 2.64T、91M，RankMixer+Transformer 为 2.51T、109M；前者在四个效果指标上更优，是最有用的近似匹配计算比较。
- E09：L 模型扩大计算和容量进一步提升，不能把全部收益只归于架构。
- E10：共享参数消融由 91M 变为 24M，效果降低，但容量也改变。
- E11：不使用金字塔，TFLOPs 从 2.64 增至 8.08，效果涨跌不一，支持其效果—计算价值。
- E12：全注意力与因果注意力指标接近，不支持“严格双向信息交换是收益必要条件”。
- E13：表 4 不明示逐行累计还是独立优化；不能相加或连乘得到组合效果。
- E14：表 5 与 DCNv2+DIN 比较，表 6 则与 RankMixer+Transformer 比较，必须分开基线。
- E15：线上 Feeds GMV/u +5.685%（p<0.05），Mall +3.670%（p<0.05）；Feeds 为 Mall 子场景，不是两个完全独立数据源的复制实验。

## 图片与表格资产

全部 PDF 页已渲染并视觉核对。图导出为 3 倍尺度，表格在译稿中重建为 Markdown，原页图作核对依据。

| Asset | 来源 | 状态 | 支持结论 | 本地文件 |
| --- | --- | --- | --- | --- |
| fig-01 | 第 2 页，Figure 1 | ready | E01 | source/assets/fig-01.png |
| fig-02 | 第 3 页，Figure 2 | ready | E02—E06 | source/assets/fig-02.png |
| fig-03 | 第 7 页，Figure 3 | ready | E16 | source/assets/fig-03.png |
| table-01 | 第 5 页，Table 1 | ready | E07 | source/assets/page-05.png |
| table-02 | 第 6 页，Table 2 | ready | E08—E09 | source/assets/page-06.png |
| table-03 | 第 7 页，Table 3 | ready | E10—E12 | source/assets/page-07.png |
| table-04 | 第 8 页，Table 4 | ready | E13 | source/assets/page-08.png |
| table-05 | 第 8 页，Table 5 | ready | E14 | source/assets/page-08.png |
| table-06 | 第 8 页，Table 6 | ready | E15 | source/assets/page-08.png |

## 工程判断

以下为 agent_inference，不是作者报告：

- I01：该模型可理解为“可共享的因果历史前缀 + 候选相关异构后缀”，既保留原始历史的多层读取，又保留用户侧复用。
- I02：Auto-Split 后的 NS 槽位为可学习隐空间槽位，不保证每个对应单一语义组；候选特征可能混入所有 NS 槽，不能自动缓存部分 NS。
- I03：金字塔需对 query 使用原始／尾部位置偏移的因果掩码。对矩形 Q/K 直接套错误对齐的三角掩码，会屏蔽本应读取的历史。
- I04：跨请求精确复用需相同模型版本、稳定历史前缀、稳定位置与裁剪策略。新增 query 仍需读取历史缓存；O(ΔL) 不是完整注意力成本。
- I05：训练内跨候选共用计算必须保留所有候选梯度，跨优化器步骤的旧 KV 通常不能当作精确等价缓存。

## 局限与未回答问题

1. “双向信息交换”的引言表述与第 3 节的严格因果前缀依赖存在张力；解读以公式和掩码为准。
2. 表 3 第 3 行是 timestamp-agnostic w/o Sep，正文列举却写 timestamp-aware；保留并提示。
3. 离线百分比增量未定义公式，不转换成绝对 AUC 或百分点。
4. 未给确切损失、任务权重、NS 槽位排序、FFN 扩展倍数、位置编码类型、L 模型头数、逐层裁剪表、缓存协议。
5. 最终顶层长度与“序列 query 数”措辞、32 倍数取整有歧义；12/16 看作 NS 数量的合理解读，不能当作完整复现配置。
6. 线上未给流量比例、实验周期、样本数、置信区间端点，不能独立重算显著性；冷启动定义未给。
7. RankMixer 扩展侧重点、混合参数容量变化、线上容量和系统优化使归因存在混杂。
8. 表 1 总样本、日均样本与线上 1.5 年训练描述不宜拼接推算完整日志周期。

## 证据索引

| ID | 类型 | 结论 | 原文位置 |
| --- | --- | --- | --- |
| E01 | author_claim | 分离序列建模和交互限制信息交换并使执行碎片化 | 第 1—2 页，§1、图 1 |
| E02 | paper_fact | CTR/CVR 候选打分任务，统一 `[S;NS]` 输入 | 第 3 页，§3、§3.1、式 1—5 |
| E03 | paper_fact | Auto-Split 与分组 NS tokenizer；时间与意图两类序列融合 | 第 4 页，§3.2、式 6—10 |
| E04 | paper_fact | S 共享，NS 按槽位独立 QKV/FFN | 第 4—5 页，§3.3、式 11—13、图 2 |
| E05 | paper_fact | 因果掩码：S 只读更早 S，NS 读完整 S 及前序 NS | 第 4 页，§3.3.1 |
| E06 | paper_fact | 尾部 query、完整当前输入 K/V，输出逐层缩短 | 第 5 页，§3.4、式 14 |
| E07 | paper_fact | 数据规模、next-batch 协议、S/L 配置 | 第 5—6 页，表 1、§4.1 |
| E08 | paper_fact | S 比强分离基线计算量接近，四指标均更优 | 第 6 页，表 2 |
| E09 | paper_fact | L：330M、8.62T，四指标均为表中最好 | 第 6 页，表 2 |
| E10 | paper_fact | 全共享参数 24M；CTR UAUC −0.29% | 第 7 页，表 3 |
| E11 | paper_fact | 关闭金字塔：8.08T；四指标收益不一致 | 第 7 页，表 3 |
| E12 | paper_fact | 全注意力与因果注意力效果接近 | 第 7 页，表 3 |
| E13 | paper_fact | 各系统优化效率变化，优化口径边界另列 | 第 8 页，表 4 |
| E14 | paper_fact | L 与 DCNv2+DIN：13.2/13.6ms，0.8/1.8GB 推理 | 第 8 页，表 5 |
| E15 | paper_fact | 用户级随机化，线上 Feeds/Mall 结果及显著性 | 第 8 页，§4.6、表 6 |
| E16 | author_claim | 扩长收益最大，深度优于宽度，观察到对数线性趋势 | 第 7—8 页，图 3、§4.5 |
| E17 | paper_fact | S 侧跨候选一次，跨请求增量；候选相关序列池化进 NS | 第 5 页，§3.5.1 |
| E18 | author_claim | Active Days +0.7478%，冷启动商品 order/u +13.59% | 第 8 页，§4.6 末段 |
| I01—I05 | agent_inference | 前缀后缀解释、缓存和实现风险 | 基于 E02—E06、E17 推导，见上文 |
