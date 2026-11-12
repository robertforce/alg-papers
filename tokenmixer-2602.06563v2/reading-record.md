---
title: "TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders"
version: arXiv:2602.06563v2
source_file: ./source/paper.pdf
translation_file: ./translation/paper.zh-CN.md
status: evidence_verified
---

# 阅读记录

## 问题与已有方法不足

研究对象是工业排序模型的稠密特征交互主干，不是生成 item ID 的自回归召回器。输入包括异构稀疏特征 embedding 及 DIN/SIM/LONGER 等序列模块提取结果；输出池化后进入多任务预测头。原版 TokenMixer 通过 split/concat 替代注意力，但混合改变坐标系和 token 数，直接残差相加会带来位置语义不对齐。深层训练稳定性、训练阶段未稀疏及碎片化算子也限制扩展。

## 关键贡献

1. mixing 后以逆布局 reverting 恢复原始 T×D 坐标系；残差跨越完整混合分支。
2. 保留不同 token 的参数隔离，升级逐 token SwiGLU。
3. Pre-RMSNorm、间隔残差、低层辅助监督、down 小初始化支持深层优化。
4. 每 token 独立专家池；生产主要为 3 个路由专家+1 个共享专家，激活 1 路由+1 共享，即 1:2。
5. 先寻找高效果 dense 配置，再做细粒度稀疏化；这是结构迭代策略，论文未明确给出 dense checkpoint 无损转换/继续训练协议。
6. Grouped GEMM、BF16 训练/FP8 推理、Token Parallel 协同。

## 实验与收益归因

### 数据、样本、损失、训练和评估

- 电商：>500 个特征、数亿用户、采样后 4 亿条/天、数据收集跨度两年；广告 3 亿/天；直播 170 亿/天。不能把收集跨度当所有模型统一训练两年。
- 电商 64 GPU，广告与直播 256 GPU；未报告具体 GPU 型号。
- sparse 异步更新、dense 同步更新；Adagrad，dense LR=.01，sparse LR=.05。
- 报告 AUC、UAUC；表 2 为 CTCVR。未报告 CTCVR 主损失的精确定义、多任务权重、辅助 loss 公式、负样本比例、标签窗口、train/test 切分。
- 参数量不含 sparse embedding；训练 FLOPs 对应 batch=2048。没有统一报告所有模型端到端耗时。

### 关键数字

- 表 2：500M Large +.94%，RankMixer +.84%，均相对 MLP；二者报告差值 .10 个百分点量级。原稿未定义 ΔAUC 为绝对差×100 还是相对百分比，文章保留原单位、不换算绝对 AUC。
- 表 2：4.6B dense 与 4.6B 总量/2.3B 激活 MoE 均 +1.14%；FLOPs 29.8T→15.1T，减少 49.33%。
- 表 5：去 mixing/reverting -.27%；改 shared SwiGLU -.21%；去残差 -.15%；改 pertoken FFN -.10%；去跨层残差+aux -.04%；去 global token -.02%。并非可相加独立贡献。
- 表 6：per-token→standard MoE -.10%，论文称总参数和激活参数对齐。
- 表 12：1:2 α=2 无报告损失，α=1 -.04%；1:4 α=4 最好，但仍 -.03%。
- 表 7：广告 ADSS +2%，电商订单 +1.66%/GMV +2.98%，直播 Pay +1.4%。线上基线与实验组规模显著不同，不能将全部业务提升归因单一架构变更。
- 没有重复种子、置信区间、显著性、线上实验时长或流量比例。

## 图片与表格资产

所有页面已渲染为 source/assets/page-01.png 至 page-12.png。关键方法、结果、消融及附录页已视觉核对；全部图片已从 PDF 页面高分辨率裁剪。图 7 仅使用可辨认的整体负载趋势，不提取细小字体中的额外数据。

| Asset | 来源 | 状态 | 支持结论 | 本地文件 |
|---|---|---|---|---|
| fig-01 | p3 Figure 1 | ready | E02/E03/E04 | source/assets/fig-01-p03.png |
| fig-02 | p4 Figure 2 | ready | E05 | source/assets/fig-02-p04.png |
| fig-03 | p5 Figure 3 | ready | E12 | source/assets/fig-03-p05.png |
| fig-04 | p7 Figure 4 | ready | E08 | source/assets/fig-04-p07.png |
| fig-05 | p7 Figure 5 | ready | E08 | source/assets/fig-05-p07.png |
| fig-06 | p10 Figure 6 | ready | E04/E09 | source/assets/fig-06-p10.png |
| fig-07 | p10 Figure 7 | ready（定性） | E10 | source/assets/fig-07-p10.png |
| fig-08 | p12 Figure 8 | ready | E06 | source/assets/fig-08-p12.png |
| tab-01 | p5 Table 1 | ready | E12 | translation/paper.zh-CN.md（转录）；原图 page-05.png |
| tab-02/tab-03 | p6 Tables 2/3 | ready | E02/E07 | translation/paper.zh-CN.md；page-06.png |
| tab-04 | p7 Table 4 | ready（基线脚注有歧义） | E08/E15 | translation/paper.zh-CN.md；page-07.png |
| tab-05/tab-06/tab-07 | p8 Tables 5/6/7 | ready | E03/E05/E07/E09/E14 | translation/paper.zh-CN.md；page-08.png |
| tab-09 | p10 Table 9 | ready | E11 | translation/paper.zh-CN.md；page-10.png |
| tab-10/tab-11/tab-12 | p11 Tables 10/11/12 | ready | E05/E09/E13 | translation/paper.zh-CN.md；page-11.png |
| tab-13/tab-14 | p12 Tables 13/14 | ready | E06/E09 | translation/paper.zh-CN.md；page-12.png |

## 工程判断

- [agent_inference] mixing/reverting 应理解为置换坐标系中的可学习分支加原坐标系残差；reverting 是布局逆变换，不会撤销中间 MLP 的信息交互。
- [agent_inference] token-specific 模型用额外存储容量换取单位样本计算可控，代价是固定 token 语义、参数常驻和小矩阵效率压力。
- [agent_inference] α 放大被激活时的专家梯度，但不是一般意义上的无偏梯度估计；路由概率非均匀、Adagrad 状态及路由/输出变化影响结论。
- [agent_inference] down 小初始化使分支近零，避免同时缩小 up/gate 将乘性路径压得过弱；但不等于所有深层梯度问题已解决。
- [agent_inference] Token Parallel 的价值不仅是装入更大模型，还在于小 batch 分摊权重读取与维持跨 block 分片布局；实际收益受 all2all 链路限制。

## 局限与未回答问题

1. ΔAUC 百分比口径及基准绝对 AUC 不明，不能自行换算。
2. 式 (12)/(16) 为 Post-Norm，但正文/图1/附录采用 Pre-Norm。文章明确以其最终设计说明解释，公式不充当可运行规格。
3. down 初始化在正文是 uniform/stddev，附录是 normal/variance scale，实际初始化代码 unverified。
4. 表 4 最后一行的基线脚注与样本量比较叙述不完全清晰，不相加、不用于精确重建数据 scaling 曲线。
5. 式 (4) token 下标数量有 off-by-one；Listing1 split 的 N/H 与限制未说明。
6. “RMSNorm 后 bias 数学冗余”不是普遍成立的等价定理；8.4% 吞吐是去 bias+换 norm 的组合结果。
7. α=1 且仅激活一个路由专家时，若 softmax 仅在被选专家中计算，router 的主损失梯度会涉及实现细节；原稿不足以确定真实梯度机制，文章不做确定断言。
8. 辅助损失公式、权重、是否 stop-gradient、是否仅训练时使用未报告。
9. 无开源实现/公开数据说明；本文未外部检索代码仓库，不能声称作者没有开源。
10. 未报告线上实验时长、流量、显著性、资源账单、P99 延迟、GPU 型号及 FP8 校准细节。
11. 许可仅明确个人/课堂使用条件；图片公开转载需确认。本地文章是内部人工审核稿。

原稿的这些局部歧义已被解释并限制在相应范围，未用于支撑确定结论；evidence_verified 表示这里实际采用的关键事实和图表均核对，不表示原文不存在歧义或实验可完全复现。

## 证据索引

所有证据位置均指向 **source/paper.pdf，arXiv v2**。

| ID | 类型 | 结论 | 原文位置 |
|---|---|---|---|
| E01 | paper_fact | 排序主干；语义分组+global token+mean pooling+任务预测 | p2 §3.1；p3 §3.2/图1 |
| E02 | paper_fact | mixing/reverting 恢复 T×D，残差设计比较 | p3 式9–16；p6 表3；p7 §4.3 |
| E03 | paper_fact | per-token SwiGLU 及其消融 | p4 式17/18；p8 表5 |
| E04 | paper_fact | 每token独立专家池；shared expert 不跨token | p4 §3.4；p10 图6/§A.2 |
| E05 | paper_fact | Pre-Norm、间隔残差、辅助损失 | p4 §3.3.3/3.3.4、图2；p8 表5；p11 表10 |
| E06 | paper_fact | down小初始化 +.03%；全部小初始化 -.10% | p12 表14/§A.7/图8 |
| E07 | paper_fact | 同规模离线比较及近半 FLOPs 的MoE | p6 表2；p8 表5/6 |
| E08 | paper_fact | 多业务扩展，较大模型需要更多数据 | p7 图4/5、表4、§4.4 |
| E09 | paper_fact | 1:2 α=2，1:4 α=4及相应损失 | p11 表12、§A.6；p12 表13 |
| E10 | paper_fact | 1:2较均衡；1:8更不均衡，均衡loss有益 | p10 图7、§A.2 |
| E11 | paper_fact | DCN增益150M .09%、500M .04%、700M .00%；MFU最高60% | p10 表9/§A.3；p11 §A.3续 |
| E12 | paper_fact | 训练算力受限/推理带宽受限；BF16训练FP8推理1.7× | p5 表1、图3、§3.5.1/2 |
| E13 | paper_fact | 通信4L→2L+1；4路batch320吞吐+29.2%/+96.6% | p5–6 §3.5.3、代码清单1 |
| E14 | paper_fact | 三场景线上收益与不同规模的线上基线 | p8 表7、§4.6 |
| E15 | unverified | 绝对AUC、显著性、表4末行基线、完整损失实现 | p6 §4.1；p7 表4脚注；p8 §4.6未报告 |
| E16 | author_claim | α通过补偿激活频率改善更新；逐token给router先验 | p11 §A.6；p8 §4.5.2 |
| E17 | agent_inference | 可逆布局不撤销可学习交互；容量/计算权衡 | 由p3式9–16、p4式17–21、p6表2推导 |
| E18 | unverified | 实际初始化分布、std或variance缩放、最终norm代码 | p3式12/16 vs p4§3.3.3/3.4.4 vs p12式25 |
