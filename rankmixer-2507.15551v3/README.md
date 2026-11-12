# RankMixer：中文翻译与详细解读

论文：**RankMixer: Scaling Up Ranking Models in Industrial Recommenders**，ByteDance，arXiv:2507.15551v3。

## 阅读入口

1. **[详细解读](article/draft.md)**：建议先读。涵盖token化、混合的维度变化与具体例子、独立FFN参数/FLOPs推导、DTSI+ReLU MoE、离线消融、线上成本拆解，以及工程判断和复现顺序。
2. **[完整中文译稿](translation/paper.zh-CN.md)**：按原文章节翻译正文，保留12组编号公式、4幅图、6张表、脚注及42项文献索引。
3. **[原文 PDF](source/paper.pdf)**：用户提供文件的原样归档。
4. [证据与阅读记录](reading-record.md)：数字与结论的原文定位、已知疑点。
5. [审查记录](article/review.md) / [文件清单与状态](manifest.md)。

## 核心理解

RankMixer 是排序阶段的稠密特征交互骨干：少量宽token → 无参数跨token通道重排 → token独立FFN。它利用统一大矩阵与融合提高GPU利用率，并可进一步扩展为稀疏MoE。

线上具体对照约为15.8M→1.1B参数、MFU 4.47%→44.57%、14.5ms→14.3ms；抖音总体活跃天数+0.2908%、使用时长+1.0836%。这组全量上线结果不能直接解释为1/8专家激活的收益。

## 注意

- 原文某些数字口径不一致，译注与解读均保留说明。
- 文献题名译为中文，作者等书目信息适当简写；完整英文书目另存source/references.extracted.txt。
- 图中英文保留，中文图注已补齐。
- 本地学习稿，未人工批准公开发布；无任何知乎操作。
