# TokenMixer-Large：中文翻译与详细解读

论文：**TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders**，arXiv:2602.06563v2，2026-02-10。

## 阅读入口

1. **[完整中文译稿](translation/paper.zh-CN.md)**：按论文顺序阅读，含全部正文、附录、公式、图表及参考文献题名；原稿问题用“译注”标明。
2. **[详细技术解读](article/draft.md)**：包含形状变换示例、残差对齐、逐 token SwiGLU、Sparse-Pertoken MoE、深层训练、算子/FP8/Token Parallel、离线及线上实验，末尾讨论证据边界和复现建议。
3. [阅读记录与证据索引](reading-record.md)：查找事实、作者解释与工程推断对应的原 PDF 位置。
4. [原始 PDF](source/paper.pdf) · [资产与状态](manifest.md) · [审核报告](article/review.md)。

## 核心结论

- 这是一篇工业**排序主干**论文，不是自回归生成物品 ID 的召回模型。
- mixing/reverting 在可学习交互后恢复 token 坐标，使残差两端位置对齐。
- 每个 token 独立专家池；上线主要采用 1:2 激活比例，含常开共享专家。
- 表 2 的 4.6B 总参数模型只激活 2.3B，训练 FLOPs 从 29.8T 降为 15.1T，报告 AUC 相同。
- 线上规模和基线不同，业务收益不能全部归因于单一模块。
- 原稿存在 Norm 公式、初始化尺度等复现歧义，译稿与解读均保留说明。

本地内容已经自动核验，文章仍为 `in_review`，没有人工批准或公开发布。
