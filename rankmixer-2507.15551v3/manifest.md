# RankMixer 论文处理清单

## 原文与来源

- 标题：RankMixer: Scaling Up Ranking Models in Industrial Recommenders
- 作者：Jie Zhu, Zhifang Fan, Xiaoxie Zhu, Yuchen Jiang, Hangyu Wang, Xintian Han, Haoran Ding, Xinmin Wang, Wenlin Zhao, Zhen Gong, Huizhi Yang, Zheng Chai, Zhe Chen, Yuchao Zheng, Qiwei Chen, Feng Zhang, Xun Zhou, Peng Xu, Xiao Yang, Di Wu, Zuotao Liu
- 机构：ByteDance
- 稳定 ID：arXiv:2507.15551
- 版本：v3；PDF 页边日期 2025-07-26；PDF 元数据创建时间 2025-07-29T00:14:57Z。两种日期用途不同，不推断额外出版信息。
- 规范来源 URL：https://arxiv.org/abs/2507.15551v3
- 对应 PDF URL：https://arxiv.org/pdf/2507.15551v3
- 实际获取方式：用户提供本地文件 `../../RankMixer.2507.15551v3.pdf`，复制为 `source/paper.pdf`，未重新下载。
- 最终下载 URL：不适用（本地导入）；以上链接为标识链接，未联网验证。
- 导入时间 UTC：2026-11-09T13:42:19Z
- SHA-256：`4ce87d626d0a9945f64dd5a91106eed7602a22ff69ad35e98372cd8b3ee4e37d`
- 文件大小：1263215 bytes
- 页数：9
- 完整性：PyMuPDF 成功打开并渲染全部 9 页；无附录；参考文献共 42 项。

## 状态

- download_status: downloaded
- translation_status: translation_complete
- reading_status: evidence_verified
- target_document_status: in_review
- publication_status: not_prepared
- human_review_required: true

`downloaded` 在这里表示用户原始 PDF 已成功导入本地归档。`evidence_verified` 表示已将解读中的事实与本地论文核对，不代表复现实验、验证作者所有因果主张或解决原文内在口径问题。

## 文件

- 原文：`source/paper.pdf`
- 全文文本抽取：`source/paper.extracted.txt`（保留双栏抽取原貌；公式以 PDF 为准）
- 原始英文书目：`source/references.extracted.txt`（含第 8 页正文与脚注的抽取上下文）
- 全页渲染：`source/assets/page-01.png` 至 `page-09.png`
- 图 1–4：`source/assets/fig-01-p03.png`、`fig-02-p05.png`、`fig-03-p06.png`、`fig-04-p06.png`
- 完整中文译稿：`translation/paper.zh-CN.md`
- 阅读与证据记录：`reading-record.md`
- 详细解读：`article/draft.md`
- 审查记录：`article/review.md`
- 目录入口：`README.md`

## 完整性核对

- 覆盖作者、摘要、CCS、关键词、§1–5、所有小节、脚注 1。
- 原文没有附录或独立补充材料入口。
- 公式（1）–（12）均保留，另保留未编号时延式。
- 图 1–4 完整裁剪、视觉核对、配中文图注；图中英文保留。
- 表 1–6 均转为 Markdown；表 4 为可读性转换成长表，数值不变。
- 42 项参考文献保留编号、中文题名和书目身份；完整英文书目另存，不将缩写书目误作英文原始全文。
- 图像均使用本地相对路径；解读使用独立复制资产。

## 已识别口径问题

1. 参数增长：引言“超过 100 倍”与具体线上表 6 的约 70 倍不同。
2. FLOPs 增幅：表 6 标 20.7 倍，但 2106/107≈19.68。
3. 表 1 AUC 增益百分数计算式未明确；译稿保留原值，不反推绝对 AUC。
4. 单样本前向、训练 FLOPs/Batch 与线上 FLOPs 的范围未统一。
5. MoE 节中 DTSI-MoE / DS-MoE 名称混用；式 10 router 输出省略专家下标。
6. 图 4 可确认 token 级动态激活，但不据此独立验证全部专家训练充分。

以上疑点在译注、阅读记录与解读末尾明确保留；没有用推测修补为论文事实。

## 许可与发布

本次交付是用户要求的本地翻译与详细解读，不进行知乎操作。PDF 可见页未明确图片复用许可；公开转载前需人工确认版权、授权及引用要求。未人工批准、未发布；不记录虚构的公开文章 URL。
