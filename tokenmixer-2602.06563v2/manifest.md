# TokenMixer-Large 论文资产清单

## 论文身份与来源

- title: TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders
- authors: Yuchen Jiang; Jie Zhu; Xintian Han; Hui Lu; Kunmin Bai; Mingyu Yang; Shikang Wu; Ruihao Zhang; Wenlin Zhao; Shipeng Bai; Sijin Zhou; Huizhi Yang; Tianyi Liu; Wenda Liu; Ziyan Gong; Haoran Ding; Zheng Chai; Deping Xie; Zhe Chen; Yuchao Zheng; Peng Xu
- affiliation: ByteDance AML / ByteDance
- stable_id: arXiv:2602.06563
- version: v2
- version_date: 2026-02-10（原 PDF 首页版本标记）
- source_url: https://arxiv.org/abs/2602.06563v2
- canonical_pdf_url: https://arxiv.org/pdf/2602.06563v2
- acquisition_method: user_provided_local_pdf
- local_original: ../../TokenMixer.2602.06563v2.pdf
- final_download_url: not_applicable（使用本地 PDF，未联网下载）
- original_download_time: unknown
- local_acquired_at_utc: 2026-11-10T03:37:57Z（本次整理会话时间，非用户原始下载时间）
- source_file: source/paper.pdf
- sha256: 62b61c910ab7b34b970ecf00353ea0431a5f15ea7f8bcbc66d335fc0cba6a887
- pages: 12
- pdf_validation: 12 页可解析、可渲染；与用户原文件校验一致
- supplementary_material: 未提供独立补充材料；主 PDF 包含附录 A.1—A.7

## 生命周期

- download_status: downloaded
- translation_status: translation_complete
- reading_status: evidence_verified
- article_status: in_review
- publication_status: not_prepared
- human_review_required: true

## 文件

| 文件 | 内容 |
|---|---|
| source/paper.pdf | 用户提供原 PDF 的逐字节副本 |
| source/paper.extracted.txt | 全文文本抽取，保留页面分隔；原始抽取可能有双栏顺序错乱 |
| source/references.extracted.txt | 完整英文原始书目，保留作者与出版信息 |
| source/assets/page-01.png … page-12.png | 全部页面渲染 |
| source/assets/fig-01-p03.png … fig-08-p12.png | 全部 8 张原图的高分辨率裁剪；具体页码见阅读记录 |
| translation/paper.zh-CN.md | 摘要、正文、25 个公式、13 张表、8 张图、参考题名及全部附录的中文译稿 |
| reading-record.md | 证据索引、技术问题、实验与资产核验、未确定项 |
| article/draft.md | 详细技术解读，事实说明与末尾工程判断分区 |
| article/review.md | 自动质检与人工审核提醒 |
| README.md | 阅读入口 |

## 完整性核验

- 正文覆盖摘要、1 引言、2 相关工作、3.1—3.5 方法、4.1—4.6 实验、5 结论。
- 附录覆盖 A.1—A.7。
- 公式 1—25 均存在；图 1—8 均有本地原图和中文图注。
- 表 1—7、9—14 均转录为 Markdown。原文无表 8，不做补造。
- 参考文献编号 1—41 保留，题名中文化，完整英文书目另存。
- 章节与分页衔接经原 PDF 对照，表 2/3/5/6/7/9/10/11/12/13/14 数值经页面视觉检查。
- 图 7 内嵌字体细小，仅核验整体负载趋势；不从不可清晰辨认的微小标签推导额外数字。
- 图像相对路径均检查存在，文章入选图与 source 图一致。

## 原稿歧义与处理

1. 式 (12)/(16) Post-Norm 与最终 Pre-Norm 描述不一致：译文保留公式并注释，解读采用图 1 说明的最终结构，标作解释性示意。
2. Xavier uniform/normal、标准差/方差 scale 口径不一致：译文与解读均标明，未猜测实际代码。
3. 表 4 末行的基线脚注含糊：不累加收益，不据此还原完整 scaling 数值关系。
4. 式 (4) token 下标数量、式 (5)/(6)/(7) 记法、Listing 1 的 N/H 含糊：保留并标注。
5. 主损失、辅助 loss、router 梯度与线上统计信息未充分报告：明确列为未核实/未报告，不作为既定结论。
6. DOI、会议和 ISBN 是模板占位，未推断录用信息。

这些歧义不阻碍核对本文实际采用的结构、图表数字和作者报告结果，但阻碍严格代码级复现；evidence_verified 不代表全部实现细节得到验证。

## 许可与发布

PDF 首页只明确个人/课堂复制等条件。原图和译稿保留在本地供学习及人工审核，未进行公开发布或知乎操作。公开转载全文或图片前需人工确认相应许可。人工审核状态未设置为 approved。
