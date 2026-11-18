# 文献与复现边界

核查日期：2026-11-18。下列数字为论文报告值，未由本项目训练复现，不组成统一排行榜。

| 工作 | 来源 | 用途与边界 |
|---|---|---|
| HIFE-YOLO: HierarchicalInformation Flow Enhancement for Lightweight Fire and Smoke Detection | [Fire 2026, 9(11), 478](https://doi.org/10.3390/fire9110478) | 首选直接对标；YOLO11n；平均 mAP50 71.95 → 73.25%；精确划分与源代码未获取 |
| ESFD-YOLOv8n: Early Smoke and Fire Detection Method Based on an Improved YOLOv8n Model | [Fire 2024, 7(9), 303](https://doi.org/10.3390/fire7090303) | 强任务对比；mAP50 79.4%；70:20:10 划分、300 epochs；实现可得性待核验 |
| LP-YOLO: Enhanced Smoke and Fire Detection via Self-Attention and Feature Pyramid Integration | [CMC 2026](https://doi.org/10.32604/cmc.2025.072058) | 含 WTConv、BiFPN、FreqFusion；Table 3 的 mAP50 81.8% 与 Table 4 的 77.9% 冲突，不用作可信榜首 |
| Integration of YOLOv11 and Histogram Equalization for Fire and Smoke-Based Detection of Forest and Land Fires | [CMC 2025, 84(3), 5361–5379](https://doi.org/10.32604/cmc.2025.067381) | YOLO11x + HE，77.1%；640、100 epochs、70:20:10；输入增强相关工作 |
| A Comparative Analysis of YOLOv9, YOLOv10, YOLOv11 for Smoke and Fire Detection | [Fire 2025, 8(1), 26](https://doi.org/10.3390/fire8010026) | 基线选择参考；摘要 85.9% 是另一 Smoke and Fire 数据集，不是 D-Fire 成绩 |

## HIFE-YOLO 已核查事实

- 正文描述 MSGhostC3k2、RetentionGSConv、CTD 的插入位置和作用。
- Table 2 的训练／验证／测试图数为 14,122 / 3,099 / 4,302。
- Table 3 记录 Ultralytics 8.4.41、PyTorch 2.7.1+cu118、Python 3.13.0、RTX 3070 8 GiB；本仓库未验证此组合的安装或兼容性。
- Table 4 明确不使用预训练骨干。不得将旧项目的 COCO warm-start 当作其原文协议。
- Table 5 的原始 seed：YOLO11n mAP50 / mAP50–95 71.70 / 40.01%，HIFE 73.37 / 41.24%。同表 YOLOv8n mAP50 73.39%，所以 HIFE 也并非同表所有指标最高。
- Table 6 的三个 seed 为 20260716、20260921、20260922；论文报告均值增益 mAP50 +1.29、mAP50–95 +1.22 个百分点。
- Data Availability Statement 称模型配置、训练／评测脚本与冻结清单可按请求提供，未在声明中给出公开官方代码链接。本次定向搜索未找到可确认官方仓库。

## 实施前仍需核查

精确数据文件清单、坏图身份、模块代码细节、可安装的软件组合、各方法代码来源及许可。新的相关论文和公开实现需要在最终实验矩阵冻结前更新检索。

引用原始模块论文时，应追溯其正式来源。论文的已发表状态和作者声称的优势，不替代本项目的复现与同口径比较。
