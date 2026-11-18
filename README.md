# D-Fire：复现近期方法并改进火灾与烟雾检测

从零建立研究项目，以 YOLO11n 为基础对照、HIFE-YOLO 为首选论文对标对象，在同一协议下复现、改进并验证能否超过对标模型。

**状态（2026-11-18）：已提供独立重建模型、云端训练入口及工程测试；尚无真实数据训练结果或复现成功声明。** 本项目不继承此前沙箱的代码、权重或实验数字。仓库已公开。

## 研究路线

1. 核查论文、代码可用性与数据划分，确定复现边界。
2. 建立 Colab / Kaggle 共用训练入口，验证保存、恢复和评测。
3. 完成 YOLO11n 与论文模型各一次完整训练；确认比较可信后立即开始改进。
4. 每轮检验一个结构假设，以验证集筛选候选。
5. 对最终候选完成三个配对 seed、消融、方法对比、效率和外部测试。
6. 在独立测试上确认优势，再完成论文和代码发布。

免费平台用于初期复现和筛选；必要时租用 RTX 4090 完成最终实验矩阵。平台变化不改变数据清单、模型选择规则和实验配置。

## 对标方法与核查发现

首选：[HIFE-YOLO，Fire 2026](https://doi.org/10.3390/fire9110478)。它直接改进 YOLO11n，与本研究路线接近，但不能据此称为全领域 SOTA。

- Table 4：640×640、batch 8、100 epochs；无预训练骨干，AdamW、AMP 训练、FP32 最终评测。
- Table 2：训练 14,122、验证 3,099、有效测试 4,302 张；原测试 4,306 张中四张被加载器拒绝。具体清单尚未取得，不能自行猜测被排除的文件。
- Table 6：三个 seed 的平均 mAP50 为 YOLO11n 71.95%、HIFE-YOLO 73.25%；论文基于未四舍五入数值报告平均提升 1.29 个百分点。
- 代码与清单：原文称可向通讯作者索取，未提供公开实现链接；本次定向搜索未找到可确认的官方代码仓库。未联系作者。

如果拿不到作者实现或精确清单，将发布明确标注的**独立实现与统一复现协议**，在该协议内同时训练原版、论文重建模型和新模型，不把不同划分的文献数字当作直接比较结果。

## 项目文档

- [可行性核查](docs/feasibility.md)：可行依据、复现风险与决策关卡。
- [研究任务大纲](docs/research-plan.md)：里程碑、首批实验及改进策略。
- [云端执行方案](docs/cloud-plan.md)：Colab、Kaggle 与 4090 的分工。
- [论文协议记录](configs/hife-paper-protocol.yaml)：从原文摘录，**不是可直接执行的训练配置**。
- [文献与复现边界](docs/references.md)：论文链接、已核查事实与待核查项。
- [实验登记模板](experiments/registry.csv)：逐次记录配置、权重和指标来源。

## 安装与运行

建议 Python 3.12，并使用平台提供的兼容 PyTorch/CUDA。

```bash
pip install -e '.[test]'
pytest -q
python -m dfire.cli audit --model hife
python -m dfire.manifest --data /path/to/data.yaml --output /persistent/data-manifest.json
python -m dfire.cli train --model baseline --data /path/to/data.yaml --project /persistent/runs --name baseline-s20260716
python -m dfire.cli train --model hife --data /path/to/data.yaml --project /persistent/runs --name hife-s20260716
python -m dfire.cli eval --weights /persistent/runs/hife-s20260716/weights/best.pt --data /path/to/data.yaml --split val
```

中断恢复：在相同环境使用 `train --resume /persistent/runs/<run>/weights/last.pt`，仍需提供 `--data`、`--name`；检查点保留原模型结构和训练协议。完整训练前先通过工程测试。

- [Colab / Kaggle 共用 Notebook](notebooks/cloud.ipynb)；[在 Colab 打开](https://colab.research.google.com/github/Cranzz/D-Fire/blob/main/notebooks/cloud.ipynb)。Kaggle 通过 File → Import Notebook 导入该 Notebook 的 GitHub 地址，Add Input 添加 smoke-fire-detection-yolo，并开启 GPU / Internet；自动读取已挂载数据。
- [本地验证记录](docs/validation.md)：6 项测试及合成数据 CLI 训练验证，云端 GPU 待验证。
- [实现规格、已知差异与首轮实验](docs/implementation.md)。`baseline / msghost / retention / ctd / hife` 对应原版、单模块消融与完整模型。
- Notebook 现已包含官方 README 推荐的 Kaggle 数据副本下载和路径配置；读取包内类别顺序、保留其划分。该划分不等同于已确认的 HIFE 作者划分。自备数据 YAML 仍需核对类别顺序；脚本不猜测作者测试文件。不提交数据、权重、输出或凭据。

工程测试使用微型合成数据验证训练、检查点和评测，其指标没有科学意义。依赖与本项目采用 AGPL-3.0，见 LICENSE 与 NOTICE。
