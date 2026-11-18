# 首版工程验证（2026-11-18）

本地环境：macOS / Apple M1 Pro，Python 3.12.10，PyTorch 2.15.0，torchvision 0.29.2，Ultralytics 8.4.41，ultralytics-thop 2.0.18；CPU 执行。此环境不同于论文环境，也不代表 Colab/Kaggle GPU 已验证。

`python -m pytest -q`：**6 passed**。

- 两类别 baseline：未融合 2,590,230 参数；完整 HIFE：未融合 2,310,053、融合 2,302,621。
- 融合后 MSGhostC3k2 两处分别 246,726 / 279,494；RetentionGSConv 两处 19,296 / 75,456；CTD 三处 993 / 1,889 / 3,681。与论文 Table 1 一致。
- 仅 CTD 变体与同 seed baseline 初始 eval 预测逐值相等；共享骨干参数一致。
- CTD 残差边界、梯度、完整检测 loss 反向传播、检查点重载通过。
- 微型合成数据训练计划为 2 epoch：保存第 1 epoch 后模拟中断，从 last.pt 恢复到第 2 epoch，加载 best.pt 评测通过。
- 数据清单的内容哈希与跨集合重复图像拦截通过。

额外实际执行 CLI：模型 audit、清单生成、HIFE 单 epoch 合成数据训练、best.pt 独立评测。训练产生真实 args.yaml、environment.json、results.csv 和检查点；输出保留在仓库外。

Notebook 代码单元语法检查通过，云端 Notebook 尚未实际执行。尚未进行真实 D-Fire 训练、CUDA/AMP 验证、吞吐量基准或多 seed 研究实验。合成数据指标不报告为研究结果。

参数量吻合是结构核对证据，不能证明未披露细节完全等同作者实现，更不能证明精度复现成功。GFLOPs 目前仅观察到上游摘要约 6.1，尚未做逐模块严格核对。
