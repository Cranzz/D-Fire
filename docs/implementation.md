# 实现边界与首轮实验

这是依据论文正文公式 1–13 与 Table 1 编写的独立重建，非作者代码。没有使用旧沙箱代码、权重或实验数字。

## 实现规格

- 使用 Ultralytics 8.4.41 原始 YOLO11n、两类别 Detect。保持图连接与 loss，不改动上游文件或 parse_model。
- MSGhostC3k2 替换第 8、22 层，隐藏宽度 128、一个瓶颈。两个 MSGhostConv 的主卷积为 3×3；3/5/7 深度卷积分支，经样本级 softmax 加权。
- RetentionGSConv 替换第 **17、20** 层（不是 P3/P4 融合层 16、19），两组 channel shuffle；池化核 3、stride 2、padding 1，rho=0.25。
- CTD 位于原始 box/classification towers 前，宽度 4、eta=0.2，双输出投影权重和偏置均为零。
- CTD 压缩和深度卷积不含偏置，selector/输出含偏置；两路各用 GroupNorm(1,4)。平均池化使用包含 padding 的 PyTorch 默认分母。这些归一化、偏置和边界细节为独立实现选择，正文未完全披露；参数量吻合不能证明作者实现完全一致。
- 所有共享参数来自相同 seed 的原版模型构造；新增模块独立初始化。不加载预训练权重。仅添加 CTD 的模型在初始 eval 状态下应与原版逐值一致；完整 HIFE 因替换结构而不具有此性质。

## 数据与协议

作者精确 train/val/test 清单及四张拒绝图像的名字尚未取得，不随意删除图像凑数。用户提供 YOLO 数据 YAML 并核对 names 的真实类别顺序。本项目不默认类别编号、不自动下载或另行划分测试集。

`python -m dfire.manifest --data /path/data.yaml --output /path/manifest.json` 展开实际 loader 的图像列表、记录图像和标签 SHA256，发现跨集合相同图像内容时终止。清单记录的是加载候选；需另查训练器损坏文件警告。缺失标签可能是正常负样本，记录 null 而非伪造标签。

可执行配置 `configs/train.yaml` 采用论文明确数值；水平翻转 0.5、垂直翻转 0、patience=100 是明示选择。其余未披露项沿用锁定 Ultralytics 默认值（包括 nbs=64 的梯度累积、warmup momentum/bias lr 等）；每次训练的 args.yaml 保存实际参数。不可称为作者完全相同协议。

云端使用平台已有的兼容 PyTorch/CUDA，安装会记录版本；不强行安装论文中未核实兼容性的 Python3.13/torch2.7.1+cu118 组合。Python 支持 3.12–3.13。GPU、驱动和软件不同会影响严格可重复性。正式对比固定同一平台、环境、提交、数据清单。

## 执行顺序

1. 安装后跑 pytest 与模型 audit，只验证工程正确性。
2. 准备 data.yaml，冻结清单；两个模型均 seed=20260716、从零训练 100 epoch。
3. 保存 environment.json、args.yaml、清单、results.csv、best.pt、last.pt。每轮更新 experiments/registry.csv；工程测试不得登记为科学精度。
4. 同协议验证集确认基线与 HIFE 重建有效，再开始单假设改进。暂不堆叠新模块。
5. 选定候选后补齐 20260921、20260922；测试集仅用于冻结后的最终报告。

训练会按 epoch 保存检查点，恢复需要完整 last.pt；只有权重不能恢复优化器和调度器。CLI 的 --resume 复用检查点协议和模型类型，不提供改变结构续训功能。首版限单 GPU；不宣称支持 DDP、ONNX/TensorRT 导出。

中断恢复测试确认 epoch、优化器与模型检查点可继续使用；未验证恢复后的随机增强轨迹与不中断训练逐位一致，不能据此声称逐位确定性恢复。
