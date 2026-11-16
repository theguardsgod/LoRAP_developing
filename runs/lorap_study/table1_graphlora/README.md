# TABLE1：GraphLoRA-adapted，3 个种子

**另一台服务器直接运行：见 [PORTABLE.md](PORTABLE.md)。** 独立入口会重新训练 None/GraphLoRA 配对，不依赖旧结果路径。

用户要求 TABLE1 新增 GraphLoRA 并将重复种子减为 3。本入口默认固定种子 **41、42、43**；覆盖 Cora、CiteSeer、MNIST、CIFAR10、ZINC × GIN/GCN/GAT × QAT/DQ/A2Q，即 **45 个 cell、135 次新增训练**，相对于 10 种子减少 70%。不更改或挑选已经完成的 Table V 十种子结果。

默认延续此前 GraphLoRA 实验的 **joint** 联合训练：从与 TABLE1 已归档无提示骨干相同的随机初始化训练，复用对应 seed 的已完成无提示对照与测试指标，不重训对照。现有 `table1_frozen` 的无提示阶段本身就是骨干联合训练；其适配阶段为冻结训练，不能混称同一协议。可显式指定 `--protocol frozen`，从同一已训练骨干只训练新增分支；报告会保留训练方式。

## 2026-11-15 数值修复：objective_version=2

量化后的零向量经原 `F.normalize` 的 `eps=1e-12` 会产生巨量的有限梯度，污染 Adam 二阶矩。新对比目标和 ZINC 结构目标统一使用 `x / max(norm(x), normalization_floor)`，默认范数下限为 **0.1**，对应归一化 Jacobian 范数上限 10；可用 `prepare --normalization-floor` 显式指定，并写入配置、报告和源码快照。低于该范数的表示不再强行映射到单位球面，因此须注明这是数值修复后的 adapted 目标，不能声称该区域与官方原目标完全一致。下限仍需在正式方法比较中公开并按验证集评估。

辅助目标至少用 FP32，关闭的损失不计算。每次 Adam 更新前检查总梯度范数，超过配置中的 `gradient_norm_limit=1e6` 就报错并保留故障记录，不通过梯度裁剪掩盖问题；逐轮保存最大梯度范数。此阈值是拒绝更新的数值保护，不是达到阈值后继续训练的裁剪规则。

修复后必须选择**新的结果目录，从初始参数重新训练**。新 worker 拒绝使用未注册 v2 目标的旧 study；旧配置与优化器状态不能混入修复后的实验。要复现旧实验，请进入其旧 `source_snapshot`。worker 同时检查归档文件与实际执行源码的哈希。

可以用 `--data-root Cora=/path/to/root --data-root MNIST=/path/to/root` 重定位数据；参考实验的原配置保持归档，重定位只作用于新作业。本次修复不改变分类任务的验证 CE 选模规则，不自动替换旧论文数值；报告始终列出实际同种子 None 对照。

CPU 回归检查：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest runs.lorap_study.test_graphlora_stability runs.lorap_study.test_graphlora -v
```

## 模型与损失

名称始终是 **GraphLoRA-adapted**，不是原论文跨源图/目标图迁移复现。方法来源沿用 `../graphlora_official/` 固定官方代码和 `../GRAPHLORA_METHOD.md`。无跨域源图，不引入 SMMD。为避免重新大规模搜索，固定 rank=2、对比/结构系数各 0.1、温度 0.5、同类正样本权重 0.2；不根据本轮测试结果修改配置。

- GIN：原消息聚合与原自节点表示进入两层低秩 MLP，中间 BN/ReLU，最后 up 全零；A2Q 自节点表示复用原消息量化器输出。QAT/DQ 沿用其原生未量化自节点项和实际量化消息聚合，DQ 训练的混合精度掩码保持不变。
- GCN：实际归一化/量化消息聚合进入低秩映射，最后 up 全零。该分支共享原图传播算子。
- GAT：原传播接收的量化投影特征进入独立低秩注意力分支，up 全零、注意力 Xavier 初始化；使用原传播边集。相较官方对原输入做低秩投影，这是复用量化消息特征的明确改动。
- 分支输出添加到各原图层输出，保留其外部残差、激活与读出。不同骨干、数据集的参数量分别统计，不把 rank 相同称为等参数量。所有适配参数为 FP32，BN 缓冲区另列，不合并进量化权重。
- 节点分类：对比目标只读取训练掩码内的标签；结构损失读取公开图结构和预测概率，不读取验证/测试标签。
- 图分类：对比目标使用池化图表示与训练图标签；结构目标复用 FP32 主分类头，无额外节点分类器。
- ZINC 回归：不构造类别正样本，使用配对图表示对比；结构约束改用归一化节点隐表示重建图内邻接，避免单输出 softmax 恒等于 1。此项为回归移植，不能声称官方分类目标原样复现。

所有原量化器每次前向调用次数不增加。新增分支不使用额外校准数据。构造时只探测训练特征的维度，并恢复原观察器、BN 和 RNG 状态。

## 训练和记录

joint 使用对应已归档 TABLE1 cell 的模型、数据划分、学习率、优化器分组和调度器，包括之前锁定的节点 GIN 基线覆盖参数。每组 200 轮，节点 GAT 250 轮；完整官方划分，batch size 128。适配器使用骨干主学习率、零权重衰减。frozen 沿用原冻结实验的节点 200/图 100 轮上限与原验证早停设置，允许初始零适配器为最佳模型。

按最低验证任务损失选模，平局保留最早轮；所有注册的训练完成后才统一解锁最终测试。复用原 TABLE1 的测试预测分块缓存，每个已提交分块不重复推理。3 种子的均值和样本标准差仅代表本轮重复，不冒充原论文 10/100 个种子。

结果保存独立配置、源码快照及哈希、对照记录及检查点哈希、初始化指纹、划分信息、逐轮记录、best/last 检查点及完整 RNG/优化器/调度器恢复状态、实际量化器位宽、参数/缓冲区开销、耗时/显存、中文报告和 TABLE1 LaTeX 行。旧结果目录仅只读访问。正式运行与真实数据 smoke 均强制 CUDA。

```sh
# 准备全表，仅写配置和对照归档。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table1_graphlora.run prepare --root result/lorap_study/table1_graphlora_3seeds --seeds 41 42 43
# 一个或多个 GPU worker 从同一队列领取任务；任务锁防止重复训练。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table1_graphlora.run worker --root result/lorap_study/table1_graphlora_3seeds --device cuda:0
# 真实数据短流程：45 个 cell × 一个种子 × 两轮，图数据前 8 张，不访问测试集。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table1_graphlora.run prepare --root result/lorap_study/table1_graphlora_smoke --smoke --seeds 41
```

实际启动 worker 时使用结果目录 `source_snapshot` 作为工作目录及 PYTHONPATH，确保执行已冻结的代码。`STOP` 文件在轮次边界停止，删除后再次启动 worker 恢复。失败保存 `failure.json`，须检查原因后归档该标记再恢复；不会静默重复失败任务。
