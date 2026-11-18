# MNIST–GAT 稳定化配方（2026-11-17）

源码基于 `70ae58df0780e74c9a51209000680a4726451b59`，另行保存本地补丁及 SHA256。
这是用户授权的训练配方变更，不能标成该原始提交原样运行的结果。

原提交的零向量归一化修复保留：norm floor 0.1、FP32 辅助损失，
rank=2、对比/结构权重 0.1/0.1、温度 0.5、同类正样本权重 0.2。
模型、量化 forward/STE、数据顺序、batch size、损失及学习率调度器不改。

本轮仅处理 QAT–MNIST–GAT，None / GraphLoRA 各种子 41、42、43，共六次新训练：

- 两种方法的主学习率均从 0.005 改为 0.001。
- 所有可训练参数统一裁剪，总范数上限 1.0。
- 在裁剪之前检查原始梯度，非有限值或总范数超过 1e6 仍拒绝更新。
- 从新初始化与新 Adam 状态开始；每项注册完整 200 轮，不复用旧失败断点。
- 前 30 轮使用完整官方训练/验证划分做数值筛查，不测试。所有六项完成筛查后，
  检查梯度、最终模型和 Adam 有限性、配对初始化及划分一致性，再释放后续训练。
  这只是数值门禁，不保证准确率或完整 200 轮都稳定；验证损失和裁剪比例原样记录。
- 后续从本次实验的第 30 轮断点继续至 200 轮。全部六项训练完成后，
  最低验证任务损失的检查点各测试一次，平局取最早轮。

新增日志：原始梯度峰值、裁剪后峰值、裁剪批次数/比例；异常时记录梯度最大的参数。
所有故障都保留；不会自动放宽阈值、换超参数或反复重试失败模型。
新结果不会覆盖原 focal 的 24 个完成模型和失败记录。

运行（在本目录所属的代码根目录、已激活兼容环境）：

```bash
python -B -m unittest runs.lorap_study.test_graphlora_stability \
  runs.lorap_study.test_graphlora runs.lorap_study.table1_graphlora.test_portable \
  runs.lorap_study.table1_graphlora.test_stabilized -v
python -u -B -m runs.lorap_study.table1_graphlora.stabilized_run \
  --root /absolute/path/to/new_result \
  --launch /absolute/path/to/launch_records \
  --data-root /absolute/path/to/datasets --device cuda:0 --workers 2
```

自动启动器要求全新结果目录。启动记录目录需事先保存 `provenance.json`。
执行顺序为 CUDA/scatter 检查、2 轮真实数据 smoke、30 轮完整数据数值筛查、
200 轮正式配对训练、一次性测试、独立结果核验。手动恢复可用快照中的 portable worker；
筛查阶段需所有任务完成后调用 `release-stability` 才能继续。
原始 `run_portable.sh` 保持原协议，不能用于启动本稳定化配方。

`completion_audit.json` 与 `post_completion_audit.json` 均通过才算完成。
`REPORT.md` 和 `paired_summary.json` 报告均值 ± 样本标准差、配对差值、验证 CE 和裁剪统计。
