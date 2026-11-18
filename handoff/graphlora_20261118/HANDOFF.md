# 给接手 AI：只续跑尚未完成的三个 GraphLoRA 模型

用户已要求暂停原服务器训练并迁移。请按本文继续执行，不要重新注册 focal 实验，也不要重训 None 或已完成的模型。本交接说明针对 2026-11-18 的实际断点；旧 PORTABLE.md 描述的是从头训练流程，不适用于此次断点迁移。

## 任务和当前状态

仅 **A2Q–CIFAR10–GAT / GraphLoRA-adapted v2 / gat_stabilized_v1** 尚未完成：

| 种子 | 已保存轮次 | 目标总轮次 | 尚需轮次 |
|---|---:|---:|---:|
| 41 | 119 | 200 | 81 |
| 42 | 99 | 200 | 101 |
| 43 | 30 | 200 | 170 |

合计续跑 352 个 epoch，完成后对三个最佳检查点各测试一次。三项均已通过 30 轮稳定性检查。包内保留模型、Adam、调度器、CPU/CUDA RNG、完整训练历史、best/last 检查点、注册配置和源码快照。可以从现有稳定配方的断点继续；这不是恢复修复前受污染的 Adam。

原服务器的其他已完成模型及故障记录全部保留，不需要在新服务器重训：本轮 expanded 实验中另有 7 个配置 × 3 个种子已训练完成，修复后的 QAT–ZINC–GAT 另有 3 个种子完成 200 轮。它们中尚未测试的模型留在原服务器后续汇总，本包不包含这些模型。已测试的 QAT–Cora–GIN 和 QAT–MNIST–GCN 等历史修复结果也不重跑。迁移包完成状态只代表上述三个任务，不能声称全部历史实验已测试完。

## 代码与固定实验协议

仓库：https://github.com/theguardsgod/LoRAP_developing

迁移分支：`codex/graphlora-handoff-20261118`。基础修复提交：`70ae58df0780e74c9a51209000680a4726451b59`。请保存实际 `git rev-parse HEAD`；此分支在基础修复上包含已运行的稳定性修复及迁移包装器。

- GraphLoRA rank=2，归一化范数下限 0.1，辅助损失至少 FP32。
- 采用原服务器已注册配方：主学习率 0.0001，全部学习率分组为原配置的 0.1 倍；原始梯度范数大于 1e6/非有限则报错，通过检查后全局裁剪到 1.0。
- 完整官方 CIFAR10 超像素图划分：45,000 / 5,000 / 10,000，batch size 128，200 轮。不要改成图像 CIFAR10 网络或抽样数据。
- 按最低验证任务损失选模，平局取最早轮；三个训练全部完成后才统一测试。恢复时校验并复用有效测试缓存。
- 不改超参数、不依据测试结果挑种子，不重新训练 None。统计报告用 **样本标准差（ddof=1），不是 standard error**。
- `study.json` 和 job 配置中旧数据绝对路径是原配置哈希的一部分，**不要手动替换**。包装器只重定向数据加载路径，不修改配置、模型或优化器。

## 获取代码与兼容环境

建议 sparse clone，避免下载仓库中不相关的历史实验档案：

```bash
git clone --filter=blob:none --sparse --single-branch \
  --branch codex/graphlora-handoff-20261118 \
  https://github.com/theguardsgod/LoRAP_developing.git LoRAP-handoff
cd LoRAP-handoff
git sparse-checkout set runs quantize_function quantize_method prompt utils handoff
git rev-parse HEAD
nvidia-smi

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --only-binary=:all: torch_scatter torch_sparse \
  -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
python -m pip install -r requirements-study.txt
```

可复用兼容环境。原环境为 Python 3.12、PyTorch 2.7.0+cu118、torch_scatter 2.1.2+pt27cu118、PyG 2.6.1、NumPy 1.26.4。新 GPU 若要求其他 CUDA wheel，保持 PyTorch 2.7.0，PyG 扩展同步匹配 CUDA，并记录差异。包装器启动时强制检查 CUDA 与 torch_scatter GPU 前向/反向，不会退回 CPU 正式训练。不同硬件的浮点与稀疏归约不保证逐位相同。

```bash
python -B -m unittest \
  runs.lorap_study.table1_graphlora.test_handoff \
  runs.lorap_study.table1_graphlora.test_portable \
  runs.lorap_study.table1_graphlora.test_stabilized
# GPU 环境下另行执行真实优化器分组回归：
python -B -m unittest runs.lorap_study.table1_graphlora.test_gat_repair
```

## 校验并解包（只做一次，不训练）

数据无需从本仓库下载。可复用已有 PyG GNNBenchmarkDataset 的 CIFAR10，路径结构为：

```text
/path/to/datasets/cifar10/CIFAR10/raw/...
/path/to/datasets/cifar10/CIFAR10/processed/...
```

没有数据时 PyG 会下载到该位置。包装器对 train/val/test 图内容及顺序做语义哈希，不读取测试预测或指标；与原服务器哈希不一致时必须停止排查，不能跳过校验。原 pickle 字节无需完全一致。

在仓库根目录执行，`--root` 必须为一个尚不存在的新目录：

```bash
python -B -m runs.lorap_study.table1_graphlora.handoff prepare \
  --root /path/to/results/graphlora_cifar10_gat_resume \
  --data-root /path/to/datasets
```

该步骤校验 archive SHA256、逐文件 SHA256、源码、配置、稳定性检查、Adam/RNG、119/99/30 轮历史以及完整数据内容。失败时保留目录和错误记录；修复环境后使用另一个新目录重新 prepare，不能覆盖已有训练。成功后会保留 STOP，不会自动开始训练。

## 续跑（按实际可用 GPU 调整）

```bash
# 三张卡分别承担一个种子；两张或一张卡也可以，删去多余设备参数即可。
# 设备编号是 CUDA_VISIBLE_DEVICES 映射后的逻辑编号。
nohup python -u -B -m runs.lorap_study.table1_graphlora.handoff run \
  --root /path/to/results/graphlora_cifar10_gat_resume \
  --devices cuda:0 cuda:1 cuda:2 \
  > /path/to/results/graphlora_cifar10_gat_resume/launch.log 2>&1 < /dev/null &
```

`run` 显式解除 STOP，从 **last.pt** 继续；不会从 best.pt 开始训练。最多三个训练 worker 即可，不需要分布式梯度通信。如果只能通过 Slurm 使用 GPU，请使用本机实际有权限的账户和分区提交同一命令，不要照搬原服务器账户。

查看根目录 `status.json`、`worker_*.log` 和每个 seed 的 `progress.json` / `history.json`。再次暂停请执行：

```bash
touch /path/to/results/graphlora_cifar10_gat_resume/STOP
```

等 worker 在轮次边界保存并退出；恢复直接重复 `handoff run`，不要再次 prepare，不要同时启动多个控制进程。若出现 failure.json，保存现场和完整 traceback，检查根因；不能删除失败记录后盲目重跑，更不能把未完成训练指标当正式结果。若必须改代码或配方，另建实验目录记录差异，再明确安排新实验，不能伪装成同配方继续训练。

## 完成检查与交付

训练完成后包装器自动统一测试并生成报告。也可只重建报告，不重新推理：

```bash
python -B -m runs.lorap_study.table1_graphlora.handoff report \
  --root /path/to/results/graphlora_cifar10_gat_resume
```

验收：`completion_audit.json` 的 `passed=true`、`fits=3`、`tests=3`；三个 history 都是连续 1–200 轮；每个 test 的有效评估次数为 1，检查点哈希与 TEST_LOCK 一致。交付：

1. `HANDOFF_REPORT.md`、`handoff_summary.json`、`handoff_per_seed.csv`：三个种子准确率、均值 ± 样本标准差、验证损失、梯度峰值和裁剪比例。
2. `completion_audit.json`、`TEST_LOCK.json`、每个 seed 的 test.json、done.json、history.json、best.pt、last.pt 及测试缓存。
3. `study.json`、`source_snapshot/`、`source_manifest.json`、`migration.json`、`runtime_*.json`、worker/launch 日志，以及实际 commit、环境和 GPU。
4. 说明失败或数值异常；无新 None 配对结果，不声称新配对优势。最终与其他已完成配置的汇总由原服务器进行。

原服务器已完成包哈希和真实数据解包验证，并保留 STOP；交接验证不推进任何正式训练。具体本地验证见 `validation.json`。
