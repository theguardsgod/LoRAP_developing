# GraphLoRA 修复实验：在另一台 GPU 服务器运行

分支：`codex/experiment-archive-20261020`。

此入口从新初始化训练配对的 **None / GraphLoRA-adapted v2**，不依赖旧服务器的结果目录、模型检查点或 Slurm 账户。不要用旧 Adam 断点继续训练修复后的目标。

## 1. 获取代码并安装环境

```bash
git clone --single-branch --branch codex/experiment-archive-20261020 \
  https://github.com/theguardsgod/LoRAP_developing.git
cd LoRAP_developing
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --only-binary=:all: torch_scatter torch_sparse \
  -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
python -m pip install -r requirements-study.txt
```

这是本次检查使用的软件组合。若改用其他 CUDA 版本，PyTorch 与 PyG 扩展的 wheel 必须匹配；参见 [PyTorch 历史版本安装](https://pytorch.org/get-started/previous-versions/)和 [PyG 安装说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。例如 CUDA 12.8 组合要同时替换两条安装命令中的 `cu118` 为 `cu128`。

## 2. 一条命令运行

只运行 Cora 的 GIN/GCN/GAT：

```bash
CUDA_VISIBLE_DEVICES=0 bash runs/lorap_study/table1_graphlora/run_portable.sh \
  result/graphlora_repair_cora cora /path/to/datasets
```

同时运行 Cora 三架构和 MNIST 的 GCN/GAT，覆盖此次调查的主要异常配置：

```bash
CUDA_VISIBLE_DEVICES=0 bash runs/lorap_study/table1_graphlora/run_portable.sh \
  result/graphlora_repair_focal focal /path/to/datasets
```

`/path/to/datasets` 是可写的数据根目录；不存在的数据由 PyG 下载。现成数据可按以下目录放置：

```text
datasets/
  cora/Cora/raw/...
  cora/Cora/processed/...
  mnist/MNIST/raw/...
  mnist/MNIST/processed/...
```

使用已有环境时先激活环境；也可以用 `PYTHON=/path/to/python` 指定解释器。脚本自动执行 CPU 回归测试、CUDA/torch_scatter 检查、两轮真实数据 smoke，随后才开始正式实验。脚本不绑定本地集群；在 Slurm 上使用时，将该命令放入新服务器自己的资源申请脚本。

## 3. 实验设置

| 设置 | `cora` | `focal` |
|---|---|---|
| Cell | QAT × Cora × GIN/GCN/GAT | 左侧三项 + QAT × MNIST × GCN/GAT |
| 方法 | 新训练 None / GraphLoRA-adapted v2 | 相同 |
| 种子 | 41、42、43 | 相同 |
| 正式训练次数 | 3 × 2 × 3 = **18** | 5 × 2 × 3 = **30** |
| 轮数 | GIN/GCN 200，节点 GAT 250 | MNIST 两项均 200 |
| 选模 | 最低验证任务损失，平局取最早轮 | 相同 |

完整官方数据划分，图任务 batch size 128。骨干、量化器、优化器分组与调度器取自 `table1_frozen/registry.py` 的配方，两种方法共享同一配方和初始化种子；此独立入口不读取旧服务器的基线覆盖文件。不要将结果与旧表历史 None 混称为配对比较。

修复版 rank=2，辅助权重 0.1/0.1，温度 0.5，同类正样本权重 0.2，归一化范数下限 **0.1**，梯度总范数拒绝阈值 **1e6**。辅助目标至少用 FP32；异常梯度在写入 Adam 前报错。范数下限改变了低范数区域的目标，需以 GraphLoRA-adapted v2 报告，并公开此配置。固定配方不保证准确率提升，不包含新的大规模调参。

Smoke 使用种子 41、两轮、图数据前 8 张，不计算测试指标。正式实验的**全部配对训练**完成后才统一解锁一次测试；完成后重启不会重复测试，测试分块缓存会校验检查点哈希。

## 4. 结果、暂停与恢复

结果目录包含 `study.json`、源码快照和哈希、逐轮日志、best/last 检查点、优化器和 RNG 恢复状态，以及 `status.json`、`summary.json`、`summary.csv`。只有生成 `completion_audit.json` 且 `passed=true` 才代表完整流程完成。

```bash
# 查看进度
python -m runs.lorap_study.table1_graphlora.portable report \
  --root result/graphlora_repair_focal
cat result/graphlora_repair_focal/status.json

# 在当前轮保存后停止
touch result/graphlora_repair_focal/STOP
```

恢复时先移走 `STOP`，再重复原来的 `run_portable.sh` 命令；使用相同输出目录、profile 和数据根目录。运行器从已保存的源码快照继续，跳过已完成训练及测试。若存在 `failure.json`，先检查具体错误，保留其记录后再恢复，不能忽略失败标记。

只运行 MNIST 两项时，可手动注册：

```bash
python -m runs.lorap_study.table1_graphlora.portable prepare \
  --root result/graphlora_repair_mnist --data-root /path/to/datasets \
  --cells QAT_MNIST_GCN QAT_MNIST_GAT
cd result/graphlora_repair_mnist/source_snapshot
python -m runs.lorap_study.table1_graphlora.portable worker \
  --root .. --device cuda:0
```

手动 worker 的 `--root ..` 是相对于上述 `source_snapshot` 的路径。该 MNIST 独立注册为 2 × 2 × 3 = **12** 次正式训练；建议先另建 `--smoke --seeds 41` 目录完成短流程。

## 5. 验证状态

提交前已通过数值稳定性、官方公式对照、配对初始化、CPU RNG、断点恢复、测试门禁和 MNIST 两种骨干的合成图检查。当前服务器没有可用 CUDA 分配，未将这些检查冒充新的 GPU 正式实验结果；新服务器的脚本会先执行 CUDA 和真实数据 smoke 检查。
