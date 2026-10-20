# LoRAP developing — code and experiment archive

截至 **2026-10-20** 的 LoRA / GLoRA / GraphLoRA-adapted baseline 实现、Table V 对照，以及 Table I 的 GraphLoRA 扩展和 GPF-LoRAP 调参记录。

**开始阅读：[中文实验总结](docs/experiment_summary_20261020.md) · [完整 Table I](result/lorap_study/table1_complete_20261019_v1/table1_complete.md) · [LaTeX](result/lorap_study/table1_complete_20261019_v1/table1_complete.tex) · [CSV](result/lorap_study/table1_complete_20261019_v1/table1_complete.csv)**

## 结果概览

- Table I：45 个配置 × 3 个种子（41、42、43），GraphLoRA 共 135 次训练/测试。
- GPF-LoRAP：10 个目标配置固定重跑 30 次，再完成四轮验证集调参。最终 10 项的均值均优于固定 GraphLoRA，但不能据此宣称统计显著或全表普遍领先。
- QAT–CIFAR10–GCN：GPF-LoRAP **47.21 ± 2.24%**，GraphLoRA **43.62 ± 0.92%**。前者最终训练 300 轮，后者 200 轮。
- DQ–ZINC–GCN：GPF-LoRAP **0.5670 ± 0.0090 MAE**，GraphLoRA **0.5676 ± 0.0097**，基本持平；仍未超过历史 GPF-plus 的 **0.534 ± 0.017**。
- Table V：LoRA 及配对无提示共 40 个模型；GraphLoRA 及配对无提示共 40 个模型，均为 MNIST 完整数据、10 个种子。
- GLoRA 已停止，没有正式测试成绩。保留代码、部分训练和停止记录。

所有 ± 为样本标准差。GraphLoRA 始终指 **GraphLoRA-adapted**：同域量化图模型的联合训练移植，无源域预训练及跨域 SMMD，不是原论文跨图迁移实验复现。四轮继续调参由此前测试比较触发，GraphLoRA 没有同等搜索预算。完整表中未更新的数值来自历史表格，详见总结中的限制及逐单元格来源。

## 目录

| 路径 | 内容 |
|---|---|
| `runs/lorap_study/` | 当前模型、训练入口、调参流程、数值及流程测试 |
| `quantize_function/`, `quantize_method/`, `prompt/`, `utils/` | 原有量化器和模型依赖 |
| `result/lorap_study/` | 主要实验原始配置、日志、测试指标、源码快照、最佳检查点 |
| `archive/early_studies/` | 更早机制/残差等研究的摘要，不用于当前 Table I 更新 |
| `archive_manifest.json` | 原文件 SHA256、归档范围和省略项 |
| `docs/publication_validation.json` | 本次发布验证结果 |
| `README_A2Q.md`, `environment.yml` | 上游工作区原始说明和历史环境；不作为当前版本锁定环境 |

原实验记录中的绝对路径保留，确保原始哈希不变。`source_snapshot` 是当时实际训练代码；当前代码包含后续扩展和归档工具。数据集、`last.pt` 续训状态、运行锁和逐 batch 预测缓存保留于原实验机器，没有上传。`best.pt` 的加载应使用对应实验的源码快照；本包不支持从任意中途 epoch 精确续训。

## 环境

已验证：Python 3.12，PyTorch 2.7.0+cu118，PyG 2.6.1，RTX 4090。推荐新建独立环境：

```sh
conda create -n lorap python=3.12
conda activate lorap
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch_scatter==2.1.2+pt27cu118 torch_sparse==0.6.18+pt27cu118 \
  -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
python -m pip install -r requirements-study.txt
```

此安装配方记录已验证版本；发布时未在全新机器重新安装。正式训练和真实数据 smoke 强制要求 CUDA，不会静默回退 CPU。

## 只读检查归档

无需 GPU、数据集或 PyTorch，不执行测试集推理：

```sh
python -B -m runs.lorap_study.verify_publication
```

校验原文件 SHA256、39 行/195 个指标单元格，以及 165 个最终测试记录与最佳检查点的一致性。原始表格生成器为 `runs/lorap_study/export_complete_table1.py`；生成器会重写导出文件中的路径，建议只在副本中运行，避免改变归档字节。

## 运行已有测试

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  runs.lorap_study.test_lora \
  runs.lorap_study.test_graphlora \
  runs.lorap_study.test_graphlora_workflow -v

TABLE1_CELLS='QAT_CIFAR10_GCN DQ_CIFAR10_GCN' TABLE1_DEVICE=cuda:0 \
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  runs.lorap_study.table1_gpf_search.test_expanded \
  runs.lorap_study.table1_gpf_search.test_refine \
  runs.lorap_study.table1_gpf_search.test_pipeline \
  runs.lorap_study.table1_frozen.test_core -v
```

第二组 21 项 GPU 回归在本次发布前通过。测试包括小型合成图的优化器/检查点恢复，部分测试依赖包内历史快照；这些不是重新运行已完成实验的最终测试集。

## 新训练

始终使用**新的输出目录**，不要在发布的完成目录上执行 worker/resume/finish。

原数据路径是 `/data/cyliu/dataset/{Cora,CiteSeer,gnn-mnist,gnn-cifar10,zinc}`，数据来自 PyG Planetoid、GNNBenchmarkDataset、ZINC subset。单数据集标准 LoRA 入口支持 `--data-root`；Table V/现有 Table I 注册配置仍保存原路径。新机器需将对应数据挂载到这些路径，或在独立开发副本中修改配置并重新注册新实验。仅修改 `registry.ROOTS` 不会自动改写已归档参考配置。

标准 LoRA 的三数据集 smoke：

```sh
python -B -m runs.lorap_study.lora_run --smoke \
  --output-dir result/lorap_study/new_lora_smoke --device cuda:0
```

MNIST GraphLoRA smoke（两种量化模式和各自对照，不访问测试集）：

```sh
python -B -m runs.lorap_study.graphlora_run --profile smoke \
  --output-dir result/lorap_study/new_graphlora_smoke --device cuda:0
```

Table I GraphLoRA 的三种子完整实验：

```sh
python -B -m runs.lorap_study.table1_graphlora.run prepare \
  --root result/lorap_study/new_table1_graphlora --seeds 41 42 43
```

准备后进入新输出目录的 `source_snapshot`，使用新输出目录的**绝对路径**运行 worker：

```sh
python -B -m runs.lorap_study.table1_graphlora.run worker \
  --root /absolute/path/to/new_table1_graphlora --device cuda:0
```

上述 `prepare` 读取本包的配对 baseline 配置和最佳权重，并生成指向当前克隆位置的新参考记录。精确的历史调参复现还涉及各轮前序目录及冻结配置；见各实验的 `RUN_COMMANDS.md`、`study.json` 和 [调参文档](runs/lorap_study/table1_gpf_search/README.md)，迁移时不能直接运行包含原机器绝对路径的历史命令。

## 方法及上游来源

- [A-2Q 原工作区来源](https://github.com/weihai-98/A-2Q)
- [标准 LoRA](https://github.com/microsoft/LoRA)
- [GraphLoRA 固定上游源码与哈希](runs/lorap_study/graphlora_official/manifest.json)；[当前移植定义](runs/lorap_study/GRAPHLORA_METHOD.md)
- [GLoRA 固定上游源码与哈希](runs/lorap_study/glora_official/manifest.json)
- [DQ 原有许可证](runs/lorap_study/table1_frozen/vendor/DQ_LICENSE)

保留第三方原有许可证和来源文件；本仓库未给所有上游代码重新授予统一许可证。
