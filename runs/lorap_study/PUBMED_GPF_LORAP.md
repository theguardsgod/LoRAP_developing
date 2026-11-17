# 补充 PubMed 的 A²Q / GPF-LoRAP 结果

只补 PubMed × GIN/GCN/GAT，种子 41、42、43。每项重新训练配对 None 与 GPF-LoRAP，共 **18 次训练，其中 GPF-LoRAP 9 次**。采用主表 GPF-LoRAP 的联合训练实现，不读取旧服务器的基线检查点，不重跑 Cora、CiteSeer 或 MLP。

## 运行

依赖环境见 [PORTABLE.md](table1_graphlora/PORTABLE.md)。PubMed 的模型计算使用 COO 边和 `torch_scatter`；运行器不需要 `torch_sparse`。旧工具仅在执行 ogbn-arxiv 的稀疏诊断时才加载该可选依赖。

先运行短流程：

```bash
python -m runs.lorap_study.pubmed_gpf_lorap prepare \
  --root result/pubmed_gpf_lorap_smoke --data-root /path/to/datasets --smoke
cd result/pubmed_gpf_lorap_smoke/source_snapshot
python -m runs.lorap_study.pubmed_gpf_lorap worker --root .. --device cuda:0
```

返回仓库根目录，确认短流程完成且测试次数为 0 后，注册正式实验：

```bash
python -m runs.lorap_study.pubmed_gpf_lorap prepare \
  --root result/pubmed_gpf_lorap --data-root /path/to/datasets
cd result/pubmed_gpf_lorap/source_snapshot
python -m runs.lorap_study.pubmed_gpf_lorap worker --root .. --device cuda:0
```

`--root ..` 相对于上述源码快照目录。数据路径为 `/path/to/datasets/pubmed/PubMed/`，缺失时 PyG 会下载。使用官方 public 划分：19,717 节点、500 维特征、3 类，train/val/test 为 60/500/1000 节点。

## 固定设置

- 复用 `table1_gpf_rerun/model.py` 的 GPF 输入提示和逐层低秩聚合提示；rank=2，num_prompts=20，提示学习率与骨干相同，提示权重衰减为 0。
- 使用已有 `residual_transfer/config.py` 的 PubMed A²Q 配方；GIN 沿用 CiteSeer 配方并令 a_loss=0.1；GCN 与 GAT 使用其 PubMed 专用设置。GAT 特征 dropout=0.6、注意力 dropout=0.3。完整解析配置保存于 study.json。
- GIN/GCN 各 200 轮，GAT 250 轮；所有骨干和提示联合训练，实际学习位宽另存 quantization.json，不冒称全模型始终固定 4 bit。
- 相同种子的两种方法具有相同骨干初始化、划分和训练配方。只根据最低验证任务损失选择检查点，平局取最早轮；全部 18 次训练完成后，才统一执行各模型的一次性测试。

这是一组新联合训练结果。旧等容量 MLP 报告采用冻结骨干，因此新值不能被当作已经与旧 MLP 形成严格配对的证据。

## 输出与恢复

- `status.json`：训练/测试完成数量及失败列表。
- `summary.json` / `summary.csv`：实际配对 None 和 GPF-LoRAP 的测试均值及样本标准差。
- `pubmed_results.json`：三种架构结果及可填表字符串。
- `pubmed_gpf_lorap_rows.tex`：三项 LaTeX 结果片段；未完成测试时显示 TBD。
- `completion_audit.json`：必须 passed=true、fits=18、tests=18 才算完成。

结果目录包含源码快照、哈希、best/last 检查点、优化器和 RNG 恢复状态。创建结果根目录下的 STOP 文件可在轮次边界暂停；移走 STOP 后，用同一快照 worker 恢复。遇到 failure.json 先检查原因并保留故障记录，不能混入修改过的实验配置。

CPU 回归检查：

```bash
python -m unittest runs.lorap_study.test_pubmed_gpf_lorap \
  runs.lorap_study.table1_graphlora.test_portable -v
```

有 GPU 且已准备真实数据时，设置 `PUBMED_DATA_ROOT=/path/to/datasets` 可额外运行三个真实 PubMed 后端的梯度检查。
