# TABLE1 GPF-LoRAP 定向重跑

从用户提供的 LaTeX 表格提取 GPF-LoRAP 均值低于 GraphLoRA 的分类单元、
以及 MAE 高于 GraphLoRA 的 ZINC 单元。预先登记 10 个组合 × 种子 41/42/43。
不根据新的验证/测试结果追加搜索或筛选种子；原始历史结果保留。

该受控重跑与 `table1_graphlora` 的 joint 协议对齐。不是冻结骨干实验，也不是
无法追溯运行配置的原表逐项精确复现。GPF-plus 与 GPF_plus_LoRA_Layer
直接复用 `prompt/GPF.py`，固定 p=20、r=2、主学习率、零提示权重衰减。
输入 GPF-plus 在输入投影之前（原子嵌入之后），所有消息层的 LoRAP
在量化聚合之后、原有 update 之前。GAT 先将各头展平，添加后恢复形状。
按宽度共享打分网络，各层独立低秩因子，因子 Xavier 初始化；不使用 GraphLoRA 损失。

DQ 的训练掩码保持原样，提示作用于合并后的高/低精度节点聚合。额外量化调用为零。
形状探测仅使用训练数据，恢复所有骨干/观察器/BN 状态及随机数流。
提示使用独立 CPU 随机数流，共同骨干初始指纹必须与配对对照一致。
已归档 GraphLoRA 的构造器会把 CUDA 随机数种子重置为 seed+930000；
此次入口显式匹配该训练起始随机数流。GPF 提示构造器本身不改变 CPU/CUDA 随机数流。

运行示例（从仓库根目录，使用 ProG Python；worker 从输出目录代码快照启动）：

```sh
python -B -m runs.lorap_study.table1_gpf_rerun.run prepare --root OUTPUT --table INPUT.tex
python -B -m runs.lorap_study.table1_gpf_rerun.run worker --root OUTPUT --device cuda:0
```

`--smoke --seeds 41` 对每个入选组合跑两轮小样本训练与验证，不接触测试集。
正式训练严格沿用配对配置：图任务 200 轮，节点 GIN/GCN 200 轮，节点 GAT 250 轮。
用最低验证 CE/MAE 选模，平局保留较早轮次；所有训练完成后每个检查点只测试一次。
支持 epoch 边界恢复、跨进程任务锁和测试锁。`REPORT.md`、`summary.json`、
`paired_differences.csv`、`resources.csv`、`table1_matched_rerun.tex` 随训练自动更新。
smoke 与正式输出目录必须分离；测试结果不参与训练或调参。
