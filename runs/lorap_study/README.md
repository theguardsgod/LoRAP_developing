# LoRAP 首轮对照实验

## 标准权重 LoRA 联合 QAT baseline

`lora_run.py` 比较 `no_prompt`（完全无提示）、`main`（固定提示方法）、`lora`（标准权重 LoRA）。
`lora` 和 `no_prompt` 同时移除输入 GPF 与聚合 LoRAP；`main` 在 MNIST 保留输入 GPF。
这与历史实验的 `none`（只去除聚合提示）不同。所有组继续联合训练骨干及既有可训练量化参数；
这是 LoRA joint-QAT 对照，不是冻结骨干的 PEFT，也不是 GLoRA/GraphLoRA 论文复现。

全部量化线性层（包括输入投影、GIN MLP 和分类头）增加
`Linear(x_q, W_q, bias) + (alpha/r) * x_q @ A.T @ B.T`。
两个分支共用一次特征量化，预算累计一次；适配因子保持 FP32，不自动合并或重新量化。
默认 rank=2、alpha=2、dropout=0，A 使用 Kaiming 初始化、B 全零，适配器学习率等于骨干主学习率，权重衰减为零。
参考：[LoRA 官方实现](https://github.com/microsoft/LoRA)。

```sh
# 单元测试；可在 CPU 上运行，CUDA 专项会在不可用时跳过。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m unittest runs.lorap_study.test_lora -v
# 宿主机 GPU 冒烟检查：三数据集 × 三方法 × 首个种子，2 轮，不评估测试集。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.lora_run --smoke --output-dir result/lorap_study/lora_smoke_repeat
# 正式首轮：Cora/CiteSeer 完整公共划分 200 轮；MNIST 512/256/256 图 30 轮。
# 三方法 × 三数据集 × 种子 41/42/43，共 27 组，默认 cuda:0。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.lora_run --output-dir result/lorap_study/lora_joint_qat_repeat
```

支持 `--datasets`、`--variants`、`--seeds`、`--node-epochs`、`--graph-epochs`、`--device`、
`--lora-rank`、`--lora-alpha`、`--threads` 和单数据集的 `--data-root`。正式训练和冒烟检查都要求 CUDA；
首次输出目录必须不存在。使用 `--resume --output-dir ...` 从保存的配置和代码快照恢复，其他命令行设置被忽略。

各组公共骨干初始化及训练随机数流配对；main 仅用训练数据校准。最低验证 CE 选模，平局取早，
每个数据集全部预定训练完成后才统一释放测试，每个最佳检查点只测试一次。异常中断的测试不会自动重复。
训练从 `source_snapshot` 执行，保存 `last.pt` 以恢复模型、优化器、调度器、随机数及数据加载器状态。
`REPORT.md`、`per_seed.csv`、`paired_differences.csv`、`status.json` 包含准确率、配对差值、参数量、
FP32 适配开销、校准缓冲区开销、实际量化位宽、训练/验证耗时和峰值显存；不是整数部署测量。
相同 rank 不代表等参数量，MNIST 小样本和三个种子不支持论文级性能结论。

仅重建报告：`python -B -m runs.lorap_study.lora_report /absolute/path/to/output`。

面向 MNIST 完整数据比较时，显式设置 `--full-data --graph-epochs 200 --seeds 41 42 43 44 45 46 47 48 49 50`，
并用 `--datasets MNIST --variants lora` 单独训练新增 baseline；需要同期无提示对照时改为 `--variants no_prompt lora`。
`--full-data` 与 `--smoke` 互斥。该入口仍按验证集 CE 选模，不能直接等同于历史脚本的最高测试准确率。
向历史表格追加结果前还须核对 Average bits 的计算定义；当前报告记录的是实际访问量化器的有效激活位宽。

独立于历史脚本，比较原 LoRAP、量化条件 LoRAP 和参数量匹配的残差 MLP。复用仓库已有量化器与 GIN 结构；不增加辅助损失。

## 使用

从仓库根目录运行。当前已验证的解释器是 `/home/cyliu/anaconda3/envs/ProG/bin/python`，实际版本为 Python 3.12、PyTorch 2.7.0+cu118、PyG 2.6.1；此会话仅可用 CPU。

```sh
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m unittest runs.lorap_study.test_study -v
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.run --profile pilot --dataset both --device cpu --threads 4
```

`pilot` 的 CiteSeer 是公共划分、三个变体、种子 41/42/43、100 轮；MNIST 是官方划分内固定分层抽样 512/256/256 图、种子 41、2 轮、batch size 128。MNIST 结果只验证功能。

支持 `--dataset CiteSeer|MNIST|both`、`--variant original|conditioned|mlp|all`、`--seeds 41 42`、`--epochs`、`--device`、`--threads`、`--output-dir`、`--data-root`、`--no-diagnostics`。输出目录必须不存在，以防覆盖。自定义数据路径需要选择单个数据集。

每次运行自动生成两个数据集的完整配置（全部数据，200 轮，种子 41–50）。也可只生成配置：

```sh
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.run --write-configs-only
```

使用生成的 `full_configs/MNIST.json` 或 `full_configs/CiteSeer.json`：

```sh
python -B -m runs.lorap_study.run --config /absolute/path/to/full_configs/MNIST.json --device cuda:0
```

## 方法与公平性

- MNIST：四层 110 维 GIN，保留输入 GPF-plus，四层共享聚合提示；CiteSeer：两层 16 维 GIN，在第二层聚合后加提示，无输入提示。
- 提示数量 20、rank 2。条件为 `log1p(入度)`、通道平均 `log(abs(scale))`、通道平均 `round(abs(bit))/8`、邻居量化值处于端点的平均比例。用实际去除自环后的消息边计算，条件停止梯度；对数仅在条件分支截断到 1e-12，不修改量化器。
- 条件化公式为 `S + 2*sigmoid(g(c))*softmax(a(S)+B(c))@(P_left@P_right)`。`B` 无偏置，`g` 含偏置，均零初始化，保留原基矩阵和打分层初始值。
- MLP 为带偏置的 `Linear(d,h) → ReLU → Linear(h,d)` 残差。按 `h(2d+1)+d` 选择最接近条件 LoRAP 参数量的宽度，差异不超过 5%。
- 骨干／输入提示和聚合提示使用分离 RNG 流；训练、dropout 和 DataLoader 使用配对种子。每次运行保存初始化 SHA256。
- 提示／适配器主学习率与骨干相同，权重衰减为零。量化超参数沿用 README 对应 GIN 设置；CiteSeer dropout 为 0.5。历史脚本中不同提示参数学习率分组在本实验中统一，因此本实验并非逐项复刻旧日志。
- 保留原量化函数及存储惩罚；历史优化器未更新的权重／消息位宽显式冻结，特征位宽继续学习。实验记录各量化器的实际位宽分配，绝不把 bit=4 或预算代理当作物理存储量。

## 评估与诊断

按最低验证交叉熵选模，平局取较早轮次。恢复最佳模型后只评估一次测试集。另在验证数据运行诊断，关闭诊断时无额外浮点参考聚合。

参考只绕过当前层的消息量化，使用同一层输入、同一边集；其他层量化与提示保持不变。记录局部误差补偿前后 Frobenius 范数、MSE、相对降低率、全局与逐节点余弦、误差谱及 rank 2/90%/95% 能量。零误差或零修正时不定义的指标保存为 null。

谱分析每层采用固定种子的随机优先级抽样，最多 4096 个验证节点；全量误差指标不受谱抽样影响。CiteSeer 消息传递仍覆盖整图，但诊断只统计验证 mask；MNIST 只访问验证图。各变体的上游表示可能不同，局部参考不是共同的完整 FP32 教师。

## 输出

默认输出到 `result/lorap_study/<时间_进程号>/`，含中文 `REPORT.md`、`summary.csv`、PNG/PDF/SVG 误差谱、完整配置、数据划分索引、环境和源代码快照。每个数据集／变体／种子有独立配置、初始化指纹、逐轮日志、`best.pt`、测试结果、误差诊断和位宽分配。

可重新生成报告：`python -B -m runs.lorap_study.report /absolute/path/to/output`。

验收以流程可训练、数据可追溯和指标正确为准。负结果同样报告；小规模实验不作显著性或论文级性能声明。本轮没有 GPU 端到端计时、实际整数部署或峰值显存测试。

## 补充机制验证：量化补偿与容量

`mechanism.py` 在 CiteSeer 上执行 QAT/FP32 × 无提示/原 LoRAP/等参数 MLP 的六组对照，均使用种子 41/42/43、100 轮。这里 MLP 为 16→12→16，412 个参数，精确匹配原 LoRAP；与前轮匹配条件 LoRAP 的 511 参数 MLP 区别开来。

```sh
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m unittest runs.lorap_study.test_study runs.lorap_study.test_mechanism -v
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.mechanism --output-dir result/lorap_study/mechanism_repeat --threads 4
```

默认在核对代码依赖、配置、划分、初始化和验证结果后，复用 `result/lorap_study/pilot_20260909_v1` 的三个 QAT 原 LoRAP 检查点及既有测试指标；`--no-reuse` 从头训练全部 18 组。FP32 完全绕过所有量化器、冻结量化参数、关闭存储惩罚；其他训练配置和选模规则保持配对。

只有验证集参与固定检查点干预：关闭提示、反号、节点置换、提示均值、沿完整 GIN 量化修正方向的平行/正交分解，以及消息误差强度插值。完整误差包含 GIN 自身节点项。分解和插值使用局部未量化参考，是诊断工具，不属于可部署方法；同一 QAT 检查点临时关闭全部量化也不能代替独立训练的 FP32。

报告同时列出准确率、交叉熵、相对无提示的配对增益、QAT–FP32 交互、每种子结果及全部干预。三种子结果只提供机制线索；未观察到误差范数降低不能排除任务相关补偿，FP32 增益也不能单独区分容量和优化效果。

## 条件化、零初始化与分层容量改进

`improved.py` 提供训练前校准、零输出低秩提示、标准化量化条件和固定总预算的层间 rank 分配。历史模块不变。

- `zero_init`：首轮条件 LoRAP 的左因子置零，其他初始化保留。
- `stable_random`：零左因子、随机正交右因子、训练统计量标准化条件、逐节点 RMS 查询归一化和提示幅度缩放。
- `stable_spectral`：相同结构，右因子改用训练校准完整 GIN 局部误差的右奇异向量，均匀 rank。
- `adaptive_spectral`：在固定总 rank 下，依据训练校准的相对误差谱分配各层 rank。MNIST 总 rank 为 8；均匀和自适应版本都共享打分、条件投影、门控，均为 3345 个聚合提示参数。CiteSeer 只有一个提示位置，保留 rank 2、497 个参数。

校准只用训练节点/图，最多 4096 个节点/层，MNIST 只用前 128 张训练图。局部未量化参考仅用于离线校准和显式开启的诊断，推理分支不运行参考或 SVD。校准保存原始与去均值误差谱、条件统计、基向量及分配依据；校准不会更新模型权重或 BatchNorm 状态。

```sh
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m unittest runs.lorap_study.test_study runs.lorap_study.test_mechanism runs.lorap_study.test_improved -v
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.improve_run --output-dir result/lorap_study/improvement_repeat --threads 4
```

默认 CiteSeer 首个种子的验证 CE 筛选 `zero_init/stable_random/stable_spectral`，锁定后仅对选中版本训练其余种子，最后测试；未选中的候选不测试。MNIST 默认原 LoRAP、均匀谱基、自适应谱基三个对照，512/256/256 图，30 轮，种子 41/42/43。30 轮是新一轮小样本开发实验，有同期原版对照，不与先前 2 轮功能验证直接比较。

可指定 `--dataset`、`--variant`、`--seeds`、`--epochs`、`--threads`、`--full-data` 和独立输出目录。完整数据命令示例：

```sh
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.improve_run --dataset MNIST --variant adaptive_spectral --full-data --epochs 200 --seeds 41 42 43 44 45 46 47 48 49 50 --output-dir result/lorap_study/improved_mnist_full --threads 4
```

结果包括中文报告、候选筛选记录、初始化与校准、逐轮日志、检查点、独立测试、误差诊断、位宽、图和完整配置。运行 `python -B -m runs.lorap_study.improve_audit /absolute/path/to/output` 可检查初始化、预算、选模和验证指标，不再次测试。

## 固定主版本：零初始化＋标准化条件＋均匀分层 rank

`uniform.py` / `uniform_run.py` 将该组合固定为 `main`：每个聚合提示层独立 rank 2；左因子置零、右因子随机正交初始化；共享打分、条件投影与门控。条件采用冻结的训练校准均值/尺度标准化并截断到 [-5,5]；RMS 查询归一化及输出幅度缩放沿用上轮结构。初始提示输出严格为零，两个低秩因子不会同时置零。

单独运行 `main` 的校准只计算量化聚合与条件统计，不运行浮点参考或 SVD。每层最多抽样 4096 个训练节点，MNIST 校准最多用前 128 张训练图；统计缓冲区随检查点保存。CiteSeer 只有第二层的一个聚合提示位置，固定 rank 2；MNIST 四层固定 [2,2,2,2]，聚合提示共 3345 个参数。

默认组件对照：无聚合提示、原共享 LoRAP、主版本、取消零初始化、取消条件标准化/截断、关闭条件投影与门控、误差谱基。默认 MNIST 子集 512/256/256、30 轮、种子 44/45/46；全部训练完成后，各最佳验证 CE 检查点只测试一次，不根据本轮结果选择主版本。无条件控制少 85 个参数，原共享 LoRAP 少 865 个参数，这些比较不是严格等容量。

```sh
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m unittest runs.lorap_study.test_study runs.lorap_study.test_mechanism runs.lorap_study.test_improved runs.lorap_study.test_uniform runs.lorap_study.test_uniform_context -v
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.uniform_run --dataset MNIST --output-dir result/lorap_study/uniform_repeat --threads 4
# 仅运行固定主版本：加 --variants main；关闭参考诊断：加 --no-diagnostics。
```

每次运行会生成两个数据集的主版本完整配置（全划分、200 轮、种子 41–50），可直接加载：

```sh
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.uniform_run --config result/lorap_study/uniform_20260909_v1/configs/MNIST_main_full.json --output-dir result/lorap_study/uniform_mnist_full --threads 4
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.uniform_run --config result/lorap_study/uniform_20260909_v1/configs/CiteSeer_main_full.json --output-dir result/lorap_study/uniform_citeseer_full --threads 4
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.uniform_audit result/lorap_study/uniform_20260909_v1
```

支持 `--config`、`--dataset`、`--variants`、`--seeds`、`--epochs`、`--threads`、`--full-data`、`--[no-]diagnostics` 和独立输出目录。`--write-configs-only` 仅生成配置。报告保存配对差值、误差谱、实际位宽与预算、逐种子结果；初始化及验证重现审计不会再次测试。新种子仍沿用旧数据划分，不能作为独立新测试集，也不能仅凭准确率判断量化补偿机制。

## 多数据集完整 GPU 训练

`full_suite.py` 提供 CUDA 强制检查、完整划分、四 GPU 调度、逐轮断点续训和自动汇总。默认 Cora/CiteSeer/MNIST/CIFAR10 × 无聚合提示/原 LoRAP/随机基改进版/谱基改进版 × 种子 41–50，每组 200 轮，共 160 组训练。GPU 不可用时明确报错，不降级为 CPU 训练。

```sh
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m unittest runs.lorap_study.test_full -v
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.full_suite --root result/lorap_study/full_gpu_repeat --gpus 0 1 2 3
```

需要在能够访问 GPU 的宿主机环境运行；沙箱可能屏蔽 `/dev/nvidia*`，此时需使用获准的宿主机执行入口。每张 GPU 同时运行一个工作进程，前半数优先节点任务，后半数优先图任务，任务完成后继续调度。每个数据集的所有预定训练完成后才统一释放该数据集的最终测试。

工作进程从结果目录的 `source_snapshot` 执行，后续工作区修改不会混入这批实验。`last.pt` 保存模型、优化器、调度器、CPU/CUDA 随机数、数据顺序及验证最优模型；`best.pt` 保存最低验证 CE 的检查点。每轮提交一次，恢复时会清理未提交的日志尾部。测试开始后异常退出会保留记录并停止该项，避免自动重复测试。

`REPORT.md`、`status.json` 和每组 `progress.json` 自动更新；完整测试结果保存到 `per_seed.csv` 和 `results.json`，并自动生成各数据集的 PNG/PDF/SVG 对照图。训练完成的检查点会在测试前重现验证损失；最终归档审计只读日志和权重，不再测试。

暂停时在结果根目录创建 `STOP`，运行中的训练会在当前轮次结束后暂停；恢复命令见该结果目录的 `RUN_COMMANDS.md`。配置和初始化变化会被拒绝，已完成训练与已有测试结果直接复用。离线训练校准和 SVD 可在 CPU 计算，所有训练更新及最终评估使用 GPU。

## Table V：固定 INT4 QAT 与 A2Q 的 LoRA 行

`table_v_run.py` 在两个协议下分别运行完整 MNIST（55,000/5,000/10,000 图）、200 轮、种子 41–50；每组去除所有提示，11 个量化线性层加入 rank=2、alpha=2 的 FP32 LoRA，共 4,426 个额外参数（17.29 KiB）。固定 INT4 使用 `u_quant_normal_debug.QLinear`，冻结全部位宽，仅优化 CE；A2Q 保留可学习特征位宽和存储约束。INT4 的尺度学习率与初始化依据 `result/GIN_normal_MNIST/GIN_110__on_MNIST_4bit-200.txt` 中记录 94.4 ± 0.2 的配置。

```sh
# 先执行两种量化模式的真实 GPU 冒烟检查（不访问测试指标）。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table_v_run --smoke --output-dir result/lorap_study/table_v_smoke
# 默认使用 cuda:0；允许通过 --devices 指定多个 GPU，每张 GPU 一个训练进程。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table_v_run --output-dir result/lorap_study/table_v_full
# 从保存的源码、配置和逐轮检查点恢复。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table_v_run --resume --output-dir result/lorap_study/table_v_full
```

两个协议所有预定训练完成后才释放测试。`REPORT.md`、`status.json` 和 `table_v_lora_rows.tex` 自动更新；完成前准确率保持 TBD。位宽为验证集实际访问的激活元素加权有效位宽，FP32 LoRA 开销单列。原表位宽定义及旧脚本最高测试准确率的选模口径与本轮未完全对应，应在论文注明，不能将先前 512/256/256 小样本结果直接填入原表。

单独运行某一种模式可使用 `lora_run --datasets MNIST --quantization fixed4`（或 `a2q`）；原入口默认配置及历史实验快照保持兼容。

## Table V 的同协议无 LoRA 对照

`table_v_control.py` 从已完成的 Table V LoRA 结果继承配置与源码快照，正式配置仅把 `variants` 改为 `no_prompt`。两种模式 × 相同种子 41–50，共补跑 20 组完整 MNIST 对照。提示及 LoRA 参数均为 0；逐种子检查共同骨干指纹和优化器基础分组。旧 LoRA 结果只读归档，不重新训练或测试。

```sh
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table_v_control --reference-root result/lorap_study/table_v_lora_full_20260911_v1 --smoke --output-dir result/lorap_study/table_v_control_smoke
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table_v_control --reference-root result/lorap_study/table_v_lora_full_20260911_v1 --output-dir result/lorap_study/table_v_control_full
# 恢复时继承保存的配置和源码，不接受临时调参。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table_v_control --resume --output-dir result/lorap_study/table_v_control_full
```

默认仍用 `cuda:0`；所有新增训练完成后才统一释放测试。`REPORT.md` 和 `status.json` 自动更新；完成后输出逐种子配对差值 `paired_differences.csv`、资源统计 `resources.csv` 和不重复测试的 `completion_audit.json`。这属于在看到 LoRA 结果后补充的固定协议对照，未重新调参；报告中保留这一说明。

## GLoRA（One-for-All）joint-QAT baseline

`glora.py` 实现官方 GLoRA 五支持张量及每层 432 种结构；`glora_search.py` 实现验证集进化搜索；`glora_run.py` 运行完整 MNIST 的固定 INT4/A2Q 联合训练、搜索、压缩导出和一次性测试。默认与已有无提示/标准 LoRA 对照使用相同初始化、划分、200 轮和种子 41–50。官方固定提交的源文件与 MIT 许可保存在 `glora_official/`；算法公式、协议差异、测试及运行命令见 [GLORA_METHOD.md](GLORA_METHOD.md)。

## GraphLoRA-adapted：Table V 图分类移植

`graphlora.py` 为四层量化 GIN 增加并行低秩图分支、图级监督对比和图内结构约束；`graphlora_run.py` 在固定 INT4 与 A2Q 下联合训练，并运行同期无适配对照。默认先做完整 MNIST 的 30 轮、单种子配对预实验，只看验证集。当前同域设置没有源图，SMMD 不启用，命名明确保留 `adapted`。具体算法差异、参数计数和完整运行命令见 [GRAPHLORA_METHOD.md](GRAPHLORA_METHOD.md)。

## GraphLoRA-adapted：TABLE1 三种子扩展

`table1_graphlora` 默认使用种子 **41、42、43**，覆盖 5 数据集 × 3 骨干 × QAT/DQ/A2Q，共 **135 次新增训练**。沿用已归档 TABLE1 后端、配置和无适配对照，独立结果目录；默认联合训练，不追加超参数搜索，不修改已完成的 Table V 十种子结果。支持 GPU 多 worker、断点恢复和统一最终测试门禁。运行方式及节点分类、GCN/GAT、ZINC 回归的具体适配说明见 [table1_graphlora/README.md](table1_graphlora/README.md)。
