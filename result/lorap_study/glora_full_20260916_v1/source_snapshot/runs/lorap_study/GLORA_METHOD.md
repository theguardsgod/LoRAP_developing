# GLoRA joint-QAT：方法与实验口径

本实现采用 **GLoRA（One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning）** 的五支持张量、随机超网训练和逐层进化结构搜索。它接入现有量化 GIN 的全部 11 个线性层，包括输入投影、GIN 内部 MLP 和分类头。输入 GPF、聚合 LoRAP 均替换为无参数恒等模块。

方法依据为[论文](https://arxiv.org/abs/2306.07967)与[官方源码](https://github.com/Arnav0400/ViT-Slim/tree/390467ddf529d747168709ce2f007a85a0399cca/GLoRA)。`glora_official/` 保存提交 `390467ddf529d747168709ce2f007a85a0399cca` 的原始文件、MIT 许可证和 SHA256 清单。GPU 测试直接调用该版本的官方线性层，对全部 432 种结构核对前向及适配参数梯度。

## 计算与结构搜索

量化层只计算一次 `x_q` 和 `W_q`，采用官方支持张量公式：

```text
Linear(x_q, W_q + W_q ⊙ A + B, b + b ⊙ D + E + W_q C)
```

其中 `⊙` 为逐元素乘法。A/B 为权重缩放与加性修正，C 为经原始量化权重投影的输入无关偏置，D/E 为偏置缩放与加性修正。A/B 的选项是低秩、向量、标量或关闭；C 是低秩、向量或关闭；D/E 是标量、关闭或向量。每层 `4×4×3×3×3=432` 种结构。

低秩分支的 down 因子置零，up 因子使用 Xavier uniform，D/E 置零；所有结构初始化时均与无提示骨干等价。沿用官方公式，不添加标准 LoRA 的 `alpha/r` 缩放。适配参数保持 FP32；量化器调用次数及特征存储预算不增加。独立适配初始化与子网抽样随机数流保持公共骨干初始化和训练数据顺序配对。

训练时每层、每次前向独立均匀抽取结构；验证时使用固定全低秩 A/B/C 与向量 D/E 结构，按最低验证交叉熵选择超网检查点，平局取较早轮次。结构搜索固定该检查点，以验证交叉熵为目标，每层独立选择五路结构。默认种群 50、20 代、10 个精英，逐路径变异概率 0.2，共评估 1,000 个不重复候选。第一代包括全路径和全关闭两个候选；变异/交叉无法填满时，使用有界重试后的随机候选填充。候选缓存及搜索随机数均可恢复。

搜索和评估不更新 BatchNorm 统计。导出会真正删除未选中路径的超网参数；保留的支持张量不合并进量化权重，也不重新量化。rank=2 的 MNIST 超网有 13,300 个适配参数（51.95 KiB），最终导出参数量随所选结构变化。资源表分别列训练超网参数和导出后实际 FP32 参数，不能把 rank 相同当作参数量相同，也不能声称量化部署零额外开销。

## 本轮训练协议

用户确认沿用 Table V：骨干与既有可训练量化参数联合训练，方法名为 **GLoRA joint-QAT**。这与官方冻结预训练 ViT 骨干的训练协议不同，是 GLoRA 在当前 GNN/QAT 协议下的移植。

- 固定 INT4：保持原固定量化器、尺度初始化和学习率，冻结全部位宽，仅使用交叉熵。
- A2Q：保持原可学习特征位宽、尺度、存储约束和优化器基础分组。
- GLoRA：主学习率与骨干相同，权重衰减为 0；默认 rank=2。
- MNIST：完整官方划分内 55,000/5,000/10,000 张图；200 轮；batch size 128；种子 41–50；默认宿主机 `cuda:0`，无 CUDA 则报错。
- 对照：读取 `table_v_control_full_20260915_v1` 归档的相同种子无提示模型与标准 LoRA，核对共同骨干指纹、优化器基础分组、数据划分、依赖版本及原训练代码哈希，不重复测试旧模型。
- 全部 20 组训练和结构搜索完成后才释放测试；每个压缩检查点测试一次，中断过的测试需先检查，禁止自动重测。

原方法使用 500 轮超网训练并按验证准确率搜索；本轮继承既有 200 轮配置并使用验证交叉熵。官方 ViT 实现不向分类头注入支持张量；本轮按既定量化模块对照口径覆盖全部量化线性层，包含分类头。GLoRA 比无适配/标准 LoRA 多访问验证集进行结构搜索，报告将该搜索成本单列，不宣称等搜索预算。训练与搜索均保留可恢复状态。

原表的位宽定义、历史最高测试准确率选模与本轮口径并未完全对应。新增 LaTeX 行使用实际激活元素加权有效位宽、导出适配参数 KiB 和单次测试准确率；合并论文表格前应统一这些定义。`--smoke` 只验证真实数据 GPU 流程，不输出测试性能结论。

选择 GLoRA 是因为其线性层支持张量能直接接入当前同数据集量化分类骨干。[GraphLoRA](https://github.com/AllminerLab/GraphLoRA) 则针对跨图域迁移，包含源图预训练、结构感知约束和域对齐；实现完整 GraphLoRA 还需另立源/目标域实验协议。本实现不使用 GraphLoRA 名称。

## 运行与产物

```sh
# 数值、搜索、训练恢复及相关回归；GPU oracle 和集成测试需要宿主机 CUDA。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m unittest runs.lorap_study.test_glora runs.lorap_study.test_glora_workflow runs.lorap_study.test_lora runs.lorap_study.test_full runs.lorap_study.test_full_validation runs.lorap_study.test_table_v runs.lorap_study.test_table_v_control -v
# 两种模式的真实数据冒烟：128/64/64、2 轮、4×2 候选，不访问测试集。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.glora_run --smoke --output-dir result/lorap_study/glora_smoke
# 正式完整 MNIST，2 种量化模式 × 10 种子。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.glora_run --output-dir result/lorap_study/glora_full
# 从保存的源码与配置恢复，不临时修改超参数。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.glora_run --resume --output-dir result/lorap_study/glora_full
```

入口支持量化模式、种子、轮数、设备、rank、搜索预算和输出目录配置。每组保存逐轮日志、`last.pt`、最低验证 CE 的 `best.pt`、`search.pt`、所选结构、删除无用参数后的 `compact.pt`、实际位宽及最终指标。结果根目录保存代码和官方源码快照、对照证据、数据索引、初始化预检、动态中文报告、逐种子配对差值、资源统计和 `table_v_glora_rows.tex`。所有测试完成后生成不重复推理的完整性审计。
