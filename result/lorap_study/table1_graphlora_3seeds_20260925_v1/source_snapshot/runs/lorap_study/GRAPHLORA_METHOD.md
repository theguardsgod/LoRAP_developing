# GraphLoRA-adapted joint-QAT

本实现用于用户确认的 MNIST Table V：固定 INT4 QAT 与 A2Q，沿用量化 GIN 骨干和联合训练。名称固定为 **GraphLoRA-adapted joint-QAT**。官方 GraphLoRA 的完整任务是源图预训练后的跨图节点分类；本实现将图分支与两个辅助目标移植到同域图分类，不声称复现原论文实验。

来源：[论文](https://arxiv.org/abs/2409.16670)、[官方固定提交](https://github.com/AllminerLab/GraphLoRA/tree/aa9ab3dec32cee143c27fdbf07806d33356690db)。原始源码及文件哈希保存在 `graphlora_official/`。测试直接导入官方 `util.py`，核对双向监督对比损失的值和梯度；结构损失另与官方的加权邻接重建公式核对。

## 保留和改动

| 项目 | 官方 GraphLoRA | 当前 Table V 移植 |
|---|---|---|
| 任务 | 跨图节点分类 | MNIST 超像素图分类 |
| 骨干 | 预训练 GAT，迁移时冻结 | 原有量化 GIN，骨干与量化参数联合更新 |
| 低秩图分支 | 与原图卷积并行，逐层相加 | 四层 GIN 分别加入 FP32 低秩图分支，逐层相加 |
| 输入对齐 | 特征投影与 SMMD，需要源图 | 无跨域源图，不添加投影，不启用 SMMD |
| 对比样本 | 末层原分支和残差分支的节点表示 | 末层原分支和合成分支的池化图表示 |
| 监督正样本 | 训练节点中同类节点 | 训练图中同类图；不向超像素传播整图标签 |
| 结构目标 | 节点预测概率重建邻接 | 共享分类头计算训练用节点概率，仅重建图内邻接 |
| 自配对 | 官方实现添加自环 | 默认排除自配对、跨图节点对和 padding |

使用合成图表示进行对比可避免初始化为零的残差表示被直接归一化；对比损失公式仍与官方双向目标一致。不同图中没有边，不构造跨图负边。图分类标签只参与图级交叉熵和对比监督。

## 低秩 GIN 分支

每层仅调用一次现有消息量化器，两个分支使用同一量化消息张量 `q(x)`。主分支保持原量化 MLP。旁路先计算 `sum_neighbors(q(x)) + (1 + eps_adapter) q(x)`，再经过低秩线性、BatchNorm、ReLU、低秩线性，输出与主分支相加。不存在独立的可训练输入提示或聚合提示。

GIN 的两个低秩线性层都使用无偏置 `up(down(x))`；down 用 Kaiming normal。第一层 up 也使用 Kaiming normal，只有末层 up 置零。末端不加 ReLU，以保留零点梯度及有符号残差。若两层都置零，整个 MLP 会失去有效梯度，因此没有直接照搬单投影 GAT 的初始化。

rank=2、宽度 110、四层时，旁路有 4,404 个训练参数，FP32 参数开销 17,616 字节（17.20 KiB）；额外 BatchNorm 缓冲区为 3,552 字节。它们不自动合并进量化权重。相比标准 LoRA 的 4,426 个参数，参数量接近但并不相等。两者计算和训练目标也不相同。

旁路初始化使用独立随机数作用域，保持公共骨干和原有数据顺序配对。旁路优化器使用骨干主学习率、零权重衰减，其余参数分组、量化位宽冻结规则和调度器沿用对应 Table V 模式。

## 损失与评估

```text
objective = graph_cross_entropy + existing_storage_penalty
            + 0.1 * supervised_graph_contrast + 0.1 * within_graph_structure_loss
```

对比温度为 0.5，同类非自身正样本权重为 0.2。系数 0.1/0.1 是本轮移植预实验事先固定的配置，并非官方 MNIST 超参数（官方没有该任务），未根据测试集选择。节点结构概率在训练时复用已有分类头的 FP32 主权重，既不新增节点分类器，也不再次调用分类头量化器；这条训练专用计算路径及旁路计算计入耗时和峰值显存。

结构损失是 sigmoid(概率内积) 与邻接的加权 BCE；每张图独立计算正负比例及有效节点对数，再对图平均。空边集保持有限值，单节点无有效配对时贡献为零。SMMD 系数固定为 0，并明确记录 `cross_graph_source=null`。

默认先进行完整 MNIST 预实验：55,000/5,000/10,000 的官方划分、batch size 128、种子 41、30 轮，两个量化模式各有无适配和 GraphLoRA-adapted 同期对照，共四组。预实验只使用训练集和验证集，测试次数为 0；它的准确率不能填到论文测试列。默认 QAT/A2Q 主学习率及其量化配置保持原样。

`full` 模式默认种子 41–50、200 轮，包含两种模式及两个变体，共 40 组；所有预定训练都完成后才统一释放一次最终测试。选模严格使用最低验证交叉熵，平局取较早轮次。测试中断不自动重测。完整实验的测试结果才会填入输出 LaTeX 行。

有效位宽按量化骨干的激活元素统计；FP32 旁路参数、缓冲区单列，旁路中间激活没有混入这个骨干位宽指标。合并原 Table V 前仍需统一位宽与历史选模口径。

## 运行

```sh
# 模块、官方损失数值对照、GPU 恢复及流程测试。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m unittest runs.lorap_study.test_graphlora runs.lorap_study.test_graphlora_workflow -v
# 两模式、两变体：128/64/64、2 轮，无测试集评估。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.graphlora_run --profile smoke --output-dir result/lorap_study/graphlora_smoke
# 完整数据配对预实验：默认一个种子、30 轮，无测试集评估。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.graphlora_run --profile pilot --output-dir result/lorap_study/graphlora_pilot
# 正式完整实验。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.graphlora_run --profile full --output-dir result/lorap_study/graphlora_full
# 从已保存的源码、配置和检查点恢复。
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.graphlora_run --resume --output-dir result/lorap_study/graphlora_pilot
```

默认 `cuda:0`，没有 CUDA 则报错。支持模式、种子、轮数、rank、两个辅助损失系数、设备与输出目录配置。创建结果根目录 `STOP` 文件，会在当前轮完整提交之后暂停；恢复入口自动删除 `STOP`。

结果保存源码快照、上游源码及哈希、配置、划分索引、初始化指纹、优化器分组、逐轮各损失与验证指标、最优及恢复检查点、有效位宽、耗时/显存、配对差值和中文报告。完整性审计不重复推理或测试。
