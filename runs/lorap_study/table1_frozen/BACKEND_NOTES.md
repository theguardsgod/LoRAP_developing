# 对照实现与历史代码差异

1. A²Q 使用原仓库模型定义快照和原量化器；节点 GAT 去掉 softmax 后再 CrossEntropy 的双重变换，返回 logits。历史 qGATConv 的未使用 weight 冻结并单列，不计入有效可训练容量。
2. QAT/DQ 使用本地 PromptRepDQ 的 IntegerQuantizer、LinearQuantized、原始 message passing。QAT 为 INT4、STE+momentum，DQ 为 INT4、gradient clipping+percentile(.01)、度 CDF 概率 .1–.2，仅在训练使用高精度保护。冻结后 eval 全走低精度分支。
3. 原 PromptRepDQ 节点 GAT 默认第二层拼接多个分类头，输出不等于类别数；本轮统一为 8 个隐藏头、一个输出头。GCN/GAT 层间保留 ReLU/dropout，以与标准两层网络及论文宽度一致。此项是透明修正，不声称逐位复现历史指标。
4. 原 ZINC A²Q GAT 构造未传递 heads，实际退化为宽度 18；本轮依表 III 的 18×h，使用 8 头、总宽 144，最后一层单头 144。其他图 GAT 为 19×8=152。
5. 保留历史图 QAT GIN 的浮点密集 MLP（量化消息/聚合/更新）；DQ 的密集层也量化。实际存储统计须反映此区别，不能假设所有权重均 INT4。
6. 所有正式检查点按验证损失选，不重用按测试准确率保存的旧权重。A²Q 可学习 signed bit 参数投影到 [1,16]，避免 b=0 产生负 qmax。所有实际位宽和适配器开销单列。
7. GPU 原子 scatter 的求和顺序噪声能在后续 INT4 量化阈值处被放大；验证重放及零分支等价检查实际触发过失败。v2 对所有方法统一使用排序 CSR 分段求和、相同方式的注意力 softmax 和图池化，运算定义仍为原来的求和/均值。v1 的探索基线保留，正式比较从 v2 重新训练，不能混用。
8. 逐项核对原 optimizer 后，保持 GAT 的 q_gat_fea.bit 固定、只学习 QLinear.fea_quant.bit；节点 GAT 存储 proxy 的 QLinear 特征项乘 hidden，额外 GAT 特征项乘 hidden×heads。v1 探索分组有偏差，v2 已修正。
9. 原图 A²Q GCN/GAT 默认不添加自环，v2 遵循该设置，记录实际消息边。图 A²Q GIN 脚本没有显式 residual，与论文表 III 的泛化描述有出入；本轮遵循该代码，不声称消除了论文与工作区的所有差异。
10. 当前工作区 lr_gat_mnist.py 的历史训练调用把 test_loader 传给 train；本轮不运行该入口，统一仅使用官方 train。不能仅由这份已修改工作区脚本推断论文实验实际使用了同一流程。
