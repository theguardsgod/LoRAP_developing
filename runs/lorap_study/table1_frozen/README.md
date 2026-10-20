# 表 I 冻结骨干实验

范围：5 数据集 × GIN/GCN/GAT × QAT/DQ/A²Q，共 45 组。10 种冻结方法、种子 41–43；先在种子 41 验证集调参，固定配置后再确认其他种子，全部结束后统一解锁测试。**必须 GPU。**

当前正式结果目录：`/home/cyliu/code/A-2Q/result/lorap_study/table1_frozen_20260914_v2`。v1 是保留的后端探索记录，preflight 是功能验证，不计入最终准确率。

协议和实现差异分别见 [PROTOCOL.md](PROTOCOL.md)、[BACKEND_NOTES.md](BACKEND_NOTES.md)。论文阈值在 registry.py，逐组配置与所有候选在结果目录 manifest.json。未结束的实验不能当作已得到性能结论。

冻结前额外完成了 [GIN 基线正则化搜索](GIN_BASELINE_SEARCH.md)，新增 10 次节点训练。Cora/CiteSeer 的 A²Q GIN 均按最低验证 CE 选择 `lr=0.01, weight_decay=0.2`；这两组实际配方以各 cell 的 `baseline_override.json` 为准。后续种子自动读取覆盖项，其余配置不变。原骨干保存在 `baseline_search/native`，所有方法共享推广后的骨干。

## 完整运行与续跑

从项目根目录运行。每张可用 GPU 一个 worker，使用文件锁取任务，完成的任务自动跳过，未结束任务从 last.pt 恢复；不会停止其他 GPU 进程。本轮先使用 GPU 2、3，GPU 0、1 空闲后也已加入队列。

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table1_frozen.suite worker --root result/lorap_study/table1_frozen_20260914_v2 --device cuda:0
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table1_frozen.suite worker --root result/lorap_study/table1_frozen_20260914_v2 --device cuda:1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table1_frozen.suite worker --root result/lorap_study/table1_frozen_20260914_v2 --device cuda:2
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table1_frozen.suite worker --root result/lorap_study/table1_frozen_20260914_v2 --device cuda:3
```

查看进度：

```sh
/home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table1_frozen.suite status --root result/lorap_study/table1_frozen_20260914_v2
```

全部训练与确认完成后的测试、推理导出、误差谱和中文报告：

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table1_frozen.finish --root result/lorap_study/table1_frozen_20260914_v2 --device cuda:2
```

该命令会检查全部选择锁和所有选定训练任务；缺任一组时拒绝访问测试集。测试预测分批保存，恢复时读取已有预测，不重复完整测试。

本轮同时启动了 `pipeline` 后续阶段，训练全部完成后自动执行上面的 finish；不是定时 Codex 唤醒。该本地进程每 30 秒更新结果目录 STATUS.md，遇到失败会留下 needs_attention.json。恢复后可运行：

```sh
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table1_frozen.pipeline --root result/lorap_study/table1_frozen_20260914_v2 --device cuda:2
```

单组调试（正式调参应使用完整队列，避免漏候选）：

```sh
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table1_frozen.train --root result/lorap_study/table1_frozen_20260914_v2 --device cuda:2 --cells A2Q_Cora_GIN --seeds 41
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.table1_frozen.adapt --root result/lorap_study/table1_frozen_20260914_v2 --device cuda:2 --cells A2Q_Cora_GIN --seed 41 --methods error
```

## 检查与部署

```sh
PYTHONDONTWRITEBYTECODE=1 TABLE1_DEVICE=cuda:2 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m unittest runs.lorap_study.table1_frozen.test_core -v
PYTHONDONTWRITEBYTECODE=1 TABLE1_DEVICE=cuda:2 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m unittest runs.lorap_study.table1_frozen.test_audits runs.lorap_study.table1_frozen.test_final_flow -v
```

每个选中方法输出 inference.pt，可直接加载：

```python
from runs.lorap_study.table1_frozen.deploy import load
model = load('/absolute/path/to/inference.pt', 'cuda:2')
logits = model(graph_without_labels.to('cuda:2'))
```

部署接口无 teacher 参数，不需要训练误差/参考特征/标签；保持原量化骨干。保存的是 fake-quantization 的 FP32 状态和量化参数，不能把 checkpoint 字节数称作已打包的 INT4 存储。

## 输出

每组：baseline、每方法每候选 config/history/progress/best/last/done、共享训练校准 calibration.pt、种子 41 的 selection.json。全局 SELECTION_LOCK.json、TEST_LOCK.json；测试后各模型的 test.json、inference.pt、diagnostics.json、quantization.json；最后生成 summary.csv、45 组误差谱图和 FINAL_REPORT.md。

GPF 和 weight LoRA 改变的位置不同，固定聚合补偿指标标为不适用。量化 task/error/MLP 与谱/条件版本的诊断同时保留自身输入和所有方法共享的无提示输入，两者不要混为一谈。

`baseline_validation_audit` 仅记录原骨干的验证选模敏感性分析；其中温度缩放是解释性诊断，未用于正式选择或部署。最终测试报告会单独披露基线搜索，区分超过论文参考与优于本轮配对骨干。
