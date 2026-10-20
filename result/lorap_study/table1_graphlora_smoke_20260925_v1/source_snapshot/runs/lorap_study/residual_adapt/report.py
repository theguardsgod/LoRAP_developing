"""Artifact-only audit and Chinese report for gated residual adaptation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
import torch

from ..run import write_json
from ..residual_channel.experiment import sha
from .model import is_bn_stat
from .run import ROOT, VARIANTS, verify_selection

LABELS=dict(frozen="原冻结模型",unit_gate="固定全量补偿",gate_only="只训练门控",
    none_bn="原模型重估 BN",unit_gate_bn="全量补偿重估 BN",gate_only_bn="学习门控后重估 BN",
    none_ft="无通道继续训练",gate_ft="门控＋骨干微调",higher16_frozen="加位宽冻结模型",
    higher16_ft="加位宽＋骨干微调")
DISPLAY=list(LABELS)
COLORS=dict(frozen="#777777",unit_gate="#b980ab",gate_only="#3073ad",none_bn="#928266",
    unit_gate_bn="#ba6483",gate_only_bn="#644fa8",none_ft="#666666",gate_ft="#d76b27",
    higher16_frozen="#80ac9d",higher16_ft="#278264")


def paired(values):
    a=np.asarray(values,float)
    mean=float(a.mean())
    radius=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return dict(mean=mean,ci95=[mean-radius,mean+radius],
                wins=int((a>1e-10).sum()),ties=int((abs(a)<=1e-10).sum()),losses=int((a< -1e-10).sum()))


def save_plot(fig,root,name):
    for suffix in ("png","pdf","svg"):
        fig.savefig(root/f"{name}.{suffix}",dpi=180,bbox_inches="tight")


def analyze(root):
    root=root.resolve()
    study=json.loads((root/"study.json").read_text())
    records,rows,histories=[],[],{}
    budgets=[]
    checked_sources={}
    for dataset in study["datasets"]:
        for seed in study["seeds"]:
            budget=json.loads((root/dataset/f"budget_seed_{seed}/result.json").read_text())
            assert budget["device"].startswith("cuda") and budget["source_unchanged"] and budget["test_evaluations"]==0
            for name,digest in budget["source_hashes"].items():
                assert sha(name)==digest
            budgets.extend(dict(dataset=dataset,seed=seed,method=k,**v) for k,v in budget["conditions"].items())
            group={}
            order_signatures=[]
            graph_indices=[]
            for variant in VARIANTS:
                path=root/dataset/variant/f"seed_{seed}"
                r=json.loads((path/"result.json").read_text())
                history=[json.loads(line) for line in (path/"epochs.jsonl").read_text().splitlines()]
                last=torch.load(path/"last.pt",map_location="cpu",weights_only=False)
                best=torch.load(path/"best.pt",map_location="cpu",weights_only=False)
                verify_selection(last,best)
                r["reference_extra_state_tensor_bytes"]=sum(v.numel()*v.element_size()
                    for name,v in best["model"].items() if not name.startswith("backbone."))
                assert last["history"]==history
                assert r["epochs"]==study["epochs"]==20
                assert r["device"].startswith("cuda") and r["test_evaluations"]==0
                assert r["source_artifacts_unchanged"] and r["quantizers_unchanged"]
                assert r["deterministic_replay"]["delta"]==0
                assert r["training_graphs"]==dict(MNIST=55000,CIFAR10=45000)[dataset]
                assert r["validation_graphs"]==5000
                for layer in r["selected_validation"]["local_diagnostics"]:
                    before,after,dot,correction=(layer[k] for k in ("before_sq","after_sq","dot","correction_sq"))
                    assert all(np.isfinite(v) for v in (before,after,dot,correction))
                    assert abs(after-(before-2*dot+correction))<=max(before,1.)*1e-6
                    assert abs(layer["mse_reduction"]-(1-after/max(before,1e-30)))<1e-12
                if variant=="gate_only":
                    assert r["trainable_parameters"]==4 and r["frozen_backbone_unchanged"]
                    source_backbone=torch.load(Path(study["source"])/dataset/"none"/f"seed_{seed}/best.pt",
                                               map_location="cpu",weights_only=False)["model"]
                    for name,intervention in r["bn_interventions"].items():
                        checkpoint=torch.load(path/f"{name}.pt",map_location="cpu",weights_only=False)
                        for key,value in checkpoint["model"].items():
                            if is_bn_stat(key):
                                assert torch.isfinite(value).all()
                                continue
                            expected=best["model"][key]
                            if name!="gate_only_bn":
                                if key.startswith("backbone."):
                                    expected=source_backbone[key.removeprefix("backbone.")]
                                elif key=="gates":
                                    expected=torch.full_like(value,0. if name=="none_bn" else 1.)
                            assert torch.equal(value,expected), (dataset,seed,name,key)
                        for metric in ("loss","accuracy","examples"):
                            assert checkpoint["validation"][metric]==intervention["after"][metric]
                if r["gates"] is not None:
                    assert all(0<=v<=1 for v in r["gates"])
                for name,digest in r["source_hashes"].items():
                    if name not in checked_sources:
                        assert sha(name)==digest
                        checked_sources[name]=digest
                    else:
                        assert checked_sources[name]==digest
                order_signatures.append([h["train"]["order_sha256"] for h in history[1:]])
                graph_indices.append(sha(path/"indices.json"))
                records.append(r)
                histories[(dataset,seed,variant)]=history
                group[variant]=r
                rows.append(dict(dataset=dataset,seed=seed,method=variant,**r["selected_validation"]))
            assert all(order==order_signatures[0] for order in order_signatures)
            assert len(set(graph_indices))==1
            assert len({r["initial_backbone_sha256"] for r in group.values()})==1
            assert len({r["quantizer_sha256"] for r in group.values()})==1
            rows.append(dict(dataset=dataset,seed=seed,method="frozen",**group["none_ft"]["initial_validation"]))
            rows.append(dict(dataset=dataset,seed=seed,method="higher16_frozen",**group["higher16_ft"]["initial_validation"]))
            for name,v in group["gate_only"]["bn_interventions"].items():
                rows.append(dict(dataset=dataset,seed=seed,method=name,**v["after"]))
            rows.append(dict(dataset=dataset,seed=seed,method="unit_gate",**group["gate_only"]["bn_interventions"]["unit_gate_bn"]["before"]))
    for name,digest in json.loads((root/"source_manifest.json").read_text()).items():
        assert sha(root/"source_snapshot"/name)==digest
    for name,digest in json.loads((root/"analysis_source_manifest.json").read_text()).items():
        assert sha(root/"analysis_snapshot"/name)==digest
    assert len(records)==24 and len(rows)==len(budgets)==60
    for b in budgets:
        r=next(r for r in rows if (r["dataset"],r["seed"],r["method"])==(b["dataset"],b["seed"],b["method"]))
        assert b["state_unchanged"] and b["examples"]==5000
        assert (b["accuracy"],b["loss"])==(r["accuracy"],r["loss"])
        assert b["total_activation_logical_bytes"]==sum(b[k] for k in
            ("feature_quantizer_logical_bytes","main_message_logical_bytes","side_message_logical_bytes"))
        for feature in b["feature_assignments"].values():
            assert sum(feature["histogram"].values())==feature["elements"]
            assert sum(float(k)*v for k,v in feature["histogram"].items())==feature["effective_bit_sum"]
    assert "Ran 7 tests" in (root/"tests.log").read_text() and "\nOK\n" in (root/"tests.log").read_text()
    comparisons=[]
    pairs=[("gate_only","frozen"),("gate_only_bn","none_bn"),("gate_only_bn","gate_only"),
           ("none_bn","frozen"),("unit_gate_bn","unit_gate"),("gate_ft","none_ft"),
           ("gate_ft","higher16_ft"),("gate_ft","frozen"),("higher16_ft","none_ft")]
    for dataset in study["datasets"]:
        for method,reference in pairs:
            a=[next(r for r in rows if r["dataset"]==dataset and r["seed"]==seed and r["method"]==method) for seed in study["seeds"]]
            b=[next(r for r in rows if r["dataset"]==dataset and r["seed"]==seed and r["method"]==reference) for seed in study["seeds"]]
            comparisons.append(dict(dataset=dataset,method=method,reference=reference,
                accuracy_delta=paired([(x["accuracy"]-y["accuracy"])*100 for x,y in zip(a,b)]),
                ce_improvement=paired([y["loss"]-x["loss"] for x,y in zip(a,b)])))
    verification=dict(training_runs=len(records),bn_interventions=18,source_files_verified=len(checked_sources),
        epochs_per_run=20,gpu_only=True,test_evaluations=0,unit_tests=7,paired_data_order=True,
        paired_initial_backbone=True,frozen_quantizers_verified=True,epoch_zero_included=True,
        best_checkpoint_tensor_identity=True,exact_validation_replays=24,
        bn_checkpoint_non_statistics_identity=True,local_squared_error_identity=True,
        selected_epoch_zero=sum(r["best_epoch"]==0 for r in records),
        gate_models_selected_with_all_zero=sum(r["gates"] is not None and max(r["gates"])==0 for r in records),
        actual_budget_conditions_verified=len(budgets))
    write_json(root/"verification.json",verification)
    write_json(root/"results.json",records)
    write_json(root/"analysis.json",dict(rows=rows,comparisons=comparisons,verification=verification,budgets=budgets))
    with (root/"per_seed.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=["dataset","seed","method","accuracy","loss","examples"])
        w.writeheader()
        for r in rows:
            w.writerow({k:v for k,v in r.items() if k in w.fieldnames})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    for dataset in study["datasets"]:
        fig,axes=plt.subplots(1,2,figsize=(13,4.4))
        for ax,metric,scale in ((axes[0],"accuracy",100),(axes[1],"loss",1)):
            for i,method in enumerate(DISPLAY):
                values=[r[metric]*scale for r in rows if r["dataset"]==dataset and r["method"]==method]
                ax.errorbar(i,np.mean(values),yerr=np.std(values,ddof=1),fmt="o",color=COLORS[method],capsize=3)
                ax.scatter(np.array([i-.13,i,i+.13]),values,s=13,color=COLORS[method],alpha=.6)
            ax.set_xticks(range(len(DISPLAY)),DISPLAY,rotation=45,ha="right",fontsize=8)
            ax.set_ylabel("Validation accuracy (%)" if metric=="accuracy" else "Validation cross-entropy")
            ax.grid(axis="y",alpha=.2)
        fig.suptitle(f"{dataset}: 20-epoch adaptation, 3 paired seeds (mean ± SD)")
        fig.tight_layout()
        save_plot(fig,root,f"{dataset}_comparison")
        plt.close(fig)

        fig,axes=plt.subplots(2,3,figsize=(13,6.5),sharex=True,sharey=True)
        for i,variant in enumerate(("gate_only","gate_ft")):
            for j,seed in enumerate(study["seeds"]):
                h=histories[(dataset,seed,variant)]
                for layer in range(4):
                    axes[i,j].plot([r["epoch"] for r in h],[r["gates"][layer] for r in h],label=f"L{layer+1}")
                selected=next(r["best_epoch"] for r in records if r["dataset"]==dataset and r["seed"]==seed and r["variant"]==variant)
                axes[i,j].axvline(selected,color="black",ls=":",lw=.8)
                axes[i,j].set_title(f"{variant}, seed {seed}")
                axes[i,j].set_ylim(-.02,1.02)
                axes[i,j].set_xlabel("Epoch")
                axes[i,j].grid(alpha=.2)
        axes[0,0].legend(fontsize=8,ncol=2)
        axes[0,0].set_ylabel("Gate strength")
        axes[1,0].set_ylabel("Gate strength")
        fig.suptitle(f"{dataset}: learned gates; dotted line = selected epoch")
        fig.tight_layout()
        save_plot(fig,root,f"{dataset}_gates")
        plt.close(fig)

        fig,axes=plt.subplots(1,3,figsize=(12,3.8))
        for j,seed in enumerate(study["seeds"]):
            for variant in VARIANTS:
                h=histories[(dataset,seed,variant)]
                axes[j].plot([r["epoch"] for r in h],[r["validation"]["loss"] for r in h],label=variant,color=COLORS[variant])
            axes[j].set_title(f"Seed {seed}")
            axes[j].set_xlabel("Epoch")
            axes[j].grid(alpha=.2)
        axes[0].set_ylabel("Validation CE")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{dataset}: validation learning curves (epoch zero retained)")
        fig.tight_layout()
        save_plot(fig,root,f"{dataset}_learning")
        plt.close(fig)

    text=["# 真实残差通道：门控与短程适配结果", "",
        "完成 MNIST、CIFAR10 各三个种子（41–43）、四个训练变体、每组 20 轮，共 24 次 GPU 训练；全部使用官方完整训练划分和完整验证划分。另完成 18 组 BN 重估干预。没有新增测试集前向。", "",
        "## 核心比较", ""]
    for dataset in study["datasets"]:
        for method,ref in (("gate_only","frozen"),("gate_ft","none_ft"),("gate_ft","higher16_ft")):
            c=next(c for c in comparisons if c["dataset"]==dataset and c["method"]==method and c["reference"]==ref)
            d=c["accuracy_delta"]
            text.append(f"- {dataset}，{LABELS[method]}相对{LABELS[ref]}：验证准确率 {d['mean']:+.3f} 个百分点，配对 95% 区间 [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]，胜/平/负 {d['wins']}/{d['ties']}/{d['losses']}。")
    text.extend(["", "这里分别判断门控是否有用、是否超出继续训练的收益、是否优于同消息预算。恢复原模型或仅优于固定全量补偿，均不能视为新机制取得性能优势。只有三个训练种子，以下区间不适合支撑论文级稳定收益；未进行多重比较校正。验证集已用于原检查点选择和前一阶段研究，属于探索性结果。", "",
        "## 完整验证结果", "", "准确率为均值 ± 样本标准差（%）；均按最低验证 CE 选模，不按最高准确率选模。", "",
        "| 数据集 | 条件 | 验证准确率 | 验证 CE |", "|---|---|---:|---:|"])
    for dataset in study["datasets"]:
        for method in DISPLAY:
            g=[r for r in rows if r["dataset"]==dataset and r["method"]==method]
            accuracy=np.array([r["accuracy"]*100 for r in g])
            text.append(f"| {dataset} | {LABELS[method]} | {accuracy.mean():.3f} ± {accuracy.std(ddof=1):.3f} | {np.mean([r['loss'] for r in g]):.6f} |")
    text.extend(["", "## 配对差值", "", "准确率差以百分点计；CE 改善 = 对照 CE − 当前 CE，正值更好。", "",
        "| 数据集 | 当前 / 对照 | 准确率差 [95% 区间] | CE 改善 [95% 区间] |", "|---|---|---:|---:|"])
    for c in comparisons:
        a,l=c["accuracy_delta"],c["ce_improvement"]
        text.append(f"| {c['dataset']} | {LABELS[c['method']]} / {LABELS[c['reference']]} | {a['mean']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | {l['mean']:+.6f} [{l['ci95'][0]:+.6f}, {l['ci95'][1]:+.6f}] |")
    text.extend(["", "## 门控、最佳轮次与最后轮次", "",
        "门控从零开始，逐步学习，每次更新投影到 [0,1]。门控只有四个标量，没有额外分类头或残差 MLP。epoch 0 参与选模；以下同时保留最后一轮表现，避免只看最佳模型掩盖适配退化。", "",
        "| 数据集 | 种子 | 条件 | 最佳轮次 | 最佳准确率 % | 最后准确率 % | 最佳门控 L1–L4 |", "|---|---:|---|---:|---:|---:|---|"])
    for r in records:
        gates="—" if r["gates"] is None else ", ".join(f"{g:.4f}" for g in r["gates"])
        text.append(f"| {r['dataset']} | {r['seed']} | {LABELS[r['variant']]} | {r['best_epoch']} | {r['selected_validation']['accuracy']*100:.3f} | {r['last_validation']['accuracy']*100:.3f} | {gates} |")
    text.extend(["", f"24 组中 {verification['selected_epoch_zero']} 组选择了训练前模型；门控模型中 {verification['gate_models_selected_with_all_zero']} 组的所选门控全为零。这些情况属于保留/恢复基线，不计作非零残差机制的成功。", "",
        "## 补偿是否仍存在", "",
        "以下测量各所选模型自身当前层输入处的局部量化误差，包含实际门控幅度。基线随骨干和前缀变化，不能把不同已微调模型的局部指标直接当成固定输入的同一比较。BN 校准与门控也不保证改善分类。", "",
        "| 数据集 | 条件 | L1 平方误差下降 % | L2 | L3 | L4 |", "|---|---|---:|---:|---:|---:|"])
    for dataset in study["datasets"]:
        for variant in ("gate_only","gate_ft"):
            g=[r for r in records if r["dataset"]==dataset and r["variant"]==variant]
            reductions=[np.mean([r["selected_validation"]["local_diagnostics"][i]["mse_reduction"]*100 for r in g]) for i in range(4)]
            text.append(f"| {dataset} | {LABELS[variant]} | "+" | ".join(f"{v:.3f}" for v in reductions)+" |")
    text.extend(["", "## 方法与公允对照", "",
        "骨干学习率 0.0001、门控学习率 0.01，均为 Adam，20 轮固定学习率、零权重衰减。只用分类 CE；所有量化步长/位宽参数固定，原存储惩罚对可训练的骨干和门控没有梯度。每轮核对量化器和固定残差基；只训练门控时骨干及 BN 全部冻结。骨干微调组均更新相同范围的骨干（含输入 GPF+、epsilon、BN 仿射及运行统计）。", "",
        "INT8 残差与 INT32 邻居传输每次前向重新计算，侧分支停止梯度；门控与主干保留梯度。该约定防止无意穿过整数转换，或用两个 STE 相减定义残差梯度。训练侧分支的值与前一阶段一致，但本轮没有训练残差基或尺度。", "",
        "BN 干预共用固定分层抽样的 1024 张训练图，逐层收集 eval 前缀下的节点加权均值和无偏方差，只更新运行统计。三种干预各自恢复对应初始状态：原模型、全量补偿模型、已选择的只训练门控模型。干预方案不按验证结果择优，全部报告。", "",
        "所有训练变体共享原始骨干检查点、配对种子和每轮图顺序；按最低验证损失选模，平局保留较早轮次，全部完成 20 轮。原模型及已有测试结果保持原样。这里仅检验固定学习率、冻结量化器和残差编码尺度的适配方式，没有搜索学习率或允许量化步长协同学习，因此不能把负结果推广为任何形式的任务适配都不可能有效。", "",
        "每节点每层额外消息仍为 16 位；固定基和尺度 3552 字节，加四个门控 16 字节，共 3568 字节。即使门控接近零，本实现仍计算侧通道，不宣称节省了消息。加位宽控制按相同消息预算增加最多 16 个通道的一位精度；本参考模型额外缓存通道索引（全 16 通道时四层共 512 字节 INT64 索引），实际新增状态张量字节见 results.json 的 reference_extra_state_tensor_bytes。若部署时将索引折叠进已有位宽/尺度槽位，可移除这个缓存，本轮未生成这种部署模型。相同消息预算不代表相同总模型存储。", "",
        "## 实际位宽与流量", "",
        "原量化器的 uniform 槽位分配依赖特征幅值，参数冻结不等于实际分配固定。以下补测全部所选条件的实际有效位宽，主激活均值包含特征量化与主消息量化，INT8 侧码另计。流量按完整验证过程中访问量化器的张量元素统计，不是峰值显存或实测网络带宽。初稿将存储惩罚写成数值常数不准确，现已更正；它只是对本轮可训练参数没有梯度，训练更新不受影响。", "",
        "| 数据集 | 条件 | 主激活平均有效位宽 | 含侧码逻辑流量 MiB | 原存储代理均值 |", "|---|---|---:|---:|---:|"])
    for dataset in study["datasets"]:
        for method in DISPLAY:
            b=[r for r in budgets if r["dataset"]==dataset and r["method"]==method]
            text.append(f"| {dataset} | {LABELS[method]} | {np.mean([r['mean_main_activation_effective_bits'] for r in b]):.5f} | {np.mean([r['total_activation_logical_bytes']/2**20 for r in b]):.3f} | {np.mean([r['legacy_storage_proxy_mean'] for r in b]):.5f} |")
    text.extend(["", "## 验证与追溯", "",
        f"7 项 GPU 单元测试通过，覆盖零初始/全量参考等价、零点门控梯度及投影、冻结量化器和编码基、BN 仅改统计、诊断和旁路一致性、epoch 0/平局选模及不可变检查点、配对顺序。24 组所选模型均通过检查点逐张量一致性和确定性验证 CE 精确复算；核对 {len(checked_sources)} 个唯一源文件哈希。", "",
        "源码及原始测试输出保存于 source_snapshot、source_manifest.json、tests.log；逐轮指标、门控梯度和恢复状态保存在各组目录。实际位宽审计使用 GPU，在验证集复算 60 个条件并逐项核对准确率与 CE 完全一致，不更新模型。报告生成阶段仅读取这些文件，不执行模型。", "",
        "## 图表", ""])
    for dataset in study["datasets"]:
        for kind,label in (("comparison","对照结果"),("gates","门控轨迹"),("learning","验证损失曲线")):
            text.extend([f"![{dataset} {label}]({root}/{dataset}_{kind}.png)",""])
    text.extend(["## 交付", "", "逐种子汇总见 per_seed.csv；结构化比较见 analysis.json；数值验收见 verification.json；复现和恢复命令见 RUN_COMMANDS.md。阶段内所有负结果保留。本轮没有启动新选基方法或扩大种子范围。"])
    (root/"REPORT.md").write_text("\n".join(text)+"\n")
    commands=f"""# GPU 适配复现与恢复

```sh
cd {root}/source_snapshot
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.residual_adapt.suite --root {root} --source {study['source']} --calibration {study['calibration']} --gpus 0 1 2 3
```

同一队列只补齐未完成作业，恢复到最后提交的轮次；已完成结果不重跑。若 STOP 文件存在，先移除才继续。

训练结束后，在 GPU 审计全部所选条件的实际位宽并生成报告（已完成的审计会跳过）：

```sh
cd {root}/analysis_snapshot
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.residual_adapt.finish --root {root} --gpus 0 1 2 3
```

单独复现一组，使用新的输出目录：

```sh
cd {root}/source_snapshot
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.residual_adapt.run --dataset CIFAR10 --variant gate_ft --seed 41 --epochs 20 --source {study['source']} --calibration {study['calibration']} --root /home/cyliu/code/A-2Q/result/lorap_study/residual_adapt_reproduction
```

只读取结果并重建报告：

```sh
cd {root}/analysis_snapshot
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.residual_adapt.report --root {root}
```

全部模型计算必须使用 GPU；CPU 仅用于读取结果和绘图。没有测试集评估命令。
"""
    (root/"RUN_COMMANDS.md").write_text(commands)
    print(json.dumps(dict(verification=verification,comparisons=comparisons),ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    analyze(parser.parse_args().root)
