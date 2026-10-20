"""Artifact-only audit, paired summaries, figures and Chinese phase-one report."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from ..run import write_json
from .experiment import DEFAULT_ROOT, MODES, sha

LABELS = dict(none="无补偿", random="随机基残差", pca="残差 PCA", graph="图传播基残差",
    higher_8="主消息加 ≤8 位", higher_16="主消息加 ≤16 位", higher_32="主消息加 ≤32 位",
    higher_64="主消息加 ≤64 位", higher_110="主消息加 ≤110 位", float_graph="浮点残差码（参考）",
    shuffle_graph="图内打乱残差码", bypass="旁路消息量化", bypass_side="旁路量化＋残差通道")
COLORS = dict(none="#777777",random="#8d79bd",pca="#3683b8",graph="#d76b27",
              higher_16="#27906d",shuffle_graph="#b45368")


def average(values):
    a = np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),std=float(a.std(ddof=1)) if len(a)>1 else 0.)


def paired(values):
    a = np.asarray(values,dtype=float)
    margin = t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0.
    return dict(mean=float(a.mean()),ci95=[float(a.mean()-margin),float(a.mean()+margin)],
                wins=int((a>1e-10).sum()),ties=int((abs(a)<=1e-10).sum()),losses=int((a< -1e-10).sum()))


def save_figure(fig, root, name):
    for ext in ("png","pdf","svg"):
        fig.savefig(root/f"{name}.{ext}",dpi=180,bbox_inches="tight")


def analyze(root):
    root = root.resolve()
    study = json.loads((root/"study.json").read_text())
    source = Path(study["source"])
    rows, local, propagated, heldout_local, benchmarks, calibration = [], [], [], [], [], []
    traffic = {}
    index_hashes = {}
    for dataset in study["datasets"]:
        for seed in study["seeds"]:
            folder = root/dataset/f"seed_{seed}"
            result = json.loads((folder/"result.json").read_text())
            assert not result["pilot"] and result["device"].startswith("cuda")
            assert result["optimizer_updates"] == result["test_evaluations"] == 0
            assert result["deterministic_algorithms"]
            assert result["model_sha256_before"]==result["model_sha256_after_calibration"]==result["model_sha256_after"]
            assert result["bypass_side_max_logit_difference"]==0
            assert result["source_hashes_before"]==result["source_hashes_after"]
            for name,digest in result["source_hashes_before"].items():
                assert sha(source/dataset/"none"/f"seed_{seed}"/name)==digest
            indices = json.loads((folder/"indices.json").read_text())
            assert len(indices["fit"])==len(indices["heldout"])==256
            assert not set(indices["fit"]) & set(indices["heldout"])
            index_hashes.setdefault(dataset,set()).add(sha(folder/"indices.json"))
            val = json.loads((folder/"validation.json").read_text())
            traffic[(dataset,seed)] = [dict(nodes=l["nodes"],edges=l["edges"]) for l in val["diagnostics"]["none"]]
            heldout = json.loads((folder/"heldout.json").read_text())
            for mode in MODES:
                assert val["metrics"][mode]["examples"]==len(indices["validation"])
            for context, values, dest in (("fixed",val["diagnostics"]["none"],local),
                                          ("heldout_fixed",heldout["diagnostics"]["none"],heldout_local)):
                for layer in values:
                    spectrum = layer["spectrum"]
                    assert spectrum["sampled_nodes"]<=4096
                    energy = np.array(spectrum["squared_singular_values"])
                    np.testing.assert_allclose(energy.cumsum()/max(energy.sum(),1e-30),spectrum["cumulative_energy"],atol=1e-10)
                    for mode,s in layer["modes"].items():
                        np.testing.assert_allclose(s["norm_ratio"]**2,1-s["mse_reduction"],atol=1e-10)
                        if "captured_sq" in s:
                            assert s["max_batch_identity_relative_gap"]<2e-5
                            assert s["compensation_condition"]==(s["after_sq"]<s["before_sq"])
                        dest.append(dict(dataset=dataset,seed=seed,layer=layer["layer"],mode=mode,
                                         context=context,spectrum=spectrum,**s))
            for mode,layers in val["diagnostics"].items():
                if mode=="none":
                    continue
                for layer in layers:
                    propagated.append(dict(dataset=dataset,seed=seed,layer=layer["layer"],mode=mode,
                                           **layer["modes"]["active"]))
            benchmarks.extend(dict(dataset=dataset,seed=seed,**b) for b in json.loads((folder/"benchmark.json").read_text()))
            calibration.extend(dict(dataset=dataset,seed=seed,**b) for b in json.loads((folder/"calibration.json").read_text()))
            rows.append(result)
    assert all(len(s)==1 for s in index_hashes.values())
    manifest = json.loads((root/"source_manifest.json").read_text())
    for name,digest in manifest.items():
        assert sha(root/"source_snapshot"/name)==digest
    tests = (root/"tests.log").read_text()
    assert "Ran 11 tests" in tests and "\nOK\n" in tests
    summary, comparisons = [], []
    for dataset in study["datasets"]:
        group = [r for r in rows if r["dataset"]==dataset]
        for mode in MODES:
            summary.append(dict(dataset=dataset, mode=mode,
                accuracy=average([r["validation"][mode]["accuracy"]*100 for r in group]),
                loss=average([r["validation"][mode]["loss"] for r in group]),
                heldout_accuracy=average([r["heldout"][mode]["accuracy"]*100 for r in group]),
                heldout_loss=average([r["heldout"][mode]["loss"] for r in group]),
                versus_none=paired([(r["validation"][mode]["accuracy"]-r["validation"]["none"]["accuracy"])*100 for r in group])))
        for ref in ("none","random","pca","higher_16","shuffle_graph"):
            comparisons.append(dict(dataset=dataset,reference=ref,
                accuracy_delta=paired([(r["validation"]["graph"]["accuracy"]-r["validation"][ref]["accuracy"])*100 for r in group]),
                ce_improvement=paired([r["validation"][ref]["loss"]-r["validation"]["graph"]["loss"] for r in group])))
    write_json(root/"results.json",rows)
    write_json(root/"analysis.json",dict(summary=summary,comparisons=comparisons,local=local,
        corrected_prefix=propagated,heldout_local=heldout_local,benchmarks=benchmarks))
    audit = dict(checkpoints=len(rows),source_artifact_hashes_verified=len(rows)*5,
        model_state_unchanged=len(rows),same_calibration_indices_across_seeds=True,
        test_evaluations=0,optimizer_updates=0,cuda_only=True,unit_tests=11,
        deterministic_bypass_exact_matches=len(rows),
        max_identity_relative_gap=max(r.get("max_batch_identity_relative_gap",0) for r in local+propagated),
        max_baseline_validation_ce_drift=max(abs(r["validation_reproduction"]["delta"]) for r in rows))
    write_json(root/"verification.json",audit)
    with (root/"per_seed.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["dataset","seed","mode","validation_accuracy","validation_loss","heldout_accuracy","heldout_loss"])
        writer.writeheader()
        for row in rows:
            for mode in MODES:
                writer.writerow(dict(dataset=row["dataset"],seed=row["seed"],mode=mode,
                    validation_accuracy=row["validation"][mode]["accuracy"],validation_loss=row["validation"][mode]["loss"],
                    heldout_accuracy=row["heldout"][mode]["accuracy"],heldout_loss=row["heldout"][mode]["loss"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    for dataset in study["datasets"]:
        group = [r for r in rows if r["dataset"]==dataset]
        modes = ["none","random","pca","graph","higher_16","shuffle_graph"]
        fig,axes=plt.subplots(1,2,figsize=(12,4.2))
        for ax,metric,scale,title in ((axes[0],"accuracy",100,"Validation accuracy (%)"),
                                      (axes[1],"loss",1,"Validation cross-entropy")):
            for i,mode in enumerate(modes):
                values=np.array([r["validation"][mode][metric]*scale for r in group])
                ax.bar(i,values.mean(),color=COLORS[mode],alpha=.65,yerr=values.std(ddof=1),capsize=3)
                ax.scatter(np.full(len(values),i)+np.linspace(-.15,.15,len(values)),values,color=COLORS[mode],s=10)
            ax.set_xticks(range(len(modes)),modes,rotation=25,ha="right")
            ax.set_ylabel(title)
        fig.suptitle(f"{dataset}: frozen checkpoints, 10 seeds (mean ± SD)")
        fig.tight_layout()
        save_figure(fig,root,f"{dataset}_validation")
        plt.close(fig)

        fig,axes=plt.subplots(2,4,figsize=(13,6))
        for i in range(4):
            values=[r for r in local if r["dataset"]==dataset and r["mode"]=="none" and r["layer"]==i+1]
            energies=np.array([r["spectrum"]["squared_singular_values"] for r in values])
            normalized=energies/energies.sum(1,keepdims=True)
            for ax,y in ((axes[0,i],normalized),(axes[1,i],normalized.cumsum(1))):
                x=np.arange(1,y.shape[1]+1)
                ax.plot(x,y.mean(0),color="#3683b8")
                ax.fill_between(x,y.min(0),y.max(0),alpha=.18,color="#3683b8")
                ax.axvline(2,color="gray",ls=":",lw=.8)
                ax.set_xlabel("Rank")
                ax.grid(alpha=.2)
            axes[0,i].set_yscale("log")
            axes[0,i].set_title(f"Layer {i+1}")
            axes[1,i].set_ylim(0,1.02)
        axes[0,0].set_ylabel("Normalized squared singular value")
        axes[1,0].set_ylabel("Cumulative energy")
        fig.suptitle(f"{dataset}: local propagated residual spectrum (≤4096 nodes/layer)")
        fig.tight_layout()
        save_figure(fig,root,f"{dataset}_error_spectrum")
        plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(11,4.2))
        modes=["random","pca","graph","higher_16","shuffle_graph"]
        for j,mode in enumerate(modes):
            reduction=[np.mean([r["mse_reduction"]*100 for r in local if r["dataset"]==dataset and r["mode"]==mode and r["layer"]==i+1]) for i in range(4)]
            axes[0].bar(np.arange(4)+(j-2)*.15,reduction,.15,label=mode,color=COLORS[mode])
        axes[0].axhline(0,color="black",lw=.7)
        axes[0].set_xticks(range(4),["L1","L2","L3","L4"])
        axes[0].set_ylabel("Local squared-error reduction (%)")
        axes[0].legend(fontsize=8)
        graph=[r for r in local if r["dataset"]==dataset and r["mode"]=="graph"]
        for key,label,color in (("truncation_sq","Rank truncation","#d76b27"),
                                ("side_quantization_sq","Side quantization","#3683b8")):
            values=[np.mean([r[key]/r["before_sq"] for r in graph if r["layer"]==i+1]) for i in range(4)]
            axes[1].plot(range(1,5),values,"o-",label=label,color=color)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Squared error / uncompensated error")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"{dataset}: fixed-input local compensation")
        fig.tight_layout()
        save_figure(fig,root,f"{dataset}_compensation")
        plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(11,4.2))
        for mode in ("none","higher_8","higher_16","higher_32","higher_64","higher_110","random","pca","graph"):
            budget=np.mean([np.mean(r["budgets"][mode]["extra_bits_per_node_per_layer"]) for r in group])
            acc=np.mean([r["validation"][mode]["accuracy"]*100 for r in group])
            reduction=np.mean([r["mse_reduction"]*100 for r in local if r["dataset"]==dataset and r["mode"]==mode])
            color=COLORS.get(mode,"#27906d")
            for column,(ax,y) in enumerate(((axes[0],acc),(axes[1],reduction))):
                ax.scatter(budget,y,color=color)
                offset = ({"graph":(12,23),"pca":(12,-24),"higher_16":(65,0)}.get(mode,(4,4))
                          if column==1 else (4,4))
                ax.annotate(mode,(budget,y),xytext=offset,textcoords="offset points",fontsize=7,
                            arrowprops=dict(arrowstyle="-",lw=.5,color=color) if offset!=(4,4) else None)
                ax.set_xlabel("Extra message bits / node / layer (mean)")
                ax.grid(alpha=.2)
        axes[0].set_ylabel("Validation accuracy (%)")
        axes[1].set_ylabel("Mean local squared-error reduction (%)")
        fig.suptitle(f"{dataset}: message budget tradeoff; residual bases add 3552 static bytes")
        fig.tight_layout()
        save_figure(fig,root,f"{dataset}_budget")
        plt.close(fig)

    report=["# 真实量化残差通道：第一阶段结果", "",
        "已完成 MNIST 和 CIFAR10 各 10 个冻结检查点（种子 41–50）的 GPU 校准和完整验证，共 20 个检查点、每个 13 种前向条件。没有参数更新，没有新增测试集评估。", "",
        "## 结论", ""]
    for dataset in study["datasets"]:
        comp=next(c for c in comparisons if c["dataset"]==dataset and c["reference"]=="none")
        graph=[r for r in local if r["dataset"]==dataset and r["mode"]=="graph"]
        beats_pca=sum(r["after_sq"]<next(p["after_sq"] for p in local if p["dataset"]==dataset and p["mode"]=="pca" and p["seed"]==r["seed"] and p["layer"]==r["layer"]) for r in graph)
        beats_bits=sum(r["after_sq"]<next(p["after_sq"] for p in local if p["dataset"]==dataset and p["mode"]=="higher_16" and p["seed"]==r["seed"] and p["layer"]==r["layer"]) for r in graph)
        d=comp["accuracy_delta"]
        report.append(f"- **{dataset}**：图基通道在 {sum(r['mse_reduction']>0 for r in graph)}/40 个种子×层组合降低局部误差；"
            f"优于普通残差 PCA 的有 {beats_pca}/40 个，优于增加 ≤16 个主消息位的有 {beats_bits}/40 个。"
            f"完整验证准确率相对无补偿改变 {d['mean']:+.3f} 个百分点，配对 95% 区间 [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]。")
    report.extend(["", "局部补偿、图基价值、同预算效率与分类收益是四个独立问题。图基在两数据集全部 20 个种子的准确率均低于无补偿模型。一种可能解释是原 QAT 训练让模型适应了量化扰动，但本轮没有验证这一因果解释；局部还原误差与分类目标也可能不一致。本阶段没有自动进入联合 QAT。", "",
        "## 验证准确率与损失", "", "准确率为均值 ± 样本标准差（%）；差值按同一训练种子配对，单位为百分点。区间为 Student t 95% 区间，未做多重比较校正。这里的验证划分曾用于原检查点选模，并非新的独立测试集。", "",
        "| 数据集 | 条件 | 验证准确率 | 验证 CE | 相对无补偿 [95% 区间] | 胜/平/负 |",
        "|---|---|---:|---:|---:|---:|"])
    for s in summary:
        a,l,d=s["accuracy"],s["loss"],s["versus_none"]
        report.append(f"| {s['dataset']} | {LABELS[s['mode']]} | {a['mean']:.3f} ± {a['std']:.3f} | {l['mean']:.5f} | {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {d['wins']}/{d['ties']}/{d['losses']} |")
    report.extend(["", "## 图基与关键控制的配对比较", "", "CE 改善 = 控制 CE − 图基 CE，正值有利于图基。", "",
        "| 数据集 | 对照 | 准确率差 [95% 区间] | CE 改善 [95% 区间] |", "|---|---|---:|---:|"])
    for c in comparisons:
        a,l=c["accuracy_delta"],c["ce_improvement"]
        report.append(f"| {c['dataset']} | {LABELS[c['reference']]} | {a['mean']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | {l['mean']:+.5f} [{l['ci95'][0]:+.5f}, {l['ci95'][1]:+.5f}] |")
    report.extend(["", "## 固定输入的局部误差", "",
        "在无补偿模型每层的同一输入 X 上比较。R=X−Q(X)，T=A+(1+epsilon)I；只绕过当前消息量化，保留上游计算与权重。以下是 10 个种子的均值，平方误差下降为 1−补偿后平方误差/补偿前平方误差。它不是完整 FP32 模型误差。", "",
        "| 数据集 | 层 | 条件 | 平方误差下降 % | 误差范数比 | 余弦 | 改善/10 |", "|---|---|---|---:|---:|---:|---:|"])
    for dataset in study["datasets"]:
        for layer in range(1,5):
            for mode in ("random","pca","graph","higher_16","float_graph","shuffle_graph"):
                g=[r for r in local if r["dataset"]==dataset and r["layer"]==layer and r["mode"]==mode]
                report.append(f"| {dataset} | {layer} | {LABELS[mode]} | {np.mean([r['mse_reduction']*100 for r in g]):.3f} | {np.mean([r['norm_ratio'] for r in g]):.4f} | {np.mean([r['cosine'] for r in g]):.4f} | {sum(r['mse_reduction']>0 for r in g)} |")
    report.extend(["", "图内打乱保持每图代码多重集和消息预算，但破坏代码与发送节点的对应关系。打乱后仍有部分改善可能来自图内共同残差，不能单凭打乱控制声称所有收益都依赖节点级信息。", "",
        "## 误差分解与所有层启用后的行为", "",
        "正交基满足 ||TX−U||² = ||TR(I−BBᵀ)||² + ||T eta||²，eta=Z−RB。前项是 rank 截断，后项是残差编码量化误差；两者均计入总误差。", "",
        "| 数据集 | 层 | 截断/原误差 % | 残差量化/原误差 % | 饱和值 % | 四层启用时本层下降 % | 训练留出局部下降 % |",
        "|---|---|---:|---:|---:|---:|---:|"])
    for dataset in study["datasets"]:
        for layer in range(1,5):
            g=[r for r in local if r["dataset"]==dataset and r["layer"]==layer and r["mode"]=="graph"]
            p=[r for r in propagated if r["dataset"]==dataset and r["layer"]==layer and r["mode"]=="graph"]
            h=[r for r in heldout_local if r["dataset"]==dataset and r["layer"]==layer and r["mode"]=="graph"]
            report.append(f"| {dataset} | {layer} | {np.mean([r['truncation_sq']/r['before_sq']*100 for r in g]):.4f} | {np.mean([r['side_quantization_sq']/r['before_sq']*100 for r in g]):.4f} | {np.mean([r['saturation_fraction']*100 for r in g]):.4f} | {np.mean([r['mse_reduction']*100 for r in p]):.3f} | {np.mean([r['mse_reduction']*100 for r in h]):.3f} |")
    report.extend(["", f"全部数值分解最大相对偏差 {audit['max_identity_relative_gap']:.3g}。20 个检查点旁路消息量化后的有/无残差通道输出逐值完全相同。开启所有层时，第 2–4 层输入会随前面层的修正变化，因此另存完整前向中的局部指标，不能直接当成固定基线输入的同一比较。原始 JSON 的 saturated_values 统计投影绝对值超出 INT8 校准范围的数量；float_graph 中仅表示潜在 INT8 溢出，浮点参考实际不截断。", "",
        "## 预算、存储与参考执行开销", "",
        "基础消息每节点每层 440 位。rank 2 INT8 通道增加 16 位（3.636%），每层基 880 字节＋尺度 8 字节，四层 3552 字节。固定基没有训练参数，但有静态存储成本。浮点残差码参考增加 64 位。提高主消息位宽对照复用已有位宽/尺度槽位，实际选择数如下；0 位对照即无补偿。", "",
        "| 数据集 | 条件 | 平均额外位/节点/层 | 四层额外静态张量字节 |", "|---|---|---:|---:|"])
    for dataset in study["datasets"]:
        group=[r for r in rows if r["dataset"]==dataset]
        for mode in ("none","graph","higher_8","higher_16","higher_32","higher_64","higher_110","float_graph"):
            report.append(f"| {dataset} | {LABELS[mode]} | {np.mean([np.mean(r['budgets'][mode]['extra_bits_per_node_per_layer']) for r in group]):.3f} | {group[0]['budgets'][mode]['extra_static_tensor_bytes']} |")
    report.extend(["", "代码和逻辑低位宽消息预算可复核；本实现仍使用原 fake-quant 主消息计算，没有原生 INT4 kernel。侧码是实际 INT8 张量，邻居累加为 INT32，实数自身系数在累加后处理。基线与残差方案总存储不相同。峰值显存是含所有控制校准对象的研究程序实测值，不是压缩部署模型大小。校准索引/源码/文件容器开销另计，详见 artifacts.json。", "",
        "| 数据集 | 原模型状态张量字节 | 加入图基后状态张量字节 |", "|---|---:|---:|"])
    for dataset in study["datasets"]:
        base=next(r["base_state_tensor_bytes"] for r in rows if r["dataset"]==dataset)
        report.append(f"| {dataset} | {base} | {base+3552} |")
    report.extend(["", "上述静态表按现有 PyTorch 张量实际 dtype 计，包括骨干、输入提示、量化器参数及 BN 缓冲区，不含 Python 对象；主权重并未打包为原生 INT4 文件。消息预算另按节点存储量和逐边逻辑载荷计入 artifacts.json，均非实测网络流量。额外计算约为每节点 440 次乘加（110×2 编码＋2×110 解码），另有每条边 2 个整数累加。", "",
        "| 数据集 | batch | 条件 | 前向中位数的种子均值 ms | 总峰值显存 MiB | 前向新增峰值 MiB |", "|---|---:|---|---:|---:|---:|"])
    for dataset in study["datasets"]:
        for batch in (32,128):
            for mode in ("none","graph","higher_16"):
                b=[r for r in benchmarks if r["dataset"]==dataset and r["batch_size"]==batch and r["mode"]==mode]
                report.append(f"| {dataset} | {batch} | {LABELS[mode]} | {np.mean([r['seconds_median']*1000 for r in b]):.3f} | {np.mean([r['peak_cuda_allocated_bytes']/2**20 for r in b]):.2f} | {np.mean([r['incremental_peak_cuda_bytes']/2**20 for r in b]):.2f} |")
    report.extend(["", "计时关闭诊断，3 次预热、10 次 GPU 同步测量，固定验证集首批；四张 RTX 4090 各一个工作进程。未执行融合或调优，不能把参考计时解释为整数部署加速比。", "",
        "## 校准、复现与验收", "",
        "两数据集在官方训练集固定分层抽样 512 图（种子 912041），再用种子 912042 划为 256 拟合/256 留出图；全部训练种子和控制共用图索引。每层最多 4096 个相同节点拟合未中心化谱基，尺度取全部拟合图投影绝对最大值/127。标签只用于固定抽样和指标，不进入残差通道或基拟合。每层独立使用冻结基线输入校准，没有顺序重校准或验证调参。", "",
        f"11 项 GPU unittest 全部通过，包括零残差、孤立节点、自身项、重边/自环/边顺序、INT8 饱和与 INT32 累加、正交性与误差恒等式、图内打乱、同预算选择、固定节点抽样、冻结模型以及诊断开关一致性。验证原检查点 CE 的最大复算偏差为 {audit['max_baseline_validation_ce_drift']:.3g}；原有 5e-4 复算界限保持不变。", "",
        "前后核对 20 个模型全部状态（含 BatchNorm），以及 100 个原始文件哈希（best、last、结果、测试开始标记、配置）。本轮测试集前向为 0，优化器更新为 0。固定基消除了新增标签训练容量这一混杂，但仍增加了残差信息和静态存储，故必须保留同预算对照。", "",
        "初版完整验证暴露了 CUDA 浮点 scatter 的非确定性：局部残差为零仍可能因独立前向的累加顺序差异而改变下游量化舍入。初版目录 residual_channel_20260912_v1 和计时试验目录均保留；正式 v2 使用确定性算法及 CUBLAS_WORKSPACE_CONFIG=:4096:8，全 20 个检查点按同一协议重跑，没有放宽旁路容差，也未改动原模型。", "",
        "## 图表", ""])
    for dataset in study["datasets"]:
        for name,label in (("validation","验证表现"),("error_spectrum","局部误差谱，阴影为种子范围"),("compensation","局部补偿与误差分解"),("budget","额外消息位预算曲线")):
            report.extend([f"![{dataset} {label}]({root}/{dataset}_{name}.png)",""])
    report.extend(["## 后续边界", "",
        "本阶段已完成机制与预算可行性检验。原 LoRAP 及邻域统计方案 A 继续保留，未启动其 150 次训练。结果支持到什么程度应由上面的四类比较分别判断；尚不能宣称图基机制已在同预算下稳定改善分类性能。若继续，先讨论允许的任务适配与联合 QAT 范围，不在本轮自动增加训练。", "",
        "完整指标见 per_seed.csv、analysis.json、每种子的 heldout.json/validation.json。误差谱、校准基和尺度、图及节点索引、参考计时、哈希验证均保留。运行命令见 RUN_COMMANDS.md。"])
    (root/"REPORT.md").write_text("\n".join(report)+"\n")
    artifacts=[]
    for row in rows:
        directory=root/row["dataset"]/f"seed_{row['seed']}"
        artifacts.append(dict(dataset=row["dataset"],seed=row["seed"],
            base_state_tensor_bytes=row["base_state_tensor_bytes"],
            calibration_container_bytes=(directory/"calibration.pt").stat().st_size,
            index_json_bytes=(directory/"indices.json").stat().st_size,
            per_layer_budget=row["budgets"],
            logical_payload_over_full_validation={mode:[dict(
                nodes=counts["nodes"],edges=counts["edges"],
                node_storage_bytes=counts["nodes"]*(440+budget["extra_bits_per_node_per_layer"][i])/8,
                edge_payload_bytes=counts["edges"]*(440+budget["extra_bits_per_node_per_layer"][i])/8)
                for i,counts in enumerate(traffic[(row["dataset"],row["seed"])] )]
                for mode,budget in row["budgets"].items()}))
    write_json(root/"artifacts.json",artifacts)
    commands=f"""# 第一阶段 GPU 实验复现

正式实验使用冻结源码。以下命令只补齐未完成检查点，不会重新测试或训练。

```sh
cd {root}/source_snapshot
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.residual_channel.suite --root {root} --source {source} --gpus 0 1 2 3
```

单个检查点（使用新的输出目录可独立复现）：

```sh
cd {root}/source_snapshot
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.residual_channel.experiment --dataset CIFAR10 --seed 41 --source {source} --root /home/cyliu/code/A-2Q/result/lorap_study/residual_channel_reproduction
```

只读取结果，重新生成报告：

```sh
cd {root}/analysis_snapshot
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.residual_channel.report --root {root}
```

全部模型计算必须能访问 CUDA；没有 CPU 回退。报告绘图只读文件，不执行模型。
本阶段没有新增完整训练命令，因为范围明确为冻结模型校准与验证；先前完整训练命令保留于原实验目录。
"""
    (root/"RUN_COMMANDS.md").write_text(commands)
    print(json.dumps(dict(verification=audit,comparisons=comparisons),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    analyze(parser.parse_args().root)
