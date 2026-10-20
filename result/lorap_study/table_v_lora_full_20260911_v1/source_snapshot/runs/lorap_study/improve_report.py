"""Report locked validation screening and equal-budget layer rank comparisons."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run import ROOT, write_json

LABELS = {"original": "原 LoRAP", "conditioned": "首轮条件 LoRAP", "none": "无提示",
          "zero_init": "条件提示＋零初始化", "stable_random": "标准化条件＋随机正交基",
          "stable_spectral": "标准化条件＋误差谱基（均匀 rank）",
          "adaptive_spectral": "标准化条件＋误差谱基＋分层 rank"}


def stats(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.)


def fmt(values, scale=1., digits=2):
    value = stats(np.asarray(values) * scale)
    return f"{value['mean']:.{digits}f} ± {value['std']:.{digits}f}"


def historical_cite(seeds, epochs):
    roots = dict(none=ROOT / "result/lorap_study/mechanism_20260909_v1/qat/none",
                 original=ROOT / "result/lorap_study/pilot_20260909_v1/CiteSeer/original",
                 conditioned=ROOT / "result/lorap_study/pilot_20260909_v1/CiteSeer/conditioned")
    records = []
    for variant, root in roots.items():
        for seed in seeds:
            path = root / f"seed_{seed}/result.json"
            if path.exists():
                r = json.loads(path.read_text())
                if r["epochs"] == epochs:
                    records.append(dict(r, variant=variant, historical_source=str(path),
                                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return records


def merged_error(rows, kind="full_update"):
    before = sum(r[kind]["before_frobenius"]**2 for r in rows)
    after = sum(r[kind]["after_frobenius"]**2 for r in rows)
    p2 = sum(r[kind]["prompt_norm"]**2 for r in rows)
    dot = sum((r[kind]["cosine"] or 0) * r[kind]["before_frobenius"] * r[kind]["prompt_norm"] for r in rows)
    return dict(reduction=1 - np.sqrt(after / before) if before else None,
                cosine=dot / np.sqrt(before*p2) if before and p2 else None)


def figure(fig, output, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def generate_report(output):
    output = Path(output)
    results = json.loads((output / "results.json").read_text())
    completed = [r for r in results if r["test"] is not None]
    cite = [r for r in completed if r["dataset"] == "CiteSeer"]
    historical = historical_cite([r["seed"] for r in cite], cite[0]["epochs"]) if cite else []
    # A historical baseline is summarized only when it covers all paired seeds.
    historical = [r for r in historical if sum(x["variant"] == r["variant"] for x in historical) == len(cite)]
    historical = [r for r in historical if r["variant"] not in {x["variant"] for x in cite}]
    for r in historical:
        peer = next(x for x in cite if x["seed"] == r["seed"])
        assert r["backbone_sha256"] == peer["backbone_sha256"]
    write_json(output / "historical_comparators.json", historical)
    flat = []
    for r in results:
        flat.append(dict(dataset=r["dataset"], variant=r["variant"], seed=r["seed"],
                         epochs=r["epochs"], ranks=str(r["ranks"]), best_epoch=r["best_epoch"],
                         parameters=r["aggregate_prompt_parameters"], validation_loss=r["validation"]["loss"],
                         validation_accuracy=r["validation"]["accuracy"],
                         test_accuracy=r["test"]["accuracy"] if r["test"] is not None else None,
                         test_loss=r["test"]["loss"] if r["test"] is not None else None,
                         test_evaluations=r["test_evaluations"], effective_bits=r["mean_activation_effective_bits"]))
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    report = ["# LoRAP 条件化、初始化与分层容量改进", "",
              "本轮实现三项改动，并使用独立输出目录保留全部正负结果。CiteSeer 在首个种子的验证集筛选候选，"
              "锁定后训练其余种子，再查看新测试结果；MNIST 使用官方划分内 512/256/256 图的小样本对照。", "",
              "## 实现", "",
              "1. **零输出初始化**：A_l=0，B_l 非零。初始提示严格为零，保留无聚合提示模型的输出与骨干梯度；"
              "第一步 A_l 可学习，其他提示参数在随后的步骤获得非零梯度，避免两个因子都置零。",
              "2. **标准化量化条件**：保留入度 log1p、平均 log 步长、位宽/8、邻居饱和率四项。"
              "仅用训练样本统计各层均值和标准差，标准差下限为 [0.5,1,0.125,0.1]，标准化结果截断在 ±5。"
              "查询特征按逐节点 RMS 归一化，提示输出按停止梯度的 RMS 缩放，以匹配当前聚合特征尺度。",
              "3. **误差谱基与分层 rank**：校准误差包含 GIN 自身节点项，"
              "E_l=(1+ε_l)(H_l−Q_l(H_l))+A(H_l)−A(Q_l(H_l))。B_l 初始化为训练校准误差的前 r_l 个右奇异向量，之后继续训练。"
              "MNIST 四层分别拥有低秩因子，共享打分层、条件投影和门控。总 rank 固定为 8，至少每层 1，"
              "逐次给归一化剩余误差能量最大的层分配一个 rank。训练开始后分配固定。", "",
              "稳定化提示公式为：", "",
              "```text",
              "u_l = sqrt(max(mean(S_l²), 1e-12))",
              "z_l = clip((stopgrad(c_l) - μ_l) / σ_l, -5, 5)",
              "p_l = max(stopgrad(u_l), τ_l) × 2 sigmoid(g(z_l))",
              "      × softmax(a(S_l / u_l) + Bc z_l) × A_l B_l",
              "output_l = S_l + p_l",
              "```", "",
              "τ_l 是训练校准聚合 RMS 均值的 0.1 倍，下限 1e-4；额外固定残差系数为 1。"
              "推理只使用量化聚合、量化器参数和训练时保存的统计量，不运行未量化参考或 SVD。"
              "离线校准最多取 4096 个训练节点/层；CiteSeer 只统计 120 个训练 mask 节点，消息传递沿用完整公共图；"
              "MNIST 仅使用固定训练子集的前 128 张图，不读验证/测试图。", "",
              "分配准则为下一奇异值平方除以该层参考更新输入的总平方范数。"
              "归一化用于避免特征幅度较大的层自动占满预算；它是误差能量代理，并非分类敏感度证明。"
              "同时保存去均值误差谱，便于检查低秩现象是否主要来自均值。", "",
              "CiteSeer 仅第二层有消息量化聚合提示，因此保留 rank 2，不增加第一层提示。"
              "CiteSeer 三个候选均为 497 个参数，与首轮条件 LoRAP 相同。"
              "MNIST 均匀/自适应分层版本均为 3345 个聚合提示参数；原共享 LoRAP 为 2480 个。"
              "所以分层版本与原版的比较包含容量变化，而均匀/自适应之间的比较严格匹配容量。输入 GPF-plus 均保留。", "",
              "训练沿用此前量化器、骨干、学习率、存储惩罚、配对初始化及随机数据顺序。"
              "提示参数使用主学习率、零权重衰减；按最低验证 CE 选模，平局保留较早轮次。"
              "此次没有修改旧量化器，也没有新增辅助损失。稳定化候选同时包含条件、查询和输出尺度改动，"
              "本轮不能将它的变化归因到其中单一组件。", ""]
    report += ["本轮沿用历史方案，聚合消息位宽参数冻结。因此位宽条件在同一层内为常量，"
               "不能据此声称模型已经学会跨位宽泛化。谱校准发生在初始骨干上，"
               "训练中表示和误差谱可能改变；一次校准的容量分配不是动态最优分配。", ""]
    selection_path = output / "CiteSeer/selection.json"
    if selection_path.exists():
        choice = json.loads(selection_path.read_text())
        report += ["## CiteSeer 验证集筛选", "",
                   f"仅使用种子 {choice['discovery_seed']} 的最低验证 CE 选择，锁定候选为 **{LABELS[choice['selected_variant']]}**。"
                   "其余候选不计算测试结果。首个种子参与设计选择，其余种子用于检查能否复现；"
                   "仍是开发阶段的小规模实验，不构成独立确认性检验。", "",
                   "| 候选 | 参数 | 最佳轮次 | 验证 CE | 验证准确率 |",
                   "|---|---:|---:|---:|---:|"]
        for r in results:
            if r["dataset"] == "CiteSeer" and r["seed"] == choice["discovery_seed"]:
                report.append(f"| {LABELS[r['variant']]} | {r['aggregate_prompt_parameters']} | {r['best_epoch']} | "
                              f"{r['validation']['loss']:.4f} | {100*r['validation']['accuracy']:.2f}% |")
    report += ["", "## 完成测试的对照", "", "均值 ± 样本标准差；准确率单位为 %。历史比较使用相同种子、轮数和初始骨干。", "",
               "| 数据集 | 方法 | 种子数 | 轮数 | 聚合提示参数 | 验证 CE | 测试 CE | 测试准确率 |",
               "|---|---|---:|---:|---:|---:|---:|---:|"]
    compared = historical + completed
    grouped = {}
    for r in compared:
        grouped.setdefault((r["dataset"], r["variant"]), []).append(r)
    for (dataset, variant), rows in grouped.items():
        report.append(f"| {dataset} | {LABELS[variant]} | {len(rows)} | {rows[0]['epochs']} | "
                      f"{rows[0]['aggregate_prompt_parameters']} | {fmt([r['validation']['loss'] for r in rows], digits=4)} | "
                      f"{fmt([r['test']['loss'] for r in rows], digits=4)} | {fmt([r['test']['accuracy'] for r in rows], 100)} |")
    for dataset in sorted({r["dataset"] for r in completed}):
        groups = {variant: rows for (name, variant), rows in grouped.items() if name == dataset}
        fig, ax = plt.subplots(figsize=(max(7, len(groups)*1.6), 3.8))
        for index, (variant, rows) in enumerate(groups.items()):
            values = [r["test"]["accuracy"]*100 for r in rows]
            value = stats(values)
            ax.bar(index, value["mean"], yerr=value["std"], capsize=4,
                   color="#4477aa" if variant in ("none", "original", "conditioned") else "#228833")
            ax.scatter([index]*len(values), values, color="black", s=16, zorder=4)
        ax.set_xticks(range(len(groups)), [name.replace("_", "\n") for name in groups])
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(dataset + (" · 512/256/256 graph subset" if dataset == "MNIST" else " · public split"))
        ax.spines[["right", "top"]].set_visible(False)
        fig.tight_layout()
        figure(fig, output, f"{dataset}_accuracy")
        report += ["", f"![{dataset} 测试准确率]({dataset}_accuracy.png)"]
    paired = []
    for r in completed:
        baseline = [x for x in compared if x["dataset"] == r["dataset"] and x["seed"] == r["seed"] and x["variant"] == "original"]
        if baseline and r["variant"] != "original":
            base = baseline[0]
            paired.append(dict(dataset=r["dataset"], variant=r["variant"], seed=r["seed"],
                               accuracy_gain_pp=100*(r["test"]["accuracy"]-base["test"]["accuracy"]),
                               validation_ce_reduction=base["validation"]["loss"]-r["validation"]["loss"],
                               test_ce_reduction=base["test"]["loss"]-r["test"]["loss"]))
    write_json(output / "paired_effects.json", paired)
    rank_effects = []
    for r in completed:
        if r["dataset"] == "MNIST" and r["variant"] == "adaptive_spectral":
            peers = [x for x in completed if x["dataset"] == "MNIST" and x["seed"] == r["seed"]
                     and x["variant"] == "stable_spectral"]
            if peers:
                base = peers[0]
                rank_effects.append(dict(seed=r["seed"],
                    accuracy_gain_pp=100*(r["test"]["accuracy"]-base["test"]["accuracy"]),
                    validation_ce_reduction=base["validation"]["loss"]-r["validation"]["loss"],
                    test_ce_reduction=base["test"]["loss"]-r["test"]["loss"]))
    write_json(output / "rank_allocation_effects.json", rank_effects)
    if rank_effects:
        report += ["", "相同 3345 参数下，自适应 rank 相对均匀 rank 的配对测试准确率变化为 **" +
                   fmt([r["accuracy_gain_pp"] for r in rank_effects]) + " 个百分点**；验证 CE 降低量为 " +
                   fmt([r["validation_ce_reduction"] for r in rank_effects], digits=4) +
                   "，测试 CE 降低量为 " + fmt([r["test_ce_reduction"] for r in rank_effects], digits=4) + "。"]
    report += ["", "## 分层 rank 与校准误差谱", "",
               "下表的谱来自训练前校准，不是测试集，也不使用测试准确率分配 rank。"
               "每层捕获比例是所分 rank 对该层校准误差能量的累计解释比例。", "",
               "| 数据集 | seed | 方法 | 各层 rank | 各层校准误差能量捕获比例 |",
               "|---|---:|---|---|---|"]
    for r in completed:
        if r["variant"] not in ("stable_spectral", "adaptive_spectral"):
            continue
        infos = json.loads((output / r["dataset"] / f"calibration_seed_{r['seed']}/calibration.json").read_text())
        capture = []
        for info, rank in zip(infos, r["ranks"]):
            energy = np.square(info["singular_values"])
            capture.append(f"{100*sum(energy[:rank])/sum(energy):.2f}%" if sum(energy) else "undefined")
        report.append(f"| {r['dataset']} | {r['seed']} | {LABELS[r['variant']]} | {r['ranks']} | {', '.join(capture)} |")
    mnist = [r for r in completed if r["dataset"] == "MNIST" and r["variant"] == "adaptive_spectral"]
    if mnist:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        rank_matrix = np.array([r["ranks"] for r in mnist])
        for layer in range(4):
            axes[0].bar(range(len(mnist)), rank_matrix[:, layer], bottom=rank_matrix[:, :layer].sum(1), label=f"Layer {layer+1}")
        axes[0].set_xticks(range(len(mnist)), [str(r["seed"]) for r in mnist])
        axes[0].set_xlabel("Seed")
        axes[0].set_ylabel("Allocated rank (total = 8)")
        axes[0].legend(frameon=False, fontsize=8)
        infos = json.loads((output / "MNIST" / f"calibration_seed_{mnist[0]['seed']}/calibration.json").read_text())
        for info in infos:
            energy = np.square(info["singular_values"])
            curve = np.cumsum(energy)/sum(energy) if sum(energy) else np.zeros_like(energy)
            axes[1].plot(range(1, min(8, len(curve))+1), curve[:8]*100, marker="o", label=info["layer"])
        axes[1].set_xlabel("Rank")
        axes[1].set_ylabel("Training calibration error energy (%)")
        axes[1].legend(frameon=False, fontsize=8)
        axes[1].set_title(f"Initial backbone · seed {mnist[0]['seed']}")
        fig.tight_layout()
        figure(fig, output, "MNIST_rank_allocation")
        report += ["", "![分层 rank 与训练校准谱](MNIST_rank_allocation.png)"]
    report += ["", "## 验证集局部量化误差", "",
               "固定当前层输入和权重，只绕过该层消息量化。完整更新误差包含 GIN 自身节点项；"
               "各方法上游表示不同，因此不是共同 FP32 教师误差。降低率为 1−‖E−p‖F/‖E‖F，正值表示降低。"
               "多批次先合并平方范数和内积，避免直接平均批次百分比。", "",
               "| 数据集 | 方法 | seed | 层 | 邻居误差降低（%） | 完整更新误差降低（%） | 完整误差余弦 |",
               "|---|---|---:|---|---:|---:|---:|"]
    for r in completed:
        path = output / r["dataset"] / r["variant"] / f"seed_{r['seed']}/full_update_diagnostics.json"
        layers = json.loads(path.read_text())
        for name, batches in layers.items():
            neighbor, full = merged_error(batches, "neighbor"), merged_error(batches)
            show = lambda value, scale=1: f"{value*scale:.5f}" if value is not None else "undefined"
            report.append(f"| {r['dataset']} | {r['variant']} | {r['seed']} | {name} | "
                          f"{show(neighbor['reduction'],100)} | {show(full['reduction'],100)} | {show(full['cosine'])} |")
    report += ["", "误差范数不衡量分类重要性；准确率提升不等于量化补偿机制得到证实。"
               "旧预算代理与实际位宽仍保存于每个结果和 quantization.json，同预算目标不能解释为相同物理存储。", "",
               "## 逐种子结果", "",
               "| 数据集 | 方法 | seed | 最佳轮次 | 验证 CE | 验证准确率（%） | 测试准确率（%） | 实际平均位宽 |",
               "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        test = f"{100*r['test']['accuracy']:.2f}" if r["test"] is not None else "未评估（未入选）"
        report.append(f"| {r['dataset']} | {r['variant']} | {r['seed']} | {r['best_epoch']} | "
                      f"{r['validation']['loss']:.4f} | {100*r['validation']['accuracy']:.2f} | {test} | {r['mean_activation_effective_bits']:.4f} |")
    report += ["", "## 复现与完整训练", "",
               "MNIST 的 30 轮仍是小样本开发实验，不能与先前 2 轮功能验证直接比较，也不是完整 MNIST 基准。"
               "本轮包含相同数据、轮数和种子的原 LoRAP 对照。",
               "以下命令从仓库根目录运行，输出目录必须不存在；完整数据 200 轮命令仅生成，本轮未执行。", "", "```sh",
               "PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m unittest runs.lorap_study.test_study runs.lorap_study.test_mechanism runs.lorap_study.test_improved -v",
               "PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.improve_run --output-dir result/lorap_study/improvement_repeat --threads 4"]
    if cite:
        report += [f"PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.improve_run --dataset CiteSeer --variant {cite[0]['variant']} --full-data --epochs 200 --seeds 41 42 43 44 45 46 47 48 49 50 --output-dir result/lorap_study/improved_cite_full --threads 4"]
    report += ["PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.improve_run --dataset MNIST --variant adaptive_spectral --full-data --epochs 200 --seeds 41 42 43 44 45 46 47 48 49 50 --output-dir result/lorap_study/improved_mnist_full --threads 4",
               "# 容量匹配的完整 MNIST 对照：将上一条的 variant 改为 stable_spectral，并使用另一个输出目录。",
               "```", "",
               "结果目录包含环境与代码快照、训练校准统计和基向量、逐轮指标、最低验证 CE 检查点、候选选择记录、"
               "每种子的最终测试、局部误差诊断、实际位宽、CSV、图和完整训练配置。"]
    findings = output / "FINDINGS.md"
    if findings.exists():
        report[4:4] = [findings.read_text().replace("# 当前结论", "## 当前结论", 1), ""]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    source = Path(__file__)
    shutil.copy2(source, output / "improve_report_snapshot.py")
    write_json(output / "report_provenance.json", dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               findings_sha256=hashlib.sha256(findings.read_bytes()).hexdigest() if findings.exists() else None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate_report(parser.parse_args().output)
