"""Render descriptive mechanism results, keeping accuracy and loss evidence separate."""

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

from .mechanism import CONTROLS, INTERVENTIONS, PRECISIONS
from .run import write_json


LABELS = dict(none="无提示", original="LoRAP", mlp_exact="等参数 MLP")
MODES = dict(learned="原提示", off="关闭提示", signflip="反转符号", shuffle="打乱节点",
             mean="所有节点用提示均值", parallel="仅保留与完整修正平行的分量（oracle）",
             orthogonal="仅保留正交分量（oracle）")


def mean_std(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                values=list(map(float, values)))


def fmt(values, scale=1., digits=2):
    stats = mean_std(np.asarray(values) * scale)
    return f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f}"


def paired_effects(results):
    lookup = {(r["precision"], r["control"], r["seed"]): r for r in results}
    seeds = sorted({r["seed"] for r in results})
    effects = []
    for control in ("original", "mlp_exact"):
        for split in ("validation", "test"):
            for seed in seeds:
                row = dict(control=control, split=split, seed=seed)
                for precision in PRECISIONS:
                    none, extra = (lookup[precision, c, seed][split] for c in ("none", control))
                    row[precision + "_accuracy_gain_pp"] = 100 * (extra["accuracy"] - none["accuracy"])
                    row[precision + "_loss_gain"] = none["loss"] - extra["loss"]
                row["interaction_accuracy_pp"] = row["qat_accuracy_gain_pp"] - row["fp32_accuracy_gain_pp"]
                row["interaction_loss"] = row["qat_loss_gain"] - row["fp32_loss_gain"]
                effects.append(row)
    return effects


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_report(output):
    output = Path(output)
    results = json.loads((output / "results.json").read_text())
    probes = json.loads((output / "interventions.json").read_text())
    effects = paired_effects(results)
    seeds = sorted({r["seed"] for r in results})
    assert len(results) == 6 * len(seeds)
    write_json(output / "paired_effects.json", effects)
    write_csv(output / "paired_effects.csv", effects)
    flat = [dict(precision=r["precision"], control=r["control"], seed=r["seed"],
                 best_epoch=r["best_epoch"], validation_loss=r["validation"]["loss"],
                 validation_accuracy=r["validation"]["accuracy"], test_loss=r["test"]["loss"],
                 test_accuracy=r["test"]["accuracy"], parameters=r["aggregate_prompt_parameters"],
                 trainable_parameters=r["trainable_parameters"],
                 actual_mean_bits=r["mean_activation_effective_bits"],
                 reused="reused_from" in r) for r in results]
    write_csv(output / "factorial.csv", flat)
    probe_flat = []
    for r in probes:
        row = dict(precision=r["precision"], seed=r["seed"], intervention=r["intervention"],
                   strength=r["strength"], validation_accuracy=r["validation"]["accuracy"],
                   validation_loss=r["validation"]["loss"])
        for kind in ("neighbor", "full_update"):
            metrics = r.get("geometry", {}).get(kind, {})
            for field in ("norm_reduction", "cosine", "prompt_to_error_norm"):
                row[kind + "_" + field] = metrics.get(field)
        probe_flat.append(row)
    write_csv(output / "interventions.csv", probe_flat)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, split in zip(axes, ("validation", "test")):
        for offset, precision, color in ((-.18, "qat", "#4477aa"), (.18, "fp32", "#ee7733")):
            groups = [[100*r[split]["accuracy"] for r in results
                       if r["precision"] == precision and r["control"] == c] for c in CONTROLS]
            x = np.arange(3) + offset
            ax.bar(x, [np.mean(g) for g in groups], width=.34,
                   yerr=[np.std(g, ddof=1) for g in groups], capsize=3, label=precision.upper(), color=color)
            for pos, group in zip(x, groups):
                ax.scatter([pos]*len(group), group, color="black", s=13, zorder=5)
        ax.set_xticks(range(3), ["No prompt", "LoRAP (412)", "MLP (412)"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(split.capitalize() + " · paired seeds 41–43")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output, "factorial_accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    grid = (0, .25, .5, 1, 2)
    for mode, label, color in (("learned", "Prompt on", "#4477aa"), ("off", "Prompt off", "#ee7733")):
        for ax, metric, scale in ((axes[0], "accuracy", 100), (axes[1], "loss", 1)):
            groups = [[r["validation"][metric]*scale for r in probes if
                       r["precision"] == "qat" and r["intervention"] == mode and r["strength"] == s] for s in grid]
            ax.errorbar(grid, [np.mean(g) for g in groups], yerr=[np.std(g, ddof=1) for g in groups],
                        capsize=3, marker="o", label=label, color=color)
            ax.set_xlabel("Local message-error multiplier λ (not bitwidth)")
            ax.set_ylabel("Validation accuracy (%)" if metric == "accuracy" else "Validation cross entropy")
            ax.axvline(1, color="grey", alpha=.4, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
    fig.suptitle("Fixed QAT checkpoint · λ=0 local FP message, λ=1 original quantized message")
    fig.tight_layout()
    save_figure(fig, output, "local_error_interventions")

    report = ["# LoRAP 机制验证：量化补偿还是额外容量", "",
              "本报告以原始 LoRAP 为对象，使用 CiteSeer 公共划分、两层 16 维 GIN、种子 41/42/43、每次训练 100 轮。"
              "QAT/FP32 × 无提示/LoRAP/等参数残差 MLP 共 18 组；其中 3 组 QAT LoRAP 复用首轮已完成结果，15 组新训练。"
              "本轮不新增 MNIST 训练；已有 MNIST 两轮结果仅是功能验证，不纳入机制判断。", "",
              "## 读法与判据", "",
              "量化补偿和额外容量可以同时存在。纯粹的量化补偿解释应得到量化依赖性的支持；"
              "如果去掉量化后 LoRAP 仍能提升独立训练模型，则其作用不能全部归因于量化补偿。"
              "下列比较没有预设必须得出哪一种结果。", "",
              "- 独立训练的 FP32 增益：检查没有量化误差时，提示是否仍提供容量、归纳偏置或优化方面的收益。",
              "- 配对交互效应：(QAT LoRAP − QAT 无提示) − (FP32 LoRAP − FP32 无提示)。"
              "准确率用百分点，CE 用损失降低量；两者正值均表示 LoRAP 在 QAT 中更有利。"
              "这是训练制度层面的交互，QAT 还包含存储惩罚和可学习量化参数，不能直接解释为纯补偿量。",
              "- 固定检查点干预：只用验证集比较关闭、反号、打乱、取均值和误差方向分解。"
              "这能衡量已训练分支的作用，但关闭提示涉及共适应，不能代替无提示独立训练。", "",
              "## 公平性与参数", "",
              "所有六组采用配对骨干初始权重和训练/dropout 随机种子；两个精度的 LoRAP 共享初始提示基及打分层。"
              "LoRAP 使用 20 个提示、rank 2，共 412 个参数；残差 MLP 为 16→12→16、ReLU、两层含偏置，恰好也是 412 个参数，参数差异 0%。"
              "提示/MLP 主学习率 0.01、权重衰减 0，骨干沿用前轮设置。"
              "前轮 511 参数 MLP 匹配的是条件 LoRAP，本轮的 412 参数 MLP 专门匹配原 LoRAP。", "",
              "FP32 完全绕过权重、特征和消息量化器，冻结不再使用的量化参数，并将存储惩罚设为 0。"
              "保留同一 GIN 结构（包括最终 ReLU）。FP32 状态文件中保留冻结的量化张量只为加载兼容；它们不参与预测或训练，"
              "不能算作 FP32 有效容量。FP32 日志中的 legacy_budget_proxy 仍由旧模型计算，但没有预算意义。", "",
              "所有新训练按最低验证 CE 选模，平局取较早轮次，恢复最佳检查点后测试一次；"
              "复用的三个旧结果不重新测试。干预仅访问验证标签，不选模、不调超参数。", "",
              "## 独立训练结果", "",
              "下表为三个种子的均值 ± 样本标准差。准确率以 % 表示。", "",
              "| 精度 | 方法 | 提示参数 | 验证准确率 | 测试准确率 | 验证 CE | 测试 CE |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for precision in PRECISIONS:
        for control in CONTROLS:
            rows = [r for r in results if r["precision"] == precision and r["control"] == control]
            report.append(f"| {precision.upper()} | {LABELS[control]} | {rows[0]['aggregate_prompt_parameters']} | " +
                          " | ".join(fmt([r[split][metric] for r in rows], scale, digits)
                                     for split, metric, scale, digits in (("validation", "accuracy", 100, 2),
                                         ("test", "accuracy", 100, 2), ("validation", "loss", 1, 4), ("test", "loss", 1, 4))) + " |")
    report += ["", "![独立训练准确率](factorial_accuracy.png)", "",
               "### 实际位宽与预算", "",
               "量化预算目标仍为旧代理的 5；量化器的位宽在训练中学习，所选轮次也可能不同，"
               "因此同预算设置不保证相同实际位宽。下表为所选检查点在验证前向中的实际平均激活位宽，"
               "按各量化器处理的张量元素数加权，覆盖整图传播，并非仅验证节点或物理存储量。", "",
               "| QAT 方法 | 平均有效激活位宽 | 最佳检查点预算代理 | 可训练总参数 |",
               "|---|---:|---:|---:|"]
    for control in CONTROLS:
        rows = [r for r in results if r["precision"] == "qat" and r["control"] == control]
        report.append(f"| {LABELS[control]} | {fmt([r['mean_activation_effective_bits'] for r in rows], digits=4)} | "
                      f"{fmt([r['validation']['legacy_budget_proxy'] for r in rows], digits=3)} | {rows[0]['trainable_parameters']} |")
    report += ["", "FP32 的无提示/LoRAP/MLP 可训练参数分别为 " + "/".join(str(next(r["trainable_parameters"] for r in results
                  if r["precision"] == "fp32" and r["control"] == control)) for control in CONTROLS) +
               "。QAT–FP32 的参数差异来自量化器；各精度内部两个附加分支均严格增加 412 个参数。"
               "本轮交互估计同时包含位宽学习、存储惩罚和选模带来的影响；它不能替代固定量化器的严格机制实验。", "",
               "## 相对无提示的配对增益与交互", "",
               "| 方法 | 划分 | QAT 准确率增益（pp） | FP32 准确率增益（pp） | 交互（pp） | QAT CE 降低 | FP32 CE 降低 | CE 交互 |",
               "|---|---|---:|---:|---:|---:|---:|---:|"]
    for control in ("original", "mlp_exact"):
        for split in ("validation", "test"):
            rows = [r for r in effects if r["control"] == control and r["split"] == split]
            fields = ("qat_accuracy_gain_pp", "fp32_accuracy_gain_pp", "interaction_accuracy_pp",
                      "qat_loss_gain", "fp32_loss_gain", "interaction_loss")
            report.append(f"| {LABELS[control]} | {split} | " + " | ".join(
                fmt([r[field] for r in rows], digits=2 if "pp" in field else 4) for field in fields) + " |")
    report += ["", "三种子不足以建立可靠的显著性或等效性结论。不能将交互估计的正负单独视作机制证明，"
               "也不能用没有观察到稳定提升来证明没有额外容量。", "",
               "## 固定检查点：提示是否有用、节点对应是否重要", "",
               "仅比较 LoRAP 检查点的验证结果。平行/正交分解使用当前层未量化输入形成的完整 GIN 修正向量，"
               "属于 oracle 分析，不能用于部署。逐节点分解保证 p=p∥+p⊥；误差为零时平行分量为零。"
               "打乱使用固定种子的全图节点置换；均值使用全图提示均值，均不使用标签。", "",
               "| 训练精度 | 干预 | 验证准确率（%） | 验证 CE |",
               "|---|---|---:|---:|"]
    for precision in PRECISIONS:
        for mode in INTERVENTIONS:
            rows = [r for r in probes if r["precision"] == precision and r["intervention"] == mode and r["strength"] == 1]
            report.append(f"| {precision.upper()} | {MODES[mode]} | " +
                          fmt([r["validation"]["accuracy"] for r in rows], 100) + " | " +
                          fmt([r["validation"]["loss"] for r in rows], digits=4) + " |")
    report += ["", "若打乱或取均值后变化很小，说明这组检查点的细粒度节点对应贡献较弱；"
               "但提示输出本身若接近常量，打乱就不是强干预。"
               "反号和关闭造成性能下降只能说明该分支有作用，不能直接证明作用是量化补偿。", "",
               "## 局部几何误差：包括 GIN 自身节点项", "",
               "固定第二层输入 H 和权重，令 Hq 为消息量化输出、A 为真实去自环边的求和聚合。"
               "邻居误差 Eagg=A(H)−A(Hq)；完整更新输入误差 Efull=(1+ε)(H−Hq)+Eagg。"
               "提示 p 在聚合后加入，补偿后误差为 E−p。这里的参考只绕过当前消息量化器，"
               "仍保留上游量化及当前权重，绝非完整 FP32 教师。", "",
               "降低率定义为 1−‖E−p‖F/‖E‖F；误差为零时不定义，保存 null。"
               "下表仅统计 500 个验证节点上的原始 QAT 提示。", "",
               "| 误差对象 | 范数降低（%） | 全局余弦 | 提示范数/误差范数（%） |",
               "|---|---:|---:|---:|"]
    for kind, label in (("neighbor", "仅邻居聚合"), ("full_update", "完整 GIN 更新输入")):
        rows = [r["geometry"][kind] for r in probes if r["precision"] == "qat" and
                r["intervention"] == "learned" and r["strength"] == 1]
        report.append(f"| {label} | {fmt([r['norm_reduction'] for r in rows], 100, 5)} | "
                      f"{fmt([r['cosine'] for r in rows], digits=4)} | "
                      f"{fmt([r['prompt_to_error_norm'] for r in rows], 100, 4)} |")
    report += ["", "Frobenius 误差度量没有按分类重要性加权；很小的范数降低不能排除任务相关的补偿。"
               "保留正交分量仍有作用也不能证明其与所有量化误差无关，因为下游含非线性和其他量化器。", "",
               "## 改变量化误差强度", "",
               "固定 QAT 检查点和上游 H，将第二层消息设为 Hλ=H+λ(Hq−H)，并重新计算对应提示。"
               "λ=0 只关闭当前消息量化；λ=1 是原模型；λ=2 是误差外推，不代表真实 2-bit 或任何位宽。"
               "绘图为三个种子的均值 ± 样本标准差。", "",
               "![局部误差干预](local_error_interventions.png)", "",
               "另记录了同一 QAT 检查点同时绕过所有量化器、提示开/关的验证结果，见 interventions.csv。"
               "该操作会造成分布变化，不能拿来替代独立训练的 FP32 对照。", "",
               "## 每种子结果", "",
               "| 精度 | 方法 | seed | 最佳轮次 | 验证准确率 | 测试准确率 | 验证 CE | 测试 CE |",
               "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        report.append(f"| {r['precision']} | {LABELS[r['control']]} | {r['seed']} | {r['best_epoch']} | "
                      f"{100*r['validation']['accuracy']:.2f} | {100*r['test']['accuracy']:.2f} | "
                      f"{r['validation']['loss']:.4f} | {r['test']['loss']:.4f} |")
    report += ["", "## 范围与复现", "",
               "本轮是单数据集、三个种子、固定超参数的机制筛查。QAT 预算代理沿用原设置；"
               "是否满足目标应查看各检查点位宽和预算日志，不能假定同预算就同存储。"
               "FP32 和 QAT 均未做独立超参数搜索，准确率增益会同时受优化、正则化与容量影响。"
               "该结果不代表所有 LoRAP 模型、位宽或数据集。", "",
               "从仓库根目录运行，输出目录必须不存在：", "", "```sh",
               "PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m unittest runs.lorap_study.test_study runs.lorap_study.test_mechanism -v",
               "PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.mechanism --output-dir result/lorap_study/mechanism_repeat --threads 4",
               "# 若要从头训练全部 18 组，加 --no-reuse。默认复用 pilot_20260909_v1 的三组原 LoRAP。",
               "```", "",
               "保存内容包括 protocol.json、源代码快照和哈希、数据划分索引、每轮日志、最佳检查点、每种子结果、"
               "配对效应与全部验证干预的 JSON/CSV，以及 PNG/PDF/SVG 图。复用检查点记录源路径与 SHA256，"
               "并核对训练依赖源码、初始化、配置、划分和验证损失的一致性。"]
    findings_path = output / "FINDINGS.md"
    if findings_path.exists():
        findings = findings_path.read_text().replace("# 当前结论", "## 当前结论", 1)
        report[4:4] = [findings, ""]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    source = Path(__file__)
    shutil.copy2(source, output / "mechanism_report_snapshot.py")
    write_json(output / "report_provenance.json", dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               findings_sha256=hashlib.sha256(findings_path.read_bytes()).hexdigest() if findings_path.exists() else None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate_report(parser.parse_args().output)
