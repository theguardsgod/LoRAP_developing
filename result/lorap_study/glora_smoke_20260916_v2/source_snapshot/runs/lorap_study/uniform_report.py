"""Chinese report for predeclared uniform-rank component ablations."""

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

from .improve_report import figure, fmt, merged_error
from .run import ROOT, write_json
from .uniform import LABELS

SHORT = dict(none="No aggregate prompt", original="Original", main="Main",
             nonzero_start="Nonzero start", raw_conditions="Raw conditions",
             no_conditions="No conditions", spectral_basis="Spectral basis")


def generate_report(output):
    output = Path(output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    config = protocol["config"]
    write_json(output / "run_config.json", config)
    results = json.loads((output / "results.json").read_text())
    assert all(row["test_evaluations"] == 1 and row["test"] is not None for row in results)
    variants, seeds = config["variants"], config["seeds"]
    groups = {v: [r for r in results if r["variant"] == v] for v in variants}
    flat = [dict(variant=r["variant"], seed=r["seed"], best_epoch=r["best_epoch"],
                 validation_CE=r["validation"]["loss"], validation_accuracy=r["validation"]["accuracy"],
                 test_CE=r["test"]["loss"], test_accuracy=r["test"]["accuracy"],
                 prompt_parameters=r["aggregate_prompt_parameters"],
                 calibration_buffer_bytes=r["calibration_buffer_bytes"],
                 activation_effective_bits=r["mean_activation_effective_bits"],
                 validation_budget=r["validation"]["legacy_budget_proxy"])
            for r in results]
    with (output / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    report = ["# 零初始化＋标准化条件＋均匀分层 rank", "",
              f"数据集：{config['dataset']}；每组 {config['epochs']} 轮；种子 {seeds}；CPU。", ""]
    findings = output / "FINDINGS.md"
    if findings.exists():
        report += [findings.read_text(), ""]
    report += ["## 固定方法与实验边界", "",
        "主版本 `main` 在训练前固定，不按本轮验证或测试结果挑选变体。配置顶层 basis 字段描述主版本，"
        "具体对照按 variant 决定其实际初始化（例如 spectral_basis 使用误差谱基）。各提示层独立使用 rank 2 的低秩因子，"
        "共享打分层、条件投影和门控。左因子为零、右因子为随机正交行向量；零的是初始修正输出，"
        "并非两个因子同时置零。首次更新可学习左因子，后续其他提示参数获得梯度。", "",
        "令 S 为量化消息的邻居和，ρ=√max(mean(S²),10⁻¹²)，z=clip((c−μ)/s,−5,5)。每层使用：", "",
        "```text", "A_l = 0,  B_l B_lᵀ = I₂  （仅初始化时）",
        "p_l = max(stopgrad(ρ), ρ_floor,l) · 2σ(g(z_l))",
        "      · softmax(a(S_l / ρ_l) + C z_l) · A_l B_l",
        "output_l = S_l + p_l", "```", "",
        "条件投影与门控参数也初始化为零，因此初始门控倍率为 1。"
        "RMS 查询归一化和幅度缩放沿用上一轮稳定化结构；本轮只改变表中指定组件，"
        "没有单独消融 RMS 操作。正交性仅是初始化，不作为训练约束。", "",
        "条件依次为 log1p 入度、消息量化器平均对数步长、平均有效位宽/8、邻居饱和比例，"
        "均由实际消息边和量化器计算并停止梯度。无邻居时饱和比例为零。训练校准 μ 为均值，"
        "s 为总体标准差与 [0.5,1,0.125,0.1] 的逐项最大值，避免常量条件除零。"
        "当前消息位宽冻结，位宽条件基本为常量；本实验不验证跨位宽泛化。", "",
        "MNIST 校准只使用训练子集的前 128 张图，每层最多固定抽样 4096 节点；CiteSeer 仅统计训练 mask。"
        "主版本校准只需量化聚合和条件统计，不运行未量化参考或 SVD。"
        "本次若包含 spectral_basis 对照，共用离线误差谱校准结果中的相同统计量；"
        "只有该对照把参考误差基向量写入可训练因子。", "",
        "| 变体 | 相对主版本的变化 |", "|---|---|"]
    changes = dict(none="移除聚合提示；MNIST 输入 GPF-plus 保留", original="原共享 LoRAP；整体结构与容量均不同",
                   main="零左因子＋标准化/截断条件＋随机正交右因子＋每层 rank 2",
                   nonzero_start="只将左因子改为 Xavier 随机初始化", raw_conditions="只取消条件标准化及截断",
                   no_conditions="移除条件投影与门控，少 85 个有效参数", spectral_basis="只将右因子初始化改为训练局部误差谱基")
    for v in variants:
        report.append(f"| `{v}` | {changes[v]} |")
    report += ["", "同种子共享骨干、输入提示、打分层初始权重与数据顺序；各控制间应相同的右因子也一致。"
        "所有提示参数使用主学习率与零权重衰减，量化器学习率、存储惩罚和调度沿用既定配置。"
        "每组以最低验证交叉熵选模，平局取早；所有组训练结束后，各最佳模型只评估一次测试集。", "",
        "新种子用于检查初始化敏感性；验证/测试划分仍与前轮相同，不能称为独立新测试集。"
        "若使用 MNIST 512/256/256 子集，这仍是小样本开发实验，不代表完整 MNIST 基准。", "",
        "## 结果：均值 ± 样本标准差", "",
        "| 方法 | 聚合提示参数 | 验证 CE↓ | 验证准确率 %↑ | 测试 CE↓ | 测试准确率 %↑ |",
        "|---|---:|---:|---:|---:|---:|"]
    for v, rows in groups.items():
        report.append(f"| {LABELS[v]} | {rows[0]['aggregate_prompt_parameters']} | "
            f"{fmt([r['validation']['loss'] for r in rows],digits=4)} | "
            f"{fmt([r['validation']['accuracy'] for r in rows],100)} | "
            f"{fmt([r['test']['loss'] for r in rows],digits=4)} | "
            f"{fmt([r['test']['accuracy'] for r in rows],100)} |")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, split, metric, scale, label in ((axes[0], "validation", "loss", 1, "Validation CE (lower is better)"),
                                           (axes[1], "test", "accuracy", 100, "Test accuracy (%)")):
        values = [np.array([r[split][metric]*scale for r in groups[v]]) for v in variants]
        ax.bar(range(len(variants)), [x.mean() for x in values],
               yerr=[x.std(ddof=1) if len(x)>1 else 0 for x in values], capsize=3,
               color=["#277da8" if v=="main" else "#b6c4cf" for v in variants])
        for i, value in enumerate(values):
            ax.scatter(np.full(len(value), i), value, color="#263238", s=12, zorder=3)
        ax.set_xticks(range(len(variants)), [SHORT[v] for v in variants], rotation=35, ha="right")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{config['dataset']} · {config['epochs']} epochs · seeds {seeds} · sample SD")
    fig.tight_layout()
    figure(fig, output, "component_comparison")
    report += ["", "![组件对照](component_comparison.png)", "", "## 与主版本的配对差值", "",
        "以下统一计算 main − 对照；准确率为百分点，CE 为原始差值。正的准确率差值有利于主版本，"
        "负的 CE 差值有利于主版本。三种子的均值不代表统计显著。", "",
        "| 对照 | seed | 验证 CE 差 | 测试准确率差（百分点） |", "|---|---:|---:|---:|"]
    paired = []
    if "main" in groups:
        for variant, rows in groups.items():
            if variant == "main":
                continue
            for row in rows:
                main = next(r for r in groups["main"] if r["seed"] == row["seed"])
                val = main["validation"]["loss"]-row["validation"]["loss"]
                test = 100*(main["test"]["accuracy"]-row["test"]["accuracy"])
                paired.append(dict(control=variant, seed=row["seed"], validation_CE_difference=val,
                                   test_accuracy_difference_pp=test))
                report.append(f"| {variant} | {row['seed']} | {val:+.4f} | {test:+.2f} |")
    write_json(output / "paired_differences.json", paired)
    if "main" in groups:
        fig, axes = plt.subplots(1, len(seeds), figsize=(4.5*len(seeds), 3.5), squeeze=False)
        for ax, seed in zip(axes[0], seeds):
            for variant in ("none", "original", "main", "spectral_basis"):
                if variant not in groups:
                    continue
                history = [json.loads(line) for line in (output/variant/f"seed_{seed}/epochs.jsonl").read_text().splitlines()]
                ax.plot([r["epoch"] for r in history], [r["validation"]["loss"] for r in history],
                        label=SHORT[variant], linewidth=2 if variant=="main" else 1)
            ax.set_title(f"Seed {seed}")
            ax.set_xlabel("Epoch")
        axes[0][0].set_ylabel("Validation CE")
        axes[0][-1].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        figure(fig, output, "validation_learning_curves")
        report += ["", "![验证损失轨迹](validation_learning_curves.png)", "",
                   "轨迹展示中途波动，最终报告仍严格使用最低验证 CE 检查点。"]
    report += ["", "## 局部量化误差", "",
        "固定当前层输入和权重，只绕过该层聚合前消息量化。完整 GIN 更新误差还包含自身节点项。"
        "各方法上游输入不同，参考不是共同的完整 FP32 教师。降低率=1−‖E−p‖F/‖E‖F；"
        "多批次先合并平方范数/内积。指标只在验证数据上计算。", "",
        "| 方法 | seed | 层 | 邻居误差降低 % | 完整更新误差降低 % | 完整误差余弦 |",
        "|---|---:|---|---:|---:|---:|"]
    error_rows = []
    show = lambda x, scale=1: f"{x*scale:.6f}" if x is not None else "未定义"
    for r in results:
        path = output / r["variant"] / f"seed_{r['seed']}/full_update_diagnostics.json"
        if not path.exists():
            continue
        for layer, batches in json.loads(path.read_text()).items():
            neighbor, full = merged_error(batches, "neighbor"), merged_error(batches)
            error_rows.append(dict(variant=r["variant"], seed=r["seed"], layer=layer,
                                  neighbor_reduction=neighbor["reduction"], full_reduction=full["reduction"],
                                  full_cosine=full["cosine"]))
            report.append(f"| {r['variant']} | {r['seed']} | {layer} | {show(neighbor['reduction'],100)} | "
                          f"{show(full['reduction'],100)} | {show(full['cosine'])} |")
    write_json(output / "local_error_summary.json", error_rows)
    calibration_rows = []
    for seed in seeds:
        for layer in json.loads((output/f"calibration_seed_{seed}/calibration.json").read_text()):
            if "singular_values" not in layer:
                continue
            energy = np.square(layer["singular_values"])
            centered = np.square(layer["centered_singular_values"])
            calibration_rows.append(dict(seed=seed, layer=layer["layer"],
                sampled_nodes=layer["sampled_nodes"],
                rank_2_energy=float(energy[:2].sum()/energy.sum()) if energy.sum() else None,
                centered_rank_2_energy=float(centered[:2].sum()/centered.sum()) if centered.sum() else None))
    if calibration_rows:
        report += ["", "### 初始训练误差的 rank-2 能量", "",
            "下表来自训练前共同无聚合提示骨干的完整 GIN 局部误差。去均值列排除了均值方向的主导作用。"
            "本轮 rank 始终为 2，没有按这些结果重新分配容量；随机正交主版本也不使用谱基。", "",
            "| seed | 层 | 抽样节点 | 原始 rank-2 能量 % | 去均值 rank-2 能量 % |",
            "|---:|---|---:|---:|---:|"]
        for row in calibration_rows:
            report.append(f"| {row['seed']} | {row['layer']} | {row['sampled_nodes']} | "
                          f"{show(row['rank_2_energy'],100)} | {show(row['centered_rank_2_energy'],100)} |")
    write_json(output / "calibration_rank_summary.json", calibration_rows)
    if config["diagnostics"]:
        # One panel per layer, mean across seed-specific deterministic reservoirs.
        records = {v: [json.loads((output/v/f"seed_{s}/diagnostics.json").read_text())["layers"] for s in seeds] for v in variants}
        layer_names = list(records[variants[0]][0])
        fig, axes = plt.subplots(1, len(layer_names), figsize=(4*len(layer_names), 3.5), squeeze=False)
        for ax, layer in zip(axes[0], layer_names):
            for variant in variants:
                curves = [np.asarray(row[layer]["cumulative_energy"])[:12]*100 for row in records[variant]]
                curve = np.mean(curves, axis=0)
                ax.plot(range(1, len(curve)+1), curve, label=SHORT[variant], linewidth=2 if variant=="main" else 1)
            ax.set_title(layer)
            ax.set_xlabel("Rank")
            ax.set_ylim(0, 101)
            ax.axvline(2, color="gray", linestyle="--", linewidth=.8)
        axes[0][0].set_ylabel("Local neighbor error energy (%)")
        axes[0][-1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        figure(fig, output, "validation_error_spectrum")
        report += ["", "![验证误差累计能量](validation_error_spectrum.png)", "",
            "图为各方法最佳检查点的局部邻居误差累计谱能量，按种子平均；每层每种子最多抽样 4096 节点。"
            "未经中心化的误差谱可能主要反映均值方向；高 rank-2 能量本身不能证明提示学到正确修正。"
            "完整奇异值保存在各组 diagnostics.json。"]
    report += ["", "准确率提升与局部误差降低是不同结果；只有分类指标改善不能证明量化补偿。", "",
        "## 逐种子结果与预算", "",
        "| 方法 | seed | 最佳轮次 | 验证 CE | 测试准确率 % | 平均有效激活位宽 | 预算代理 | 校准缓冲区字节 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        report.append(f"| {r['variant']} | {r['seed']} | {r['best_epoch']} | {r['validation']['loss']:.4f} | "
            f"{100*r['test']['accuracy']:.2f} | {r['mean_activation_effective_bits']:.4f} | "
            f"{r['validation']['legacy_budget_proxy']:.4f} | {r['calibration_buffer_bytes']} |")
    condition_rows = []
    for r in groups.get("main", []):
        path = output / "main" / f"seed_{r['seed']}/condition_drift.json"
        if path.exists():
            for layer, values in json.loads(path.read_text())["layers"].items():
                condition_rows.append((r["seed"], layer, values))
    if condition_rows:
        report += ["", "## 标准化条件的验证集检查", "",
            "该检查只观察实际提示输入，不增加量化或参考计算，验证输出与普通评估完全一致。"
            "下面依次列出入度、对数步长、位宽、饱和比例四个条件超过 ±5 截断阈值的节点比例。", "",
            "| seed | 层 | 四项条件截断比例 % | 平均门控倍率 | 提示/聚合范数 |",
            "|---:|---|---|---:|---:|"]
        for seed, layer, values in condition_rows:
            fractions = ", ".join(f"{100*x:.2f}" for x in values["potential_clipped_fraction"])
            ratio = values["prompt_to_aggregate_norm"]
            report.append(f"| {seed} | {layer} | {fractions} | {values['gate_mean']:.4f} | {show(ratio)} |")
    report += ["", f"A²Q 预算目标为 {config['a_storage']}，惩罚系数为 {config['a_loss']}。"
        "该目标是历史训练代理，不是物理存储测量，也不保证实际达到目标。提示按 FP32 每参数 4 字节计；"
        "MNIST 还共享 140 个输入提示参数。原共享 LoRAP 的 2480 个聚合提示参数与主版本 3345 个不同，"
        "整体对比不能单独排除额外容量；no_conditions 少 85 个参数，也不是严格等容量控制。", "",
        "## 复现与完整训练", "",
        "从仓库根目录运行，输出目录必须不存在。完整配置为全划分、200 轮、种子 41–50；"
        "本轮仅生成，未执行完整训练。`--no-diagnostics` 关闭验证参考分支；"
        "主版本单独运行时训练校准也不使用浮点参考。", "", "```sh"]
    prefix = "PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B"
    report += [prefix + " -m unittest runs.lorap_study.test_study runs.lorap_study.test_mechanism runs.lorap_study.test_improved runs.lorap_study.test_uniform runs.lorap_study.test_uniform_context -v",
        prefix + f" -m runs.lorap_study.uniform_run --config {output}/run_config.json --output-dir result/lorap_study/uniform_repeat --threads 4"]
    for dataset in ("MNIST", "CiteSeer"):
        report.append(prefix + f" -m runs.lorap_study.uniform_run --config {output}/configs/{dataset}_main_full.json --output-dir result/lorap_study/uniform_{dataset.lower()}_full --threads 4")
    report += [prefix + f" -m runs.lorap_study.uniform_audit {output}", "```", "",
        "每组保存配置、初始化指纹、逐轮日志、最佳检查点、一次最终测试和诊断。"
        "pre_test_results.json 保留所有训练完成且尚未测试的状态；源代码、数据索引、校准和环境信息随结果归档。"]
    if (output / "verification.json").exists():
        audit = json.loads((output / "verification.json").read_text())
        report += ["", f"验收审计：{audit['training_runs']} 组、{audit['training_epochs']} 轮，"
            f"{audit['final_test_evaluations']} 次最终测试；初始化、最低验证损失检查点和验证指标均已重现。"
            "审计不再次访问测试指标计算。"]
        if "unit_tests_passed" in audit:
            report += [f"{audit['unit_tests_passed']} 项单元测试通过，记录见 unit_tests.log。"]
    (output / "REPORT.md").write_text("\n".join(report)+"\n")
    shutil.copy2(__file__, output / "uniform_report_snapshot.py")
    write_json(output / "report_provenance.json", dict(source=str(Path(__file__).resolve()),
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        findings_sha256=hashlib.sha256(findings.read_bytes()).hexdigest() if findings.exists() else None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate_report(parser.parse_args().output)
