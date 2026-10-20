"""Produce a Chinese study report, CSV and standalone spectrum figures."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import DATASETS, VARIANTS

NAMES = dict(original="原 LoRAP", conditioned="量化条件 LoRAP", mlp="残差 MLP")
COLORS = dict(original="#4477AA", conditioned="#228833", mlp="#CC6677")


def formatted(values, scale=1, digits=2):
    values = np.asarray(values, dtype=float) * scale
    if len(values) == 1:
        return f"{values[0]:.{digits}f}（单次）"
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def generate_report(output):
    output = Path(output).resolve()
    result_text = (output / "results.json").read_text()
    results = json.loads(result_text)
    provenance = json.loads((output / "provenance.json").read_text())
    study = json.loads((output / "study.json").read_text())
    config_by_dataset = {config["dataset"]: config for config in study["configs"]}
    devices = sorted({json.loads((output / r["dataset"] / r["variant"] / f"seed_{r['seed']}" / "config.json").read_text())["device"] for r in results})
    expected_runs = sum(len(c["seeds"]) for c in study["configs"]) * len(study["variants"])
    lines = ["# LoRAP 首轮实验报告", "",
             "本报告由实际保存的检查点、指标和诊断文件生成。原 LoRAP、量化条件 LoRAP 和容量匹配的残差 MLP 使用相同骨干初始化、配对种子和数据顺序。",
             f"\n已完成 {len(results)}/{expected_runs} 次计划训练。" + (" **当前为进行中的部分结果。**" if len(results) < expected_runs else ""),
             "", "## 实验设置", "",
             f"- 环境：Python {provenance['python_version']}；PyTorch {provenance['versions']['torch']}；CPU 线程 {provenance['torch_threads']}。",
             f"- 实际训练设备：{', '.join(devices)}。",
             "- 选模：最低验证交叉熵，平局选较早轮次；加载最佳检查点后，每次训练仅评估一次测试集。",
             "- 所有提示参数使用各数据集的主学习率，权重衰减为零；本轮不增加辅助损失。",
             "- A²Q 的权重和消息位宽保持原优化器中的固定设置，特征位宽继续学习；初始 bit=4 不代表最终每个张量都是固定 4 bit。",
             "- 各变体共享骨干指纹，原／条件 LoRAP 共享初始提示基及打分层；指纹记录于 initialization.json。",
             "", "## 准确率与开销", "",
             "± 表示种子间样本标准差，不是置信区间。位宽按验证阶段实际访问的量化张量元素加权；非负特征按原实现的额外符号位约定换算。", "",
             "| 数据集 | 方法 | 种子数 | 测试准确率（%） | 验证损失 | 激活有效位宽 | 聚合提示参数 | 全部提示 FP32 KiB | 训练总秒数 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    csv_rows = []
    for name in DATASETS:
        for variant in VARIANTS:
            rows = [r for r in results if r["dataset"] == name and r["variant"] == variant]
            if not rows:
                continue
            lines.append(f"| {name} | {NAMES[variant]} | {len(rows)} | "
                         f"{formatted([r['test']['accuracy'] for r in rows], 100)} | "
                         f"{formatted([r['validation']['loss'] for r in rows], digits=4)} | "
                         f"{formatted([r['mean_activation_effective_bits'] for r in rows], digits=3)} | "
                         f"{rows[0]['aggregate_prompt_parameters']} | {rows[0]['prompt_fp32_bytes']/1024:.3f} | "
                         f"{sum(r['training_seconds'] for r in rows):.1f} |")
            for row in rows:
                csv_rows.append(dict(dataset=name, variant=variant, seed=row["seed"], epochs=row["epochs"],
                                     best_epoch=row["best_epoch"], test_accuracy=row["test"]["accuracy"],
                                     validation_loss=row["validation"]["loss"],
                                     activation_bits=row["mean_activation_effective_bits"],
                                     aggregate_parameters=row["aggregate_prompt_parameters"],
                                     prompt_fp32_bytes=row["prompt_fp32_bytes"],
                                     training_seconds=row["training_seconds"],
                                     legacy_budget_proxy=row["validation"]["legacy_budget_proxy"]))
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines += ["", "## 配对结果与解释", ""]
    for name in DATASETS:
        base = {r["seed"]: r for r in results if r["dataset"] == name and r["variant"] == "original"}
        conditioned = {r["seed"]: r for r in results if r["dataset"] == name and r["variant"] == "conditioned"}
        mlp = {r["seed"]: r for r in results if r["dataset"] == name and r["variant"] == "mlp"}
        paired = sorted(base.keys() & conditioned.keys())
        if paired:
            differences = [conditioned[s]["test"]["accuracy"] - base[s]["test"]["accuracy"] for s in paired]
            wins = sum(delta > 0 for delta in differences)
            lines.append(f"- {name}：条件提示相对原 LoRAP 的配对准确率变化为 {formatted(differences, 100)} 个百分点；{wins}/{len(paired)} 个种子提升。")
            val_differences = [conditioned[s]["validation"]["loss"] - base[s]["validation"]["loss"] for s in paired]
            lines.append(f"- {name}：对应的验证损失变化为 {formatted(val_differences, digits=4)}（越低越好）；{sum(delta < 0 for delta in val_differences)}/{len(paired)} 个种子降低验证损失。测试准确率的变化不能替代验证表现。")
        paired_mlp = sorted(mlp.keys() & conditioned.keys())
        if paired_mlp:
            differences = [conditioned[s]["test"]["accuracy"] - mlp[s]["test"]["accuracy"] for s in paired_mlp]
            lines.append(f"- {name}：条件提示相对容量匹配 MLP 的配对准确率变化为 {formatted(differences, 100)} 个百分点；MLP 参数量差异为 {mlp[paired_mlp[0]]['mlp_budget_difference'] * 100:.2f}%。")
        if name in config_by_dataset:
            config = config_by_dataset[name]
            lines.append(f"- {name} 设置：{config['epochs']} 轮，种子 {config['seeds']}，隐藏维度 {config['hidden_units']}，层数 {config['layers']}。")
            if config["subset_sizes"]:
                sizes = config["subset_sizes"]
                lines.append(f"- **{name} 仅为功能验证**：官方训练／验证／测试划分内固定分层抽样 {sizes['train']}/{sizes['val']}/{sizes['test']} 张图；短程准确率不代表收敛表现。")
    lines += ["", "小规模结果不用于宣称论文级收益。真实位宽和预算代理可能随训练变化，不能将同一预算惩罚等同于完全相同的存储成本。",
              "消息量化位宽在本轮保持固定，因此本轮尚未检验跨位宽泛化；条件分支的变化也包含度信息、步长和饱和比例的作用。",
              "", "| 数据集 | 方法 | 预算目标（原代理单位） | 选中模型预算代理 |",
              "|---|---|---:|---:|"]
    for name in DATASETS:
        for variant in VARIANTS:
            rows = [r for r in results if r["dataset"] == name and r["variant"] == variant]
            if rows:
                lines.append(f"| {name} | {NAMES[variant]} | {config_by_dataset[name]['a_storage']} | "
                             f"{formatted([r['validation']['legacy_budget_proxy'] for r in rows], digits=3)} |")
    lines += ["", "预算代理超过目标表示该选中模型尚未满足原惩罚项的目标；准确率应结合这一点解读。",
              "", "## 局部聚合误差", "",
              "参考分支使用同一层输入、相同边，仅绕过该层聚合前消息量化；它仍可能包含前面层产生的量化误差和提示影响，不是完整 FP32 教师。诊断只读取验证节点／验证图。",
              "", "误差降低率 = 1 − 补偿后 Frobenius 误差 / 补偿前误差；负值表示局部误差变大。余弦是整个验证误差矩阵与提示修正的夹角。各变体的上游表示可能不同，跨变体绝对误差不能直接当作共同目标上的误差。", "",
              "| 数据集 | 方法 | 层 | 补偿前误差 | 补偿后误差 | 降低率（%） | 余弦 | rank 2 能量（%） | 95% 能量所需秩 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    diagnostics = {}
    for result in results:
        path = output / result["dataset"] / result["variant"] / f"seed_{result['seed']}" / "diagnostics.json"
        if path.exists():
            diagnostics[(result["dataset"], result["variant"], result["seed"])] = json.loads(path.read_text())["layers"]
    for name in DATASETS:
        layer_names = sorted({layer for key, layers in diagnostics.items() if key[0] == name for layer in layers})
        if not layer_names:
            continue
        columns = min(2, len(layer_names))
        fig, axes = plt.subplots(math.ceil(len(layer_names) / columns), columns,
                                 figsize=(6 * columns, 4 * math.ceil(len(layer_names) / columns)), squeeze=False)
        for index, layer in enumerate(layer_names):
            ax = axes.flat[index]
            for variant in VARIANTS:
                entries = [layers[layer] for key, layers in diagnostics.items()
                           if key[0] == name and key[1] == variant and layer in layers]
                if not entries:
                    continue
                cells = []
                for metric, scale, digits in (("error_before_frobenius", 1, 2), ("error_after_frobenius", 1, 2),
                        ("relative_error_reduction", 100, 4), ("global_cosine", 1, 3),
                        ("energy_at_rank_2", 100, 1), ("rank_95", 1, 1)):
                    values = [e[metric] for e in entries if e[metric] is not None]
                    cells.append(formatted(values, scale, digits) if values else "未定义（零误差／零修正）")
                lines.append(f"| {name} | {NAMES[variant]} | {layer} | " + " | ".join(cells) + " |")
                curves = np.array([e["cumulative_energy"] for e in entries])
                x = np.arange(1, curves.shape[1] + 1)
                mean = curves.mean(0)
                ax.plot(x, mean, label=variant, color=COLORS[variant], linewidth=2)
                if len(entries) > 1:
                    std = curves.std(0, ddof=1)
                    ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                                    color=COLORS[variant], alpha=0.12)
            ax.axvline(2, color="#666666", linestyle="--", linewidth=1)
            ax.set(title=f"{name}: {layer}", xlabel="Rank", ylabel="Cumulative squared singular-value energy", ylim=(0, 1.02))
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
        for ax in list(axes.flat)[len(layer_names):]:
            ax.set_visible(False)
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"{name}_error_spectrum.{extension}", dpi=180)
        plt.close(fig)
    lines += ["", "每层最多固定抽样 4096 个验证节点；谱图展示补偿前局部误差的累计奇异值平方能量，阴影为种子间标准差。它只用于检查低秩可逼近性，不能证明实际 softmax 提示能达到最优低秩近似。", ""]
    for name in DATASETS:
        reductions = [layer["relative_error_reduction"] for key, layers in diagnostics.items()
                      if key[0] == name and key[1] == "conditioned" for layer in layers.values()
                      if layer["relative_error_reduction"] is not None]
        if reductions:
            lines.append(f"- {name}：条件提示在 {sum(value > 0 for value in reductions)}/{len(reductions)} 个层／种子组合中降低局部误差。需结合降低幅度解读，准确率收益本身不证明误差补偿机制。")
    lines.append("")
    for name in DATASETS:
        if (output / f"{name}_error_spectrum.png").exists():
            lines.append(f"![{name} error spectrum]({name}_error_spectrum.png)")
    lines += ["", "## 复现与完整训练", "",
              "本轮配置和源文件快照位于本目录；生成的完整配置使用官方全部数据、200 轮和种子 41–50。实际执行规模以上述数据集设置和完成次数为准。", "",
              "```sh", f"cd {Path(__file__).resolve().parents[2]}"]
    for name in DATASETS:
        lines.append(f"PYTHONDONTWRITEBYTECODE=1 {provenance['python']} -B -m runs.lorap_study.run "
                     f"--config {output / 'full_configs' / (name + '.json')} --variant all --device cuda:0")
    lines += ["```", "", "以上完整训练命令面向 GPU 环境，也可显式改为 `--device cpu`。迁移机器时用 `--data-root` 指定本地数据路径。", "",
              "逐种子数据见 summary.csv；逐轮训练／验证指标见 epochs.jsonl；测试结果见 result.json；诊断见 diagnostics.json；位宽分配见 quantization.json；最佳状态见 best.pt。",
              "", "训练耗时对应本报告记录的设备和实验规模，包含验证和保存检查点；本轮未测量端到端 GPU 延迟或物理峰值显存。"]
    verification_path = output / "verification.json"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text())
        lines += ["", "## 实现验证", "",
                  f"{verification['unit_tests_passed']} 项单元测试通过；另已核对 CiteSeer 与历史前向／预算代理的一致性。"
                  f"{verification['completed_runs']} 次训练、{verification['epoch_records']} 条逐轮记录通过最终审计，详情见 [verification.json](verification.json)。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    # Report rendering may be improved after training. Preserve its exact source
    # separately from the training-start snapshot and bind it to the input rows.
    source = Path(__file__).read_bytes()
    (output / "report_source_snapshot.py").write_bytes(source)
    (output / "report_provenance.json").write_text(json.dumps(dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        report_source_sha256=hashlib.sha256(source).hexdigest(),
        results_sha256=hashlib.sha256(result_text.encode()).hexdigest()), indent=2) + "\n")
    print(f"REPORT={output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    generate_report(parser.parse_args().output)
