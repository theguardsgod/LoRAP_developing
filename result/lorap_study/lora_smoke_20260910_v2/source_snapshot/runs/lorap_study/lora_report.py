"""Regenerate the LoRA comparison from saved results, without evaluating models."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .run import write_json

LABELS = dict(no_prompt="完全无提示", main="固定 main 提示方法", lora="LoRA joint-QAT")


def mean_std(values, scale=1.):
    a = np.asarray(values, dtype=float) * scale
    if not len(a):
        return "—"
    return f"{a.mean():.3f} ± {a.std(ddof=1):.3f}" if len(a) > 1 else f"{a[0]:.3f} (n=1)"


def generate_report(root):
    root = Path(root)
    study = json.loads((root / "study.json").read_text())
    results = []
    expected = 0
    for config in study["configs"]:
        expected += len(config["variants"]) * len(config["seeds"])
        for seed in config["seeds"]:
            for variant in config["variants"]:
                path = root / config["dataset"] / variant / f"seed_{seed}" / "result.json"
                if path.exists():
                    results.append(json.loads(path.read_text()))
    tested = sum(r["test_evaluations"] for r in results)
    complete = len(results) == expected and (study["smoke"] or tested == expected)
    write_json(root / "results.json", results)
    write_json(root / "status.json", dict(fits_completed=len(results), fits_total=expected,
               tests_completed=tested, tests_expected=0 if study["smoke"] else expected,
               smoke=study["smoke"], complete=complete))
    flat = []
    for r in results:
        flat.append(dict(dataset=r["dataset"], variant=r["variant"], seed=r["seed"], epochs=r["epochs"],
                    best_epoch=r["best_epoch"], validation_CE=r["validation"]["loss"],
                    validation_accuracy=r["validation"]["accuracy"],
                    test_CE=r["test"]["loss"] if r["test"] else None,
                    test_accuracy=r["test"]["accuracy"] if r["test"] else None,
                    total_parameters=r["total_parameters"], trainable_parameters=r["trainable_parameters"],
                    prompt_parameters=r["prompt_parameters"], lora_parameters=r["lora_parameters"],
                    prompt_fp32_bytes=r["prompt_fp32_bytes"], lora_fp32_bytes=r["lora_fp32_bytes"],
                    prompt_buffer_bytes=r["prompt_buffer_bytes"],
                    activation_effective_bits=r["mean_activation_effective_bits"],
                    legacy_budget_proxy=r["validation"]["legacy_budget_proxy"],
                    training_seconds=r["training_seconds"], peak_cuda_memory_bytes=r["peak_cuda_memory_bytes"],
                    device=r["device"], gpu=r["gpu_name"]))
    if flat:
        with (root / "per_seed.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    pairs = []
    for config in study["configs"]:
        for left, right in (("lora", "main"), ("lora", "no_prompt"), ("main", "no_prompt")):
            by_key = {(r["variant"], r["seed"]): r for r in results
                      if r["dataset"] == config["dataset"] and r["test"] is not None}
            for seed in config["seeds"]:
                a, b = by_key.get((left, seed)), by_key.get((right, seed))
                if a is not None and b is not None:
                    pairs.append(dict(dataset=config["dataset"], comparison=f"{left}-{right}", seed=seed,
                                      accuracy_difference_pp=100 * (a["test"]["accuracy"] - b["test"]["accuracy"]),
                                      CE_difference=a["test"]["loss"] - b["test"]["loss"]))
    if pairs:
        with (root / "paired_differences.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    report = ["# 标准权重 LoRA：联合 QAT 对照", "",
              f"训练完成 {len(results)}/{expected} 组；最终测试完成 {tested} 组；"
              f"{'已完成' if complete else '进行中'}。", "",
              "## 协议", "",
              "三组均联合训练骨干与既有可训练量化参数。no_prompt 和 lora 移除所有提示；"
              "main 沿用固定提示方法，MNIST 保留输入 GPF。LoRA 是标准权重分支的联合 QAT 对照，"
              "不是冻结骨干的 PEFT，也不是 GLoRA/GraphLoRA 论文复现。", "",
              "LoRA 使用量化线性层已有的量化输入：Linear(x_q,W_q,b)+(alpha/r)x_q AᵀBᵀ；"
              "A 为 Kaiming 初始化，B 为零，dropout=0，因子为 FP32。分支不会合并后重新量化，"
              "特征预算只累计一次。公共骨干初始化、训练随机种子和图数据顺序配对。", "",
              "各组按最低验证交叉熵选模，平局取较早轮次；每个数据集全部预定训练完成后才释放测试。"
              "main 的条件统计只由训练节点/图校准。历史图任务的 none 仍保留输入 GPF，"
              "与本轮完全无提示的 no_prompt 不同，因此本轮从头训练同期对照。", "",
              "| 数据集 | 划分 | epochs | seeds | LoRA rank / alpha |",
              "|---|---|---:|---|---|"]
    for c in study["configs"]:
        split = str(c["subset_sizes"]) if c["subset_sizes"] else "完整公共划分"
        report.append(f"| {c['dataset']} | {split} | {c['epochs']} | {c['seeds']} | {c['lora_rank']} / {c['lora_alpha']} |")
    if study["smoke"]:
        report += ["", "本次为 GPU 冒烟检查，不访问测试指标，不作为正式性能结果。"]
    report += ["", "## 测试准确率与资源", "",
               "准确率为百分比，均值 ± 样本标准差；完整逐种子数据见 per_seed.csv。", "",
               "| 数据集 | 方法 | 已训练 / 已测试 | 测试准确率 % | 测试 CE | 可训练参数 | 提示 / LoRA 参数 | 额外参数 KiB | 有效激活位宽 | 训练秒 | 峰值显存 MiB |",
               "|---|---|---|---|---|---:|---|---:|---|---|---|"]
    for config in study["configs"]:
        for variant in config["variants"]:
            group = [r for r in results if r["dataset"] == config["dataset"] and r["variant"] == variant]
            tested_group = [r for r in group if r["test"] is not None]
            first = group[0] if group else None
            report.append(f"| {config['dataset']} | {LABELS[variant]} | {len(group)} / {len(tested_group)} "
                          f"| {mean_std([r['test']['accuracy'] for r in tested_group], 100)} "
                          f"| {mean_std([r['test']['loss'] for r in tested_group])} "
                          f"| {first['trainable_parameters'] if first else '—'} "
                          f"| {str(first['prompt_parameters']) + ' / ' + str(first['lora_parameters']) if first else '—'} "
                          f"| {(first['prompt_fp32_bytes'] + first['lora_fp32_bytes']) / 1024 if first else '—'} "
                          f"| {mean_std([r['mean_activation_effective_bits'] for r in group])} "
                          f"| {mean_std([r['training_seconds'] for r in group])} "
                          f"| {mean_std([r['peak_cuda_memory_bytes'] for r in group], 1 / 2**20)} |")
    report += ["", "## 配对差值", "", "准确率差值单位为百分点；正值表示前一方法更高。", "",
               "| 数据集 | 比较 | n | 准确率差值 | CE 差值 |", "|---|---|---:|---|---|"]
    for dataset, comparison in sorted({(p["dataset"], p["comparison"]) for p in pairs}):
        group = [p for p in pairs if p["dataset"] == dataset and p["comparison"] == comparison]
        report.append(f"| {dataset} | {comparison} | {len(group)} "
                      f"| {mean_std([p['accuracy_difference_pp'] for p in group])} "
                      f"| {mean_std([p['CE_difference'] for p in group])} |")
    report += ["", "## 解释与复现", "",
               "相同 rank 不代表等容量；输入维度较大的节点模型尤其如此。FP32 适配因子的参数字节数"
               "单列，校准缓冲区字节数见 CSV；量化预算代理与逻辑位宽不代表实际整数部署或物理模型大小。"
               "训练秒包含训练和逐轮验证，不含离线校准及磁盘检查点写入；峰值为每组训练的 CUDA allocated memory。", "",
               "MNIST 是固定小样本验证，三个种子不支持论文级显著性结论。原数据划分被重复使用，"
               "不是新的独立测试集。CUDA scatter 归约可能产生微小数值差异；验证复现容差沿用既有 5e-4 CE。", "",
               "所有训练从 source_snapshot 执行。配置、初始化指纹、逐轮记录、best.pt、last.pt、"
               "量化观测和一次性测试记录均已保存；恢复命令见 RUN_COMMANDS.md。", "",
               "方法出处：[LoRA 官方实现](https://github.com/microsoft/LoRA)。"]
    (root / "REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    generate_report(parser.parse_args().root)
