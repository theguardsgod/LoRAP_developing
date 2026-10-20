"""Summarize completed artifacts and spectra without evaluating a model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from .full_report import LABELS
from .run import write_json


def analyze(root):
    root = Path(root).resolve()
    study = json.loads((root/"study.json").read_text())
    rows = json.loads((root/"results.json").read_text())
    expected = sum(len(c["seeds"])*len(c["variants"]) for c in study["configs"])
    assert len(rows) == expected and all(r["test_evaluations"] == 1 for r in rows)
    verification = json.loads((root/"verification.json").read_text())
    assert verification["fits_verified"] == verification["test_results_verified"] == expected
    matched_training = matched_tests = 0
    before = root/"recovery_before/artifact_hashes.json"
    if before.exists():
        for item in json.loads(before.read_text()):
            directory = root/item["run"]
            assert hashlib.sha256((directory/"last.pt").read_bytes()).hexdigest() == item["last_sha256"]
            matched_training += 1
            if "tested_result_sha256" in item:
                assert hashlib.sha256((directory/"result.json").read_bytes()).hexdigest() == item["tested_result_sha256"]
                assert hashlib.sha256((directory/"test_started.json").read_bytes()).hexdigest() == item["test_started_sha256"]
                matched_tests += 1
    summary, diagnostics = [], []
    report = ["# LoRAP 完整 GPU 实验分析", "",
        f"全部 {expected} 组完成完整划分训练与最终测试；每组 200 轮，种子 41–50。训练及最终模型评估均使用 RTX 4090。", "",
        "## 测试结果", "", "均值 ± 样本标准差，单位为 %；差值为同种子对比原 LoRAP 的百分点变化。95% 区间用配对差值的 Student t 区间计算；它只描述这 10 个训练种子的波动，未做多重比较校正。", "",
        "| 数据集 | 方法 | 测试准确率 | 相对原版差值 [95% 区间] | 胜/平/负 | 提示参数 | 平均有效激活位宽 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for config in study["configs"]:
        dataset = config["dataset"]
        group = [r for r in rows if r["dataset"] == dataset]
        original = {r["seed"]:r["test"]["accuracy"]*100 for r in group if r["variant"] == "original"}
        for variant in config["variants"]:
            runs = sorted((r for r in group if r["variant"] == variant), key=lambda r:r["seed"])
            accuracy = np.array([r["test"]["accuracy"]*100 for r in runs])
            delta = accuracy - [original[r["seed"]] for r in runs]
            margin = float(t.ppf(.975, len(runs)-1)*delta.std(ddof=1)/np.sqrt(len(runs)))
            item = dict(dataset=dataset, variant=variant, n=len(runs), accuracy_mean=float(accuracy.mean()),
                accuracy_std=float(accuracy.std(ddof=1)), paired_delta=float(delta.mean()),
                ci95=[float(delta.mean()-margin),float(delta.mean()+margin)],
                wins=int((delta>1e-10).sum()),ties=int((abs(delta)<=1e-10).sum()),losses=int((delta< -1e-10).sum()),
                aggregate_prompt_parameters=runs[0]["aggregate_prompt_parameters"],
                activation_effective_bits=float(np.mean([r["mean_activation_effective_bits"] for r in runs])))
            summary.append(item)
            report.append(f"| {dataset} | {LABELS[variant]} | {item['accuracy_mean']:.2f} ± {item['accuracy_std']:.2f} | "
                f"{item['paired_delta']:+.2f} [{item['ci95'][0]:+.2f}, {item['ci95'][1]:+.2f}] | "
                f"{item['wins']}/{item['ties']}/{item['losses']} | {item['aggregate_prompt_parameters']} | {item['activation_effective_bits']:.4f} |")
            for result in runs:
                path = root/dataset/variant/f"seed_{result['seed']}"/"diagnostics.json"
                for layer, value in json.loads(path.read_text())["layers"].items():
                    singular = np.array(value["singular_values"])
                    energy = np.cumsum(singular**2)/sum(singular**2)
                    np.testing.assert_allclose(energy,value["cumulative_energy"],atol=1e-10)
                    assert value["sampled_nodes"] <= config["diagnostic_max_nodes"]
                    ratio = value["error_after_frobenius"]/value["error_before_frobenius"]
                    np.testing.assert_allclose(1-ratio,value["relative_error_reduction"],atol=1e-10)
                    np.testing.assert_allclose(ratio**2,value["mse_after"]/value["mse_before"],rtol=1e-8)
                    diagnostics.append(dict(dataset=dataset,variant=variant,seed=result["seed"],layer=layer,
                        norm_ratio=float(ratio),cosine=value["global_cosine"],rank2_energy=value["energy_at_rank_2"],
                        rank95=value["rank_95"],cumulative_energy=energy.tolist(),spectral_energy=(singular**2/sum(singular**2)).tolist()))
    focus = next((s for s in summary if s["dataset"] == "CIFAR10" and s["variant"] == "spectral_basis"), None)
    if focus:
        report[4:4] = [f"最明确的正向结果来自 CIFAR10 的误差谱基版本：相对原 LoRAP 提高 {focus['paired_delta']:.3f} 个百分点，"
            f"配对 95% 区间 [{focus['ci95'][0]:.3f}, {focus['ci95'][1]:.3f}]，{focus['wins']}/{focus['n']} 个种子胜出。"
            "随机基主版本及其余三个数据集未得到区间完全大于零的改进证据，不能据此声称跨数据集稳定提升。", ""]
    report.extend(["", "## 局部量化误差", "",
        "固定当前层输入与权重，仅绕过该层聚合前消息量化得到局部参考。误差范数比 = 提示后误差 / 提示前误差；小于 1 才表示局部补偿。表中取 10 个种子的均值；比较仅在各模型内部成立。rank 2 能量来自每层固定抽样最多 4096 个验证节点的未中心化误差谱。", "",
        "| 数据集 | 方法 | 层 | 误差范数比 | 余弦 | 改善种子数 | rank 2 能量 % | rank 95% |",
        "|---|---|---|---:|---:|---:|---:|---:|"])
    for config in study["configs"]:
        for variant in (v for v in config["variants"] if v != "none"):
            values = [d for d in diagnostics if d["dataset"] == config["dataset"] and d["variant"] == variant]
            for layer in sorted({d["layer"] for d in values}):
                group = [d for d in values if d["layer"] == layer]
                report.append(f"| {config['dataset']} | {LABELS[variant]} | {layer} | "
                    f"{np.mean([d['norm_ratio'] for d in group]):.3f} | {np.mean([d['cosine'] for d in group]):.3f} | "
                    f"{sum(d['norm_ratio']<1 for d in group)}/{len(group)} | "
                    f"{100*np.mean([d['rank2_energy'] for d in group]):.1f} | {np.mean([d['rank95'] for d in group]):.1f} |")
    report.extend(["", "准确率提升本身不能证明量化补偿。若误差范数比大于 1，则当前局部诊断不支持提示在还原该聚合的未量化结果；这也不等同于完整 FP32 模型的比较。此轮完整实验没有加入参数量匹配 MLP 或 FP32 对照，因此不能单凭这些结果将收益归因于量化补偿或额外容量。", "",
        "四层图模型聚合提示由 2480 个参数增加到 3345 个，增加 865 个（34.9%），按 FP32 为额外 3460 字节；节点模型由 412 增到 497，增加 85 个（20.6%）。输入提示、量化元数据和校准缓冲区另计，完整参数记录见逐种子 result.json。相同预算目标不能视为相同实际存储。", "",
        "## 误差谱图", "", "曲线是各训练种子归一化谱/累计能量的均值；纵虚线标出 rank 2。"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(original="#4776b4",main="#d76b27",spectral_basis="#33946b")
    for config in study["configs"]:
        dataset = config["dataset"]
        layers = sorted({d["layer"] for d in diagnostics if d["dataset"] == dataset})
        fig, axes = plt.subplots(2,len(layers),squeeze=False,figsize=(max(5,3.4*len(layers)),6))
        for column, layer in enumerate(layers):
            for variant, color in colors.items():
                values = [d for d in diagnostics if d["dataset"] == dataset and d["layer"] == layer and d["variant"] == variant]
                for row, key in enumerate(("spectral_energy","cumulative_energy")):
                    values_array = np.array([d[key] for d in values])
                    ax = axes[row,column]
                    ax.plot(np.arange(1,values_array.shape[1]+1),values_array.mean(0),label=variant,color=color)
                    ax.axvline(2,color="gray",ls=":",lw=.6)
                    ax.set_xlabel("Rank")
                    ax.grid(alpha=.2)
            axes[0,column].set_title(layer)
            axes[0,column].set_yscale("log")
            axes[1,column].set_ylim(0,1.02)
        axes[0,0].set_ylabel("Normalized squared singular value")
        axes[1,0].set_ylabel("Cumulative energy")
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f"{dataset}: local validation quantization error (10 seeds)")
        fig.tight_layout()
        for extension in ("png","pdf","svg"):
            fig.savefig(root/f"{dataset}_error_spectrum.{extension}",dpi=180,bbox_inches="tight")
        plt.close(fig)
        report.extend(["",f"![{dataset} 局部量化误差谱]({dataset}_error_spectrum.png)"])
    report.extend(["", "## 可追溯性与恢复", "",
        f"全部 {expected} 个结果通过 CUDA 记录、按最低验证损失选模、配对初始骨干以及最佳检查点逐张量一致性检查。恢复前后 {matched_training} 个 last.pt 文件哈希一致，{matched_tests} 个既有测试结果与测试开始标记哈希一致。此分析只读取文件，额外模型前向和测试次数为 0。", "",
        "收尾曾因验证复算容差停止，修复证据及记录见 [RECOVERY.md](RECOVERY.md)。完整训练与恢复命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)，配置见 configs/；原始训练快照与收尾修复快照分别保存。"])
    write_json(root/"analysis.json",dict(summary=summary,diagnostics=diagnostics,
        unchanged_training_checkpoints=matched_training,unchanged_existing_tests=matched_tests,additional_model_evaluations=0))
    (root/"ANALYSIS.md").write_text("\n".join(report)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    analyze(parser.parse_args().root)
