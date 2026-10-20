"""Live and final Chinese summaries for the GPU full-training queue."""

import csv
import json
from pathlib import Path
import time

import numpy as np

from .run import write_json

LABELS=dict(none="无聚合提示",original="原 LoRAP",main="零初始化＋标准化条件＋均匀 rank（随机基）",
            spectral_basis="同组合（误差谱基）")


def generate_report(root, queue):
    root=Path(root)
    study=json.loads((root/"study.json").read_text())
    completed=[]
    for job in queue:
        if job["phase"]!="fit":
            continue
        path=root/job["dataset"]/job["variant"]/f"seed_{job['seed']}/result.json"
        if path.exists():
            completed.append(json.loads(path.read_text()))
    write_json(root/"results.json",completed)
    fitted=len(completed)
    tested=sum(r["test_evaluations"] for r in completed)
    total=sum(j["phase"]=="fit" for j in queue)
    failed=[j for j in queue if j["status"]=="failed"]
    running=[j for j in queue if j["status"]=="running"]
    flat=[]
    for r in completed:
        flat.append(dict(dataset=r["dataset"],variant=r["variant"],seed=r["seed"],epochs=r["epochs"],
            best_epoch=r["best_epoch"],validation_CE=r["validation"]["loss"],
            validation_accuracy=r["validation"]["accuracy"],test_CE=r["test"]["loss"] if r["test"] else None,
            test_accuracy=r["test"]["accuracy"] if r["test"] else None,
            aggregate_prompt_parameters=r["aggregate_prompt_parameters"],gpu=r["gpu_name"],
            activation_effective_bits=r["mean_activation_effective_bits"],
            budget_proxy=r["validation"]["legacy_budget_proxy"]))
    if flat:
        with (root/"per_seed.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0]))
            writer.writeheader(); writer.writerows(flat)
    report=["# 多数据集 LoRAP 完整 GPU 训练", "",
        f"已完成训练 **{fitted}/{total}** 组，最终测试 **{tested}/{total}** 组。"
        f"运行中 {len(running)} 项，失败 {len(failed)} 项。", "",
        "只有训练和最终测试全部完成的行才构成完整结果；运行状态不代表训练已经完成。", "",
        "## 固定协议", "",
        "仅使用 CUDA，没有 CPU 训练降级路径。主实验采用全划分、200 轮、种子 41–50；"
        "具体运行参数以 configs 中的文件为准。Cora/CiteSeer 使用公共节点划分，"
        "MNIST/CIFAR10 使用各自官方完整训练/验证/测试划分。", "",
        "两个节点任务沿用两层、16 维 GIN，仅第二层有聚合提示；两个图任务使用四层、110 维 GIN，"
        "保留输入 GPF-plus，每个聚合提示层 rank 2，20 个提示。CIFAR10 输入为三通道特征加二维位置，"
        "使用 README 的 GIN-CIFAR10 量化学习率与 norm 步长初始化；Cora 使用 README 的 GIN-Cora 存储惩罚。", "",
        "改进版沿用此前零左因子、条件标准化/截断、RMS 查询及幅度缩放、均匀分层 rank；"
        "随机基和误差谱基单列。本轮没有悄悄修正上一轮发现的条件分布变化问题，"
        "因此这是对既定方法的完整验证。离线训练校准和误差谱计算可在 CPU 完成，所有训练更新必须在 GPU。", "",
        "同数据集、同种子共享初始骨干和数据顺序。GPU scatter 的归约可能存在微小数值不确定性，"
        "配对种子不等于 GPU 运算逐位可复现。每轮保存模型、优化器、调度器、随机数和数据顺序状态；"
        "中断后从已提交轮次继续。按最低验证 CE 选模，平局取早。每个数据集全部预定训练完成后，"
        "每个最佳检查点只测试一次；不按测试结果筛选变体。", "",
        "校准只用训练节点/图，最多 4096 节点/层，图任务最多用前 128 张训练图。"
        "局部误差诊断只在验证数据上计算，固定当前层输入，仅绕过当前层消息量化；"
        "不是完整 FP32 模型差异。提示开销和实际有效位宽单独记录，存储惩罚目标不保证预算达成。", "",
        "## 当前运行", "", "| GPU | 数据集 | 方法 | seed | 阶段 | 轮次 |", "|---|---|---|---:|---|---|"]
    for job in running:
        path=root/job["dataset"]/job["variant"]/f"seed_{job['seed']}/progress.json"
        progress=json.loads(path.read_text()) if path.exists() else {}
        epoch=f"{progress.get('epoch',0)}/{progress.get('epochs','?')}" if job["phase"]=="fit" else "—"
        report.append(f"| {job.get('gpu')} | {job['dataset']} | {job['variant']} | {job['seed']} | {job['phase']} | {epoch} |")
    report += ["", "## 完成的结果", "",
        "均值 ± 样本标准差；n 明确给出当前已测试种子数。未完成全部种子时为阶段性结果。", "",
        "| 数据集 | 方法 | 已训练 | 已测试 n | 测试准确率 % | 验证 CE | 聚合提示参数 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for config in study["configs"]:
        for variant in config["variants"]:
            rows=[r for r in completed if r["dataset"]==config["dataset"] and r["variant"]==variant]
            tests=[r for r in rows if r["test"] is not None]
            if tests:
                a=np.array([r["test"]["accuracy"]*100 for r in tests])
                acc=f"{a.mean():.2f} ± {a.std(ddof=1) if len(a)>1 else 0:.2f}"
            else:
                acc="尚未测试"
            val=f"{np.mean([r['validation']['loss'] for r in rows]):.4f}" if rows else "—"
            params=str(rows[0]["aggregate_prompt_parameters"]) if rows else "—"
            report.append(f"| {config['dataset']} | {LABELS[variant]} | {len(rows)}/{len(config['seeds'])} | {len(tests)} | {acc} | {val} | {params} |")
    if failed:
        report += ["", "## 未完成项目", ""]
        for job in failed:
            report.append(f"- {job['dataset']}/{job['variant']}/seed_{job['seed']}/{job['phase']}："
                          f"退出码 {job.get('exit_code')}；见 logs/{job['id']}.log。")
    report += ["", "## 运行与恢复", "",
        "所有工作进程从此次 source_snapshot 执行，后续工作区修改不影响这批训练。"
        "progress.json 记录实时轮次，last.pt 支持续训，best.pt 保存验证最优权重。"
        "对同一结果目录重复恢复不会重复已完成训练或已有测试结果。", "",
        "暂停：在结果根目录创建 STOP 文件，运行中的训练完成当前轮次后暂停。"
        "恢复和查看状态的命令见 RUN_COMMANDS.md。测试开始后若进程异常退出且结果未保存，会停止该项，"
        "不会自动重复测试。", "",
        "结果包含配置、数据划分索引、代码与环境快照、校准、逐轮日志、检查点、局部误差、实际位宽和逐种子 CSV。"
        "没有完成全部实验前，不据阶段性结果宣称方法优越。"]
    recovery=root/"RECOVERY.md"
    if recovery.exists():
        report.extend(["", "## 收尾复核修复", "", recovery.read_text().strip()])
    (root/"REPORT.md").write_text("\n".join(report)+"\n")
    write_json(root/"status.json",dict(fits_completed=fitted,fits_total=total,tests_completed=tested,
        running=running,failed=failed,complete=fitted==total and tested==total,updated_unix=time.time()))
    # Produce standalone figures once all seeds of a dataset have been tested.
    for config in study["configs"]:
        rows=[r for r in completed if r["dataset"]==config["dataset"] and r["test"] is not None]
        if len(rows)!=len(config["seeds"])*len(config["variants"]):
            continue
        path=root/f"{config['dataset']}_comparison.png"
        if not path.exists():
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from .improve_report import figure
            groups=[[r["test"]["accuracy"]*100 for r in rows if r["variant"]==v] for v in config["variants"]]
            fig,ax=plt.subplots(figsize=(7,4))
            ax.bar(range(len(groups)),[np.mean(x) for x in groups],
                   yerr=[np.std(x,ddof=1) if len(x)>1 else 0 for x in groups],capsize=4)
            ax.set_xticks(range(len(groups)),config["variants"],rotation=20,ha="right")
            ax.set_ylabel("Test accuracy (%)")
            ax.set_title(f"{config['dataset']} · full split · {config['epochs']} epochs · {len(config['seeds'])} seeds")
            fig.tight_layout()
            figure(fig,root,f"{config['dataset']}_comparison")
        with (root/"REPORT.md").open("a") as stream:
            stream.write(f"\n![{config['dataset']} 完整结果]({path.name})\n")
