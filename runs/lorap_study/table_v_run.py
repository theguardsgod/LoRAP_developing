"""Full MNIST LoRA rows for fixed INT4 QAT and adaptive A2Q.

One independent fit per GPU at a time. All declared fits in BOTH protocols must
finish before the coordinator releases their validation-selected checkpoints.
"""

import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import torch

from .lora import baseline_config
from .lora_report import generate_report
from .lora_run import fit, release_tests, require_device, study_data, test_once
from .run import ROOT, provenance, write_json

MODES = ("fixed4", "a2q")


def table_config(mode, seeds, epochs, smoke=False):
    config = baseline_config("MNIST", mode)
    config.update(variants=["lora"], seeds=seeds, epochs=epochs,
                  subset_sizes=dict(train=128, val=64, test=64) if smoke else None,
                  profile=f"table_v_lora_{mode}", quantization_mode=mode)
    return config


def summarize(root):
    study = json.loads((root / "suite.json").read_text())
    rows, progress = [], []
    for mode in MODES:
        subroot = root / mode
        generate_report(subroot)
        status = json.loads((subroot / "status.json").read_text())
        results = json.loads((subroot / "results.json").read_text())
        tested = [r for r in results if r["test_evaluations"] == 1]
        ready = len(tested) == len(study["seeds"]) and not study["smoke"]
        rows.append(dict(protocol=mode, fits=status["fits_completed"], tests=len(tested),
                         seeds=study["seeds"], ready=ready,
                         additional_parameters=4426, fp32_bytes=17704, additional_KiB=17704 / 1024,
                         mean_accuracy_percent=statistics.mean(r["test"]["accuracy"] * 100 for r in tested) if ready else None,
                         sample_std_accuracy_percent=statistics.stdev(r["test"]["accuracy"] * 100 for r in tested) if ready and len(tested)>1 else None,
                         average_activation_effective_bits=statistics.mean(r["mean_activation_effective_bits"] for r in results) if results else None))
        for seed in study["seeds"]:
            path = subroot / "MNIST/lora" / f"seed_{seed}" / "progress.json"
            if path.exists():
                progress.append(dict(protocol=mode, seed=seed, **json.loads(path.read_text())))
    complete = all(r["ready"] for r in rows) if not study["smoke"] else all(r["fits"] == len(study["seeds"]) for r in rows)
    write_json(root / "status.json", dict(complete=complete, smoke=study["smoke"], rows=rows, progress=progress))
    write_json(root / "table_rows.json", rows)
    tex = ["% Insert each row inside its named block in Table V.",
           "% Validation-selected checkpoints; mean +/- SAMPLE std across declared seeds.",
           "% Average bits: effective activation bits weighted by elements observed on validation.",
           "% Confirm the original table uses this bit metric before combining rows.",
           "% KiB = 1024 bytes. LoRA factors are FP32 and remain unmerged."]
    report = ["# Table V：QAT 与 A2Q 的标准权重 LoRA", "",
              f"完成状态：{'已完成' if complete else '进行中'}。训练 {sum(r['fits'] for r in rows)}/{2*len(study['seeds'])}，"
              f"测试 {sum(r['tests'] for r in rows)}/{0 if study['smoke'] else 2*len(study['seeds'])}。", "",
              f"MNIST {'冒烟子集' if study['smoke'] else '官方完整划分'}，{study['epochs']} 轮，种子 {study['seeds']}，batch size 128；"
              "4 层 GIN，宽度 110。两组均移除全部提示，在 11 个量化线性层加入 rank=2、alpha=2 的 FP32 LoRA。", "",
              "| Table V 分组 | 配置 | Average bits¹ | 额外参数 | MNIST Acc. % | 已训练 / 已测试 |",
              "|---|---|---|---|---|---|"]
    for row in rows:
        label = "INT4 / QAT" if row["protocol"] == "fixed4" else "A2Q"
        bits = "4.00" if row["protocol"] == "fixed4" else (f"{row['average_activation_effective_bits']:.4f}" if row["ready"] else "待测")
        acc = (f"{row['mean_accuracy_percent']:.2f} ± {row['sample_std_accuracy_percent']:.2f}"
               if row["ready"] and row["sample_std_accuracy_percent"] is not None else "待测")
        report.append(f"| {label} | + LoRA | {bits} | 4,426 / 17.29 KiB | {acc} | {row['fits']} / {row['tests']} |")
        tex += [f"% {label} block", "$+\\,\\mathrm{LoRA}\\;(r=2)$ & " + (bits if bits != "待测" else "\\textit{TBD}") +
                " & $17.29\\,\\mathrm{KiB}$ & " + ("$" + acc.replace("±", "\\pm") + "$" if acc != "待测" else "\\textit{TBD}") + r" \\"]
    report += ["", "¹ 位宽按最佳检查点在完整验证集上实际访问的激活元素加权统计；非负特征扣除内部代理符号位，"
               "权重位宽另见各种子的 quantization.json。FP32 LoRA 开销单列，不混入量化骨干位宽。"
               "原截图的 Average bits 定义尚未确认，A2Q 数值不能直接假定与原表 3.75 同口径。", "",
               "固定 INT4：使用 u_quant_normal_debug.QLinear，所有位宽冻结，CE 损失；"
               "尺度学习率及初始化取自记录 94.4 ± 0.2 的 GIN_normal_MNIST 日志。"
               "A2Q：沿用既有自适应特征位宽和存储惩罚配置。两组量化尺度与骨干均可训练，LoRA 学习率 0.001、权重衰减 0。", "",
               "选模：验证交叉熵最低，平局取较早轮次；两组全部预定训练完成后才释放测试，各最佳检查点仅测试一次。"
               "旧脚本逐轮取最高测试准确率，且旧表 GPF 参数量与当前输入 GPF 的实现存在差异，"
               "因此这些新增结果应注明评估协议；原表其余数字未改写。", "",
               "所有执行来自 source_snapshot，配置、划分、逐轮日志、初始化、检查点、位宽、资源耗时及测试记录"
               "分别位于 fixed4/ 和 a2q/。进度见 status.json；LaTeX 插入行见 table_v_lora_rows.tex。"]
    if study["smoke"]:
        report += ["", "本次仅为冒烟检查，不输出测试准确率，不能填入论文结果。"]
    (root / "REPORT.md").write_text("\n".join(report) + "\n")
    (root / "table_v_lora_rows.tex").write_text("\n".join(tex) + "\n")
    with (root / "table_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def worker(root, worker_index, phase):
    suite = json.loads((root / "suite.json").read_text())
    torch.set_num_threads(suite["threads"])
    device = require_device(suite["devices"][worker_index])
    configs = {m: json.loads((root / m / "MNIST/config.json").read_text()) for m in MODES}
    data, _ = study_data(configs[MODES[0]])
    jobs = [(mode, seed) for seed in suite["seeds"] for mode in MODES]
    jobs = jobs[worker_index::len(suite["devices"])]
    with (root / f"worker_{worker_index}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for mode, seed in jobs:
            function = fit if phase == "fit" else test_once
            function(root / mode, configs[mode], "lora", seed, data, None, device)


def phase_workers(root, phase):
    suite = json.loads((root / "suite.json").read_text())
    processes, streams = [], []
    try:
        for index in range(len(suite["devices"])):
            stream = (root / f"{phase}_worker_{index}.log").open("a")
            streams.append(stream)
            command = [sys.executable, "-B", "-m", "runs.lorap_study.table_v_run", "--worker-root", str(root),
                       "--worker-index", str(index), "--phase", phase]
            processes.append(subprocess.Popen(command, cwd=root / "source_snapshot", stdout=stream, stderr=subprocess.STDOUT))
        write_json(root / "workers.json", dict(phase=phase, pids=[p.pid for p in processes], devices=suite["devices"]))
        while any(p.poll() is None for p in processes):
            summarize(root)
            time.sleep(10)
        codes = [p.wait() for p in processes]
        if any(codes):
            raise RuntimeError(f"{phase} worker exit codes: {codes}; inspect {phase}_worker_*.log")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        for stream in streams:
            stream.close()
    summarize(root)


def coordinator(root):
    suite = json.loads((root / "suite.json").read_text())
    with (root / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phase_workers(root, "fit")
        if not suite["smoke"]:
            # No worker reads test metrics before BOTH protocols have all fits.
            for mode in MODES:
                config = json.loads((root / mode / "MNIST/config.json").read_text())
                release_tests(root / mode, config)
            write_json(root / "test_release.json", dict(modes=list(MODES), seeds=suite["seeds"]))
            phase_workers(root, "test")
        summarize(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(41, 51)))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--coordinator-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("fit", "test"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_root:
        worker(args.worker_root, args.worker_index, args.phase)
        return
    if args.coordinator_root:
        coordinator(args.coordinator_root)
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    root = args.output_dir.resolve()
    if not args.resume:
        if min(args.epochs, args.threads) < 1 or len(set(args.seeds)) != len(args.seeds):
            parser.error("Positive epochs/threads and unique seeds required")
        if len(set(args.devices)) != len(args.devices):
            parser.error("Devices must be unique")
        for name in args.devices:
            require_device(name)
        torch.set_num_threads(args.threads)
        seeds = args.seeds[:1] if args.smoke else args.seeds
        epochs = 2 if args.smoke else args.epochs
        devices = args.devices[:min(len(args.devices), 2 * len(seeds))]
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "suite.json", dict(seeds=seeds, epochs=epochs, devices=devices,
                   threads=args.threads, smoke=args.smoke, modes=list(MODES), protocol_version=1))
        provenance(root)
        # Preserve the historical configuration evidence too; these are never executed.
        record = json.loads((root / "provenance.json").read_text())
        for relative in ("lr_gin_mnist_normal.py", "lr_gin_mnist_bit.py",
                         "result/GIN_normal_MNIST/GIN_110__on_MNIST_4bit-200.txt",
                         "result/GIN_MNIST/GIN_110__on_MNIST_4bit-200.txt"):
            source = ROOT / relative
            destination = root / "protocol_evidence" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            record.setdefault("protocol_evidence_sha256", {})[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        write_json(root / "provenance.json", record)
        for mode in MODES:
            config = table_config(mode, seeds, epochs, args.smoke)
            subroot = root / mode
            write_json(subroot / "MNIST/config.json", config)
            write_json(subroot / "study.json", dict(configs=[config], smoke=args.smoke, method=f"{mode} + LoRA"))
            (subroot / "source_snapshot").symlink_to("../source_snapshot", target_is_directory=True)
            command = f"{sys.executable} -B -m runs.lorap_study.table_v_run --resume --output-dir {root}"
            (subroot / "RUN_COMMANDS.md").write_text(command + "\n")
            data, manifest = study_data(config)
            write_json(subroot / "MNIST/split_manifest.json", manifest)
            del data
        (root / "RUN_COMMANDS.md").write_text(command + "\n")
        summarize(root)
    elif not (root / "suite.json").exists():
        parser.error("No saved suite to resume")
    command = [sys.executable, "-B", "-m", "runs.lorap_study.table_v_run", "--coordinator-root", str(root)]
    subprocess.run(command, cwd=root / "source_snapshot", check=True)


if __name__ == "__main__":
    main()
