"""Small artifact-only progress view while the immutable GPU queue runs."""

import argparse
import json
from pathlib import Path

from .run import ROOT


def progress(root):
    status=json.loads((root/"status.json").read_text())
    text=["# 门控适配进度", "",
        f"已完成 {status['complete']}/{status['total']} 组，失败 {len(status['failed'])} 组。", "",
        "| 数据集 | 变体 | 种子 | 阶段 | 轮次 | 当前最佳轮次 |", "|---|---|---:|---|---:|---:|"]
    compact=[]
    for row in status["running"]:
        j,p=row["job"],row["progress"] or {}
        text.append(f"| {j['dataset']} | {j['variant']} | {j['seed']} | {p.get('phase','loading')} | {p.get('epoch','—')}/20 | {p.get('best_epoch','—')} |")
        compact.append(f"{j['dataset']}/{j['variant']}/{j['seed']} {p.get('phase','loading')} {p.get('epoch','—')}/20 best={p.get('best_epoch','—')}")
    text.extend(["", "训练使用完整训练划分。以上为中间状态；最终结论以全部种子完成后的 REPORT.md 为准。", "",
                 f"状态记录时间（Unix）：{status['updated_unix']}。"])
    (root/"PROGRESS.md").write_text("\n".join(text)+"\n")
    print(f"complete={status['complete']}/{status['total']} failed={len(status['failed'])}")
    print("\n".join(compact))
    if status["failed"]:
        print(json.dumps(status["failed"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    progress(parser.parse_args().root)
