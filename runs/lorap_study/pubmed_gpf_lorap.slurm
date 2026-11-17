#!/bin/bash
#SBATCH --job-name=PubMed-GPF-LoRAP
#SBATCH --account=en
#SBATCH --partition=h07gpuq1
#SBATCH --qos=h07gpuq1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=result/lorap_study/pubmed-gpf-%j.out
#SBATCH --error=result/lorap_study/pubmed-gpf-%j.err
set -euo pipefail
PUBMED_REPO="${SLURM_SUBMIT_DIR:?Submit from repository root}"
PUBMED_PYTHON="$PUBMED_REPO/.venv/bin/python"
PUBMED_ROOT="$PUBMED_REPO/result/lorap_study/pubmed_gpf_lorap_20261117_v1"
PUBMED_SMOKE="${PUBMED_ROOT}_smoke"
export PUBMED_DATA_ROOT="$PUBMED_REPO/data/pubmed_ready"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$PUBMED_REPO/.toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$PUBMED_ROOT/source_snapshot"
"$PUBMED_PYTHON" -B -m unittest runs.lorap_study.test_pubmed_gpf_lorap -v
cd "$PUBMED_SMOKE/source_snapshot"
"$PUBMED_PYTHON" -B -m runs.lorap_study.pubmed_gpf_lorap worker --root "$PUBMED_SMOKE" --device cuda:0
"$PUBMED_PYTHON" -c 'import json,sys; from pathlib import Path; a=json.loads((Path(sys.argv[1])/"completion_audit.json").read_text()); assert a["passed"] and a["tests"]==0' "$PUBMED_SMOKE"
cd "$PUBMED_ROOT/source_snapshot"
"$PUBMED_PYTHON" -B -m runs.lorap_study.pubmed_gpf_lorap worker --root "$PUBMED_ROOT" --device cuda:0
"$PUBMED_PYTHON" -c 'import json,sys; from pathlib import Path; a=json.loads((Path(sys.argv[1])/"completion_audit.json").read_text()); assert a["passed"] and a["fits"]==18 and a["tests"]==18' "$PUBMED_ROOT"
