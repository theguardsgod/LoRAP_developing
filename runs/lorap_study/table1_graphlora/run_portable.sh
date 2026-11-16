#!/usr/bin/env bash
# Usage: bash run_portable.sh OUTPUT_ROOT [cora|focal] [DATA_ROOT]
set -euo pipefail
PORTABLE_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PORTABLE_PYTHON="${PYTHON:-python}"
cd "$PORTABLE_REPO"
PORTABLE_ROOT="$("$PORTABLE_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${1:-result/graphlora_repair_portable}")"
PORTABLE_PROFILE="${2:-cora}"
PORTABLE_DATA="$("$PORTABLE_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${3:-data}")"
PORTABLE_SMOKE="${PORTABLE_ROOT}_smoke"
PORTABLE_DEVICE="${DEVICE:-cuda:0}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PORTABLE_PYTHON" -B -m unittest runs.lorap_study.test_graphlora_stability \
    runs.lorap_study.test_graphlora runs.lorap_study.table1_graphlora.test_portable -v
"$PORTABLE_PYTHON" -c 'import sys,torch,torch_scatter; from torch_scatter import segment_csr; assert torch.cuda.is_available(), "CUDA required"; d=torch.device(sys.argv[1]); x=torch.ones(2,device=d); print(torch.__version__,torch.cuda.get_device_name(d),segment_csr(x,torch.tensor([0,2],device=d)))' "$PORTABLE_DEVICE"
for PORTABLE_EXISTING in "$PORTABLE_SMOKE" "$PORTABLE_ROOT"; do
    if [[ -f "$PORTABLE_EXISTING/study.json" ]]; then
        "$PORTABLE_PYTHON" -c 'import json,sys; from pathlib import Path; from runs.lorap_study.table1_graphlora.portable import PROFILES; s=json.loads((Path(sys.argv[1])/"study.json").read_text()); assert s["cells"]==PROFILES[sys.argv[2]], "Profile changed: use a new output root"; assert all(Path(j["config"]["data_root"])==Path(sys.argv[3])/j["config"]["dataset"].lower() for j in s["jobs"]), "Data root changed: use a new output root"' "$PORTABLE_EXISTING" "$PORTABLE_PROFILE" "$PORTABLE_DATA"
    fi
done
if [[ ! -f "$PORTABLE_SMOKE/study.json" ]]; then
    "$PORTABLE_PYTHON" -B -m runs.lorap_study.table1_graphlora.portable prepare \
        --root "$PORTABLE_SMOKE" --data-root "$PORTABLE_DATA" --profile "$PORTABLE_PROFILE" --smoke
fi
cd "$PORTABLE_SMOKE/source_snapshot"
"$PORTABLE_PYTHON" -B -m runs.lorap_study.table1_graphlora.portable worker --root "$PORTABLE_SMOKE" --device "$PORTABLE_DEVICE"
"$PORTABLE_PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); a=json.loads((p/"completion_audit.json").read_text()); assert a["passed"] and a["tests"]==0' "$PORTABLE_SMOKE"
cd "$PORTABLE_REPO"
if [[ ! -f "$PORTABLE_ROOT/study.json" ]]; then
    "$PORTABLE_PYTHON" -B -m runs.lorap_study.table1_graphlora.portable prepare \
        --root "$PORTABLE_ROOT" --data-root "$PORTABLE_DATA" --profile "$PORTABLE_PROFILE"
    "$PORTABLE_PYTHON" -m pip freeze > "$PORTABLE_ROOT/environment.freeze.txt"
fi
cd "$PORTABLE_ROOT/source_snapshot"
"$PORTABLE_PYTHON" -B -m runs.lorap_study.table1_graphlora.portable worker --root "$PORTABLE_ROOT" --device "$PORTABLE_DEVICE"
