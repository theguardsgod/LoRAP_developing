"""Read-only result/checkpoint audit; never runs model inference or test data."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source_snapshot'))
import torch
from torch_geometric.data import Data
from runs.lorap_study.full_run import sha_config
from runs.lorap_study.full_validation import verify_checkpoint, validation_reproduction
from runs.lorap_study.lora import build_baseline, baseline_counts, common_backbone_digest
from runs.lorap_study.models import state_digest

torch.set_num_threads(2)
provenance = json.loads((ROOT / 'provenance.json').read_text())
for relative, expected in provenance['source_sha256'].items():
    assert hashlib.sha256((ROOT / 'source_snapshot' / relative).read_bytes()).hexdigest() == expected, relative
for relative, expected in provenance['protocol_evidence_sha256'].items():
    assert hashlib.sha256((ROOT / 'protocol_evidence' / relative).read_bytes()).hexdigest() == expected, relative
suite = json.loads((ROOT / 'suite.json').read_text())
rows = {r['protocol']: r for r in json.loads((ROOT / 'table_rows.json').read_text())}
assert suite['seeds'] == list(range(41, 51)) and suite['epochs'] == 200
records, markers, fit_times = [], [], []
reference_split = None
for mode in ('fixed4', 'a2q'):
    config = json.loads((ROOT / mode / 'MNIST/config.json').read_text())
    config_hash = sha_config(config)
    split = json.loads((ROOT / mode / 'MNIST/split_manifest.json').read_text())
    assert {s: len(v) for s, v in split['indices'].items()} == dict(train=55000, val=5000, test=10000)
    assert all(v == list(range(len(v))) for v in split['indices'].values())
    if reference_split is not None:
        assert split == reference_split
    reference_split = split
    release = json.loads((ROOT / mode / 'MNIST/test_release.json').read_text())
    assert release['config_sha256'] == config_hash and release['fits'] == 10
    mode_results = []
    for seed in suite['seeds']:
        directory = ROOT / mode / 'MNIST/lora' / f'seed_{seed}'
        result = json.loads((directory / 'result.json').read_text())
        initial = json.loads((directory / 'initialization.json').read_text())
        checkpoint = torch.load(directory / 'best.pt', weights_only=False, map_location='cpu')
        selected = verify_checkpoint(directory, checkpoint, config_hash)
        assert result['epochs'] == 200 and result['config_sha256'] == config_hash
        assert result['best_epoch'] == selected['epoch']
        validation_reproduction(checkpoint['validation_loss'], result['validation']['loss'])
        assert result['test_evaluations'] == 1 and result['test']['examples'] == 10000
        assert result['validation']['examples'] == 5000 and result['device'] == 'cuda:0'
        assert all(math.isfinite(result['test'][k]) for k in ('loss', 'accuracy'))
        marker = json.loads((directory / 'test_started.json').read_text())
        assert marker['checkpoint_epoch'] == checkpoint['epoch']
        markers.append(marker['started_unix'])
        fit_times.append((directory / 'quantization.json').stat().st_mtime)
        # Model construction depends only on input dimensions, not graph contents.
        example = Data(x=torch.zeros(1, 1), pos=torch.zeros(1, 2))
        model = build_baseline(config, 'lora', seed, {'train': [example]})
        assert common_backbone_digest(model) == initial['common_backbone_sha256']
        assert state_digest(model, 'all') == initial['initial_state_sha256']
        counts = baseline_counts(model)
        assert all(initial[k] == value for k, value in counts.items())
        assert counts['prompt_parameters'] == 0 and counts['lora_parameters'] == 4426
        assert counts['lora_fp32_bytes'] == 17704 and len(counts['lora_targets']) == 11
        frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
        initial_state = model.state_dict()
        assert all(torch.equal(initial_state[name], checkpoint['model'][name]) for name in frozen)
        changed = {name for name, p in model.named_parameters()
                   if not torch.equal(p, checkpoint['model'][name])}
        for suffix in ('lora_A', 'lora_B', '.weight', 'fea_quant.gama', 'weight_quant.alpha', 'q_xw.alpha'):
            assert any(name.endswith(suffix) for name in changed), (mode, seed, suffix)
        if mode == 'fixed4':
            assert all(not p.requires_grad for name, p in model.named_parameters() if name.endswith('.bit'))
            assert result['mean_activation_effective_bits'] == 4
        else:
            assert any(name.endswith('fea_quant.bit') for name in changed)
        records.append(dict(mode=mode, seed=seed, best_epoch=checkpoint['epoch'],
                            test_accuracy_percent=100 * result['test']['accuracy'],
                            frozen_tensors_checked=len(frozen), training_seconds=result['training_seconds'],
                            peak_cuda_memory_bytes=result['peak_cuda_memory_bytes']))
        mode_results.append(result)
    accuracy = [100 * r['test']['accuracy'] for r in mode_results]
    assert math.isclose(rows[mode]['mean_accuracy_percent'], statistics.mean(accuracy), abs_tol=1e-12)
    assert math.isclose(rows[mode]['sample_std_accuracy_percent'], statistics.stdev(accuracy), abs_tol=1e-12)
    assert math.isclose(rows[mode]['average_activation_effective_bits'],
                        statistics.mean(r['mean_activation_effective_bits'] for r in mode_results), abs_tol=1e-12)
assert len(records) == 20 and max(fit_times) < min(markers)
assert json.loads((ROOT / 'status.json').read_text())['complete']
audit = dict(passed=True, audited_unix=time.time(), source_hashes_checked=len(provenance['source_sha256']),
             evidence_hashes_checked=len(provenance['protocol_evidence_sha256']), runs=records,
             fits_completed=20, tests_completed=20, additional_test_evaluations_during_audit=0,
             all_fits_preceded_first_test=True, exact_earliest_validation_minima_verified=True,
             initialization_reconstructed=True, frozen_parameters_verified=True,
             aggregate_statistics_verified=True, training_hours=sum(r['training_seconds'] for r in records)/3600,
             audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT / 'completion_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({k: v for k, v in audit.items() if k != 'runs'}, indent=2))
