"""Audit completed baseline artifacts; never execute a test-set evaluation."""
import hashlib
import json
from pathlib import Path
import sys

import torch

from runs.lorap_study.full_run import sha_config
from runs.lorap_study.full_validation import verify_checkpoint, VALIDATION_REPLAY_ATOL
from runs.lorap_study.lora import baseline_counts, build_baseline, common_backbone_digest
from runs.lorap_study.lora_run import study_data

root = Path(sys.argv[1]).resolve()
study = json.loads((root / 'study.json').read_text())
status = json.loads((root / 'status.json').read_text())
assert status['complete'] and status['fits_completed'] == 27 and status['tests_completed'] == 27
provenance = json.loads((root / 'provenance.json').read_text())
for name, expected in provenance['source_sha256'].items():
    assert hashlib.sha256((root / 'source_snapshot' / name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((Path.cwd() / name).read_bytes()).hexdigest() == expected, ('workspace drift', name)
torch.set_num_threads(4)
rows = []
for config in study['configs']:
    data, manifest = study_data(config)
    assert json.loads((root / config['dataset'] / 'split_manifest.json').read_text()) == manifest
    release = json.loads((root / config['dataset'] / 'test_release.json').read_text())
    assert release['config_sha256'] == sha_config(config) and release['fits'] == 9
    for seed in config['seeds']:
        digests = []
        calibration = torch.load(root / config['dataset'] / f'calibration_seed_{seed}.pt',
                                 map_location='cpu', weights_only=False)
        assert all(c['split'] == 'train' and not c['fp_reference_used'] for c in calibration)
        for variant in config['variants']:
            directory = root / config['dataset'] / variant / f'seed_{seed}'
            result = json.loads((directory / 'result.json').read_text())
            initial = json.loads((directory / 'initialization.json').read_text())
            checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
            selected = verify_checkpoint(directory, checkpoint, sha_config(config))
            last = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
            assert last['epoch'] == config['epochs'] == len(last['history'])
            assert result['best_epoch'] == selected['epoch']
            assert result['test_evaluations'] == 1 and result['test'] is not None
            assert (directory / 'test_started.json').exists()
            assert abs(result['validation']['loss'] - selected['validation']['loss']) <= VALIDATION_REPLAY_ATOL
            assert all(0 <= result[split]['accuracy'] <= 1 for split in ('validation', 'test'))
            assert result['device'] == 'cuda:0' and result['peak_cuda_memory_bytes'] > 0
            assert result['test']['examples'] == (256 if config['task'] == 'graph' else 1000)
            model = build_baseline(config, variant, seed, data, calibration)
            digests.append(common_backbone_digest(model))
            assert digests[-1] == initial['common_backbone_sha256']
            counts = baseline_counts(model)
            assert all(initial[key] == value == result[key] for key, value in counts.items())
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    assert torch.equal(p.detach().cpu(), checkpoint['model'][name]), ('frozen parameter changed', name)
            optimizer_groups = json.loads((directory / 'optimizer_groups.json').read_text())
            assert sum(g['parameters'] for g in optimizer_groups) == counts['trainable_parameters']
            if variant != 'main':
                assert counts['prompt_parameters'] == 0
            if variant == 'lora':
                assert len(counts['lora_targets']) == (11 if config['task'] == 'graph' else 2)
                assert all(t.dtype == torch.float32 for n, t in checkpoint['model'].items()
                           if n.endswith(('lora_A', 'lora_B')))
                groups = [g for g in optimizer_groups if g['name'] == 'lora']
                assert len(groups) == 1 and groups[0]['weight_decay'] == 0 and groups[0]['lr'] == config['lr']
            model.load_state_dict(checkpoint['model'])
            rows.append(dict(dataset=config['dataset'], variant=variant, seed=seed,
                             best_epoch=selected['epoch'], passed=True))
        assert len(set(digests)) == 1
audit = dict(passed=True, runs=len(rows), source_files=len(provenance['source_sha256']),
             checks=['source and workspace hashes', 'official split manifests', 'training-only calibration',
                     'paired initialization', 'exact earliest minimum-validation checkpoints', 'frozen parameters',
                     'zero prompt parameters in controls', 'FP32 LoRA factors', 'optimizer parameter counts',
                     '27 CUDA fits and exactly-once test records', 'validation replay tolerance'],
             test_evaluations_performed_by_audit=0, rows=rows)
(root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({k: v for k, v in audit.items() if k != 'rows'}, indent=2))
