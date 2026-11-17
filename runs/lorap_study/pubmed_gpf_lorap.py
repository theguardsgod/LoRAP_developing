"""PubMed A2Q None/GPF-LoRAP paired experiments, seeds 41/42/43."""
import argparse
from pathlib import Path

from .table1_frozen.registry import CELLS as TABLE1
from .table1_frozen.common import write_json
from .table1_graphlora import portable as engine

ARCHITECTURES = ('GIN', 'GCN', 'GAT')


def configurations():
    result = {}
    for architecture in ARCHITECTURES:
        c = dict(TABLE1[f'A2Q_CiteSeer_{architecture}'])
        c.update(cell=f'A2Q_PubMed_{architecture}', dataset='PubMed',
                 paper_none=None, paper_none_sd=None, paper_gpf_lorap=None,
                 recipe_note='PubMed recipe from residual_transfer/config.py; joint GPF-LoRAP implementation from table1_gpf_rerun/model.py')
        if architecture == 'GIN':
            c.update(a_loss=.1)
        elif architecture == 'GCN':
            c.update(lr=.01, weight_decay=.01, dropout=.5, lr_feature=.005,
                     lr_weight=.005, lr_message=.005, lr_bit=.02, a_loss=.1)
        else:
            c.update(lr=.005, lr_feature=.005, lr_weight=.001, lr_message=.005,
                     lr_bit=.015, a_loss=.025, attention_dropout=.3)
        result[c['cell']] = c
    return result


def prepare(root, data_root, seeds=None, smoke=False):
    if not smoke and seeds is not None and sorted(seeds) != [41, 42, 43]:
        raise ValueError('Formal PubMed results require seeds 41, 42, and 43')
    configs = configurations()
    return engine.prepare(root, data_root, list(configs), seeds, smoke,
                          adaptation='gpf_lorap', cell_configs=configs)


def report(root):
    root = Path(root).resolve()
    study = engine.read(root / 'study.json')
    engine.generate(root, study)
    rows = engine.read(root / 'summary.json')
    result, tex = [], []
    for arch in ARCHITECTURES:
        cell = f'A2Q_PubMed_{arch}'
        methods = {r['variant']: r for r in rows if r['cell'] == cell}
        gpf = methods['gpf_lorap']
        complete = not study['smoke'] and all(r['tests'] == len(study['seeds']) for r in methods.values())
        text = f"{gpf['test_mean']:.2f} ± {gpf['test_sd']:.2f}" if complete and gpf['test_sd'] is not None else 'TBD'
        result.append(dict(dataset='PubMed', architecture=arch, protocol='joint', seeds=study['seeds'],
                           none=methods['none'], gpf_lorap=gpf, table_value=text))
        value = '$' + text.replace(' ± ', r'\pm') + '$' if text != 'TBD' else 'TBD'
        tex.append(f'PubMed & {arch} & {value}' + r' \\')
    write_json(root / 'pubmed_results.json', result)
    (root / 'pubmed_gpf_lorap_rows.tex').write_text(
        '% PubMed A2Q, joint training, three-seed sample SD; use actual paired None from summary.csv.\n' + '\n'.join(tex) + '\n')
    print('\n'.join(f"{r['architecture']}: {r['table_value']}" for r in result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'worker', 'report'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.root, a.data_root, a.seeds, a.smoke)
    elif a.action == 'worker':
        try:
            engine.worker(a.root, a.device)
        finally:
            report(a.root)
    else:
        report(a.root)


if __name__ == '__main__':
    main()
