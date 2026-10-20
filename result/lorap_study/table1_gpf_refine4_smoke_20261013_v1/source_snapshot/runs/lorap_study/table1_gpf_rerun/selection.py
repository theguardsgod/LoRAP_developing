"""Parse the supplied table; register every requested underperforming cell."""
import re

DATASETS = ('Cora', 'CiteSeer', 'MNIST', 'CIFAR10', 'ZINC')


def parse_table(text):
    rows = {}
    blocks = re.split(r'%\s*(QAT|DQ|A2Q)\s*\+\s*(GIN|GCN|GAT)', text)
    for i in range(1, len(blocks), 3):
        framework, architecture, body = blocks[i:i + 3]
        for name, marker in (('gpf_lorap', 'GPF-LoRAP'), ('graphlora', r'\revision{GraphLoRA}')):
            if marker not in body:
                raise ValueError(f'Missing {name} in {framework} {architecture}')
            tail = body.split(marker, 1)[1].split(r'\\', 1)[0]
            values = re.findall(r'(\d+(?:\.\d+)?)\s*\\pm\s*(\d+(?:\.\d+)?)', tail)
            if len(values) != 5:
                raise ValueError(f'Expected five metrics for {framework} {architecture} {name}')
            for dataset, (mean, sd) in zip(DATASETS, values):
                key = f'{framework}_{dataset}_{architecture}'
                rows.setdefault(key, {})[name] = dict(mean=float(mean), sd=float(sd),
                    decimals=len(mean.partition('.')[2]))
    if len(rows) != 45:
        raise ValueError(f'Expected 45 table cells, got {len(rows)}')
    return rows


def select(rows):
    return [key for key, row in rows.items() if
            (row['gpf_lorap']['mean'] > row['graphlora']['mean'] if '_ZINC_' in key else
             row['gpf_lorap']['mean'] < row['graphlora']['mean'])]
