"""One-time source snapshot; never executes historical training entrypoints."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DQROOT = Path('/home/cyliu/code/PromptRepDQ')


def main():
    v = HERE/'vendor'
    v.mkdir(exist_ok=True)
    (v/'__init__.py').touch()
    manifest = []
    for name in ('quantization','linear_quantized','baseline_quant','multi_quant'):
        source = DQROOT/'dq'/f'{name}.py'
        text = source.read_text().replace('from dq.', 'from .')
        target = v/f'{name}.py'
        target.write_text(text)
        manifest.append(dict(source=str(source), target=str(target.relative_to(HERE)),
                             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), change='relative imports only'))
    source = Path('/home/cyliu/code/degree-quant/LICENSE')
    (v/'DQ_LICENSE').write_bytes(source.read_bytes())
    sources = dict(a2q_node='node_level_1.py', a2q_node_gat='gat_nc_lsb.py',
                   a2q_graph_gin='lr_gin_mnist_bit.py', a2q_graph_gcn='lr_gcn_mnist_bit.py',
                   a2q_graph_gat='lr_gat_mnist.py', a2q_zinc_gin='runs/zinc/lr_gin_zinc_bit.py',
                   a2q_zinc_gcn='lr_gcn_zinc_bit.py')
    for name, relative in sources.items():
        source = ROOT/relative
        tree = ast.parse(source.read_text())
        tree.body = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef))]
        # No dataset load, optimizer, parser, training, printing, or telemetry survives.
        target = v/f'{name}.py'
        target.write_text('"""Snapshot of model definitions from '+relative+'."""\n'+ast.unparse(tree)+'\n')
        manifest.append(dict(source=str(source), target=str(target.relative_to(HERE)),
                             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), change='AST imports/classes only'))
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()
