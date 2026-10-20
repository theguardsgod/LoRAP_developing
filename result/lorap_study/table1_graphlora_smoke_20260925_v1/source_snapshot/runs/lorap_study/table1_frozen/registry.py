"""Paper numbers transcribed from Table I, page 8 of the attached PDF."""
from itertools import product

DATASETS = ('Cora', 'CiteSeer', 'MNIST', 'CIFAR10', 'ZINC')
ARCHITECTURES = ('GIN', 'GCN', 'GAT')
FRAMEWORKS = ('QAT', 'DQ', 'A2Q')
METHODS = ('lorap', 'gpf', 'gpf_lorap', 'conditioned', 'spectral',
           'task', 'error', 'mlp', 'weight_lora', 'gate_only')
ROOTS = dict(Cora='/data/cyliu/dataset/Cora', CiteSeer='/data/cyliu/dataset/CiteSeer',
             MNIST='/data/cyliu/dataset/gnn-mnist', CIFAR10='/data/cyliu/dataset/gnn-cifar10',
             ZINC='/data/cyliu/dataset/zinc')
NONE = {
 'QAT': [[44.2,18.7,91.0,48.7,.555], [66.4,63.4,70.6,38.5,.707], [52.8,30.1,76.3,41.9,.702]],
 'DQ': [[61.0,58.4,92.4,50.7,.540], [72.5,68.6,87.0,49.5,.657], [68.7,41.9,90.3,54.2,.556]],
 'A2Q': [[77.4,66.6,95.7,54.9,.400], [76.1,69.7,90.5,52.4,.504], [82.6,71.2,92.5,61.9,.674]]}
NONE_SD = {
 'QAT': [[1.5,3.7,.5,1.7,.024], [7.,2.2,2.4,1.3,.056], [.7,14.5,1.2,1.1,.011]],
 'DQ': [[6.9,2.1,.4,1.6,.019], [3.7,3.3,1.2,.8,.032], [7.5,5.9,.5,.5,.022]],
 'A2Q': [[.9,.1,.2,.9,.016], [.3,1.6,.5,.8,.009], [.8,.9,.6,1.2,.007]]}
LORAP = {
 'QAT': [[45.1,22.8,91.8,55.8,.528], [71.3,63.9,73.6,43.6,.690], [66.8,40.6,78.2,43.8,.685]],
 'DQ': [[68.8,64.6,92.9,53.0,.512], [78.0,71.0,92.4,53.4,.570], [72.5,54.4,94.0,57.5,.508]],
 'A2Q': [[78.5,70.7,96.4,57.4,.361], [79.1,71.8,91.8,53.5,.482], [83.6,72.5,93.1,62.9,.628]]}


def cell_config(dataset, architecture, framework):
    i, j = ARCHITECTURES.index(architecture), DATASETS.index(dataset)
    node = j < 2
    c = dict(cell=f'{framework}_{dataset}_{architecture}', dataset=dataset, architecture=architecture,
             framework=framework, data_root=ROOTS[dataset], node=node, regression=dataset=='ZINC',
             width=(8 if architecture=='GAT' else 16) if node else
                   (110 if architecture=='GIN' else ((145 if dataset=='ZINC' else 146) if architecture=='GCN' else (18 if dataset=='ZINC' else 19))),
             heads=8, layers=2 if node else 4, bit=4, num_deg=1000, batch_size=128,
             paper_none=NONE[framework][i][j], paper_none_sd=NONE_SD[framework][i][j],
             paper_gpf_lorap=LORAP[framework][i][j], metric='mae' if dataset=='ZINC' else 'accuracy_percent',
             baseline_epochs=250 if node and architecture=='GAT' else 200,
             adapter_epochs=200 if node else 100, seeds=[41,42,43],
             lr=.001, weight_decay=4e-5, dropout=.5 if node else 0., init='norm',
             lr_feature=.01, lr_message=.01, lr_weight=.001, lr_bit=.001,
             lr_attention=.001, lr_edge=.1, a_loss=0., a_storage=5. if node else .1,
             bit_bounds=[1.,16.], stop_patience=30, min_epochs=50 if node else 20,
             aggregation_implementation='stable_csr_v2',optimizer_recipe='native_groups_v2')
    if framework != 'A2Q':
        if not node and framework=='QAT':
            c['lr'] = .005 if dataset=='MNIST' else (.05 if dataset=='ZINC' else .001)
        c.update(dq_low=.1, dq_change=.1, percentile=.01 if framework=='DQ' else None,
                 use_ste=framework=='QAT', use_momentum=framework=='QAT')
        return c
    c.update(lr=.01 if node else .001, weight_decay=0., init='uniform')
    if node:
        if architecture=='GIN':
            c.update(weight_decay=.01, lr_feature=.05, lr_message=.005, lr_weight=.005,
                     lr_bit=.02, a_loss=2. if dataset=='Cora' else .5)
        elif architecture=='GCN':
            c.update(weight_decay=.02 if dataset=='Cora' else .015,
                     dropout=.35 if dataset=='Cora' else .5, lr_feature=.04,
                     lr_message=.008, lr_weight=.02 if dataset=='Cora' else .008,
                     lr_bit=.1, a_loss=2.5 if dataset=='Cora' else 1.5)
        else:
            c.update(lr=.01 if dataset=='Cora' else .005, weight_decay=.001, dropout=.6,
                     lr_feature=.05, lr_message=.05, lr_weight=.005 if dataset=='Cora' else .01,
                     lr_bit=.1, lr_attention=.005, lr_edge=.005,
                     a_loss=.3 if dataset=='Cora' else .25, a_storage=10.)
    elif dataset=='ZINC':
        c.update(init='norm', a_storage=1., a_loss=.03 if architecture=='GIN' else .01,
                 lr_message=.05, lr_weight=.01 if architecture=='GIN' else .001,
                 lr_bit=.01 if architecture=='GIN' else .001)
        if architecture=='GAT':
            c.update(a_loss=.00002, a_storage=.00001, lr_bit=.0005, init='uniform')
    elif architecture=='GIN':
        if dataset=='MNIST':
            c.update(lr_feature=.005, lr_message=.005, lr_weight=.0005, lr_bit=.0001, a_loss=.00005)
        else:
            c.update(lr_feature=.01, lr_message=.01, lr_weight=.001, lr_bit=.0002, a_loss=.0002, init='norm')
    elif architecture=='GCN':
        if dataset=='MNIST':
            c.update(lr_feature=.001, lr_message=.01, lr_weight=.001, lr_bit=.001, a_loss=.001, init='norm')
        else:
            c.update(lr_feature=.01, lr_message=.01, lr_weight=.001, lr_bit=.0003, a_loss=.0002)
    else:
        c.update(lr_feature=.01, lr_message=.01, lr_weight=.0001 if dataset=='MNIST' else .001,
                 lr_bit=.01 if dataset=='MNIST' else .0005,
                 lr_attention=.0001 if dataset=='MNIST' else .001,
                 lr_edge=.05 if dataset=='MNIST' else .1, a_loss=.0002 if dataset=='MNIST' else .00002)
    return c


CELLS = {c['cell']: c for c in (cell_config(d,a,f) for d,a,f in product(DATASETS, ARCHITECTURES, FRAMEWORKS))}


def candidates(method):
    if method not in METHODS: raise ValueError(method)
    options = [dict(lr=lr, rank=2, num_prompts=20, error_weight=.1) for lr in (.001,.01)]
    if method in ('conditioned','spectral','task','error','weight_lora','gate_only','mlp'):
        options.append(dict(lr=.003, rank=4, num_prompts=20, error_weight=.1))
    if method=='error':
        options += [dict(lr=.003, rank=2, num_prompts=20, error_weight=w) for w in (.01,1.)]
    return [dict(x, candidate=i) for i,x in enumerate(options)]
