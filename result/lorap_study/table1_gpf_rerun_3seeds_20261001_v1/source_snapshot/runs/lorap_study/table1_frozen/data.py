"""Official splits. Test data are loaded only by the gated final evaluator."""
import hashlib
import torch
from torch_geometric.datasets import Planetoid, GNNBenchmarkDataset, ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree


class Bundle:
    def __init__(self, c):
        self.c = c
        if c['node']:
            self.dataset = Planetoid(c['data_root'],c['dataset'],split='public')
            self.data = self.dataset[0]
            self.train = self.dataset
            self.val = self.dataset
            self.classes = self.dataset.num_classes
            self.features = self.dataset.num_features
        else:
            self.train = self._graphs('train')
            self.val = self._graphs('val')
            self.classes = 1 if c['regression'] else self.train.num_classes
            self.features = 28 if c['regression'] else self.train[0].x.size(1)+self.train[0].pos.size(1)

    def _graphs(self, split):
        if self.c['regression']:
            return ZINC(self.c['data_root'],subset=True,split=split)
        return GNNBenchmarkDataset(self.c['data_root'],self.c['dataset'],split=split)

    def batches(self, split, seed, epoch, device, limit=None, allow_test=False):
        if split=='test' and not allow_test: raise RuntimeError('Test split is locked')
        if self.c['node']:
            batch = self.data.clone().to(device)
            if self.c['framework']=='DQ': add_prob_mask(batch)
            yield batch
            return
        dataset = self._graphs('test') if split=='test' else getattr(self,split)
        if limit is not None: dataset=dataset[:limit]
        # Reconstruct generator per epoch: restart and all methods share exact order.
        generator=torch.Generator().manual_seed(seed+200000+epoch)
        loader=DataLoader(dataset,batch_size=self.c['batch_size'],shuffle=split=='train',
                          generator=generator,num_workers=0,pin_memory=True)
        for batch in loader:
            if self.c['framework']=='DQ':
                # Native degree-CDF protection is per graph, not across unrelated graphs.
                graphs=batch.to_data_list()
                for graph in graphs: add_prob_mask(graph)
                batch=type(batch).from_data_list(graphs)
            yield batch.to(device,non_blocking=True)

    def manifest(self):
        if self.c['node']:
            return dict(split='public',features=self.features,classes=self.classes,
                        nodes=self.data.num_nodes,edges=self.data.num_edges,
                        indices={s:getattr(self.data,s+'_mask').nonzero().flatten().tolist() for s in ('train','val','test')},
                        x_sha256=hashlib.sha256(self.data.x.numpy().tobytes()).hexdigest(),
                        edges_sha256=hashlib.sha256(self.data.edge_index.numpy().tobytes()).hexdigest())
        return dict(split='official_full',subset=self.c['regression'],train_graphs=len(self.train),
                    val_graphs=len(self.val),features=self.features,classes=self.classes,
                    test_loaded=False)


def add_prob_mask(graph):
    indegree=degree(graph.edge_index[1],graph.num_nodes,dtype=torch.long)
    counts=torch.bincount(indegree)
    probabilities=(counts.float()*(.1/graph.num_nodes)).cumsum(0)+.1
    graph.prob_mask=probabilities[indegree]


def labels_and_output(c,batch,output,split):
    if c['node']:
        mask=getattr(batch,split+'_mask')
        return output[mask],batch.y[mask]
    return output,batch.y.reshape(-1)
