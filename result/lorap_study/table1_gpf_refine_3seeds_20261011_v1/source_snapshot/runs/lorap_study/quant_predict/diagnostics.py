import torch


class Diagnostics:
    def __init__(self,mask,c):
        ids=mask.nonzero().flatten()
        if len(ids)>c['max_diagnostic_nodes']:
            generator=torch.Generator(device=ids.device).manual_seed(915041)
            ids=ids[torch.randperm(len(ids),generator=generator,device=ids.device)[:c['max_diagnostic_nodes']]].sort().values
        self.ids,self.config,self.rows=ids,c,[]

    @torch.no_grad()
    def __call__(self,i,target,delta):
        error,p=target[self.ids].double(),delta[self.ids].double()
        before=error.square().sum()
        after=(error-p).square().sum()
        dot=(error*p).sum();norm=p.square().sum()
        gap=float(abs(after-before+2*dot-norm)/before.clamp_min(1e-30))
        assert gap<1e-8
        energy=torch.linalg.svdvals(error).square()
        after_energy=torch.linalg.svdvals(error-p).square()
        generator=torch.Generator(device=error.device).manual_seed(915041+i)
        shuffled=[]
        for _ in range(self.config['diagnostic_permutations']):
            perm=torch.randperm(len(p),device=p.device,generator=generator)
            shuffled.append((error-p[perm]).square().sum())
        shuffled=torch.stack(shuffled).mean()
        self.rows.append(dict(layer=2 if self.config['architecture']=='GIN' else i+1,
            nodes=len(self.ids),before_sq=float(before),after_sq=float(after),dot=float(dot),correction_sq=float(norm),
            mse_reduction=None if float(before)==0 else float(1-after/before),
            cosine=None if float(before*norm)==0 else float(dot/(before*norm).sqrt()),
            shuffled_mse_reduction=None if float(before)==0 else float(1-shuffled/before),
            identity_relative_gap=gap,squared_singular_values=energy.cpu().tolist(),
            after_squared_singular_values=after_energy.cpu().tolist(),
            cumulative_energy=(energy.cumsum(0)/energy.sum().clamp_min(1e-30)).cpu().tolist()))
