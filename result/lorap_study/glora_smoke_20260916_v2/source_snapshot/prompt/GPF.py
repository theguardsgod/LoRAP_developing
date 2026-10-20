import torch
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import time
from torch_geometric.nn import GCNConv,GINConv,global_mean_pool,global_add_pool


# —— 1. 定义 STE 二值门控 —— 
class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        # 前向：logits>0 → 1，否则 0
        return (logits > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # 反向：直接把上游梯度当作自己的梯度（identity）
        return grad_output

class GPF(torch.nn.Module):
    def __init__(self, in_channels: int):
        super(GPF, self).__init__()
        self.global_emb = torch.nn.Parameter(torch.Tensor(1,in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: torch.Tensor):
        return x + self.global_emb

class GPF_plus(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPF_plus, self).__init__()
        self.p_list = torch.nn.Parameter(torch.zeros(p_num, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        # torch.nn.init.constant_(self.p_list, 0.0)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p
    def concat(self, x: torch.Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return torch.cat([x, p], dim=1)
    def concat_fft(self, x: torch.Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)
        p_fft = torch.fft.fft(torch.fft.fft(p, dim=-1),dim=-2).real

        return torch.cat([x, p, p_fft], dim=1)
    def product(self, x: torch.Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x * p
    

class GPF_plus_gradient(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPF_plus_gradient, self).__init__()
        self.p_list = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        if torch.is_grad_enabled():
            weight.register_hook(lambda grad: setattr(self, "weight_grad_buffer", grad.clone()))

        p = weight.mm(self.p_list)

        if torch.is_grad_enabled():
            p.register_hook(lambda grad: setattr(self, "node_grad_buffer", grad.clone()))

        x = x + p
        return x

class GPF_plus_bin(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int, n_bin: int):
        super(GPF_plus_bin, self).__init__()
        self.p_list = torch.nn.Parameter(torch.Tensor(n_bin, p_num, in_channels))  # 每个 bin 一组 p_list
        self.a = torch.nn.Linear(in_channels, p_num)
        self.n_bin = n_bin
        self.p_num = p_num
        self.in_channels = in_channels
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor, node_bin: torch.Tensor):
        # x: [N, in_channels], node_bin: [N]
        score = self.a(x)  # [N, p_num]
        weight = F.softmax(score, dim=1)  # [N, p_num]

        # 根据 node_bin 索引 p_list：[N, p_num, in_channels]
        selected_p_list = self.p_list[node_bin]  # [N, p_num, in_channels]

        # 按照 weight 做 batch 矩阵乘法，得到 [N, in_channels]
        p = torch.bmm(weight.unsqueeze(1), selected_p_list).squeeze(1)  # [N, 1, p_num] x [N, p_num, in_channels] → [N, 1, in_channels] → squeeze → [N, in_channels]

        return x + p
    def add_gradient(self, x: torch.Tensor):
        # x: [N, in_channels], node_bin: [N]
        score = self.a(x)  # [N, p_num]
        weight = F.softmax(score, dim=1)  # [N, p_num]

        if torch.is_grad_enabled():
            weight.register_hook(lambda grad: setattr(self, "weight_grad_buffer", grad.clone()))

        # [N, p_num, in_channels]
        
        selected_p_list = self.p_list.mean(dim=0)
        
        # [N, in_channels]
        p = weight.mm(selected_p_list)

        if torch.is_grad_enabled():
            p.register_hook(lambda grad: setattr(self, "node_grad_buffer", grad.clone()))

        return x + p
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as glorot

class GPF_plus_bin_gradient(nn.Module):
    def __init__(self, in_channels: int, p_num: int, n_bin: int):
        super(GPF_plus_bin_gradient, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(n_bin, p_num, in_channels))  # 每个 bin 一组 p_list
        self.a = nn.Linear(in_channels, p_num)
        self.n_bin = n_bin
        self.p_num = p_num
        self.in_channels = in_channels
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor, node_bin: torch.Tensor):
        # x: [N, in_channels], node_bin: [N]
        score = self.a(x)  # [N, p_num]
        weight = F.softmax(score, dim=1)  # [N, p_num]

        if torch.is_grad_enabled():
            weight.register_hook(lambda grad: setattr(self, "weight_grad_buffer", grad.clone()))

        # [N, p_num, in_channels]
        selected_p_list = self.p_list[:]
        selected_p_list = selected_p_list.mean(dim=1)
        # [N, in_channels]
        p = torch.bmm(weight.unsqueeze(1), selected_p_list).squeeze(1)

        if torch.is_grad_enabled():
            p.register_hook(lambda grad: setattr(self, "node_grad_buffer", grad.clone()))

        return x + p

class GPF_plus_mask(nn.Module):
    def __init__(
        self,
        in_channels: int,
        p_num: int,
        mlp_hidden: int = 32,
    ):
        """
        in_channels: x 的特征维 C
        p_num: p_list 中向量数量
        mlp_hidden: mask MLP 的隐藏层维度
        """
        super().__init__()

        # —— 原有部分 ——  
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)

        # —— 新增：小型 MLP 生成 mask ——  
        self.mask_mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()
        # MLP 内部参数一并初始化
        for layer in self.mask_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def add(self, x: torch.Tensor):
        # 1) 计算 p
        score  = self.a(x)                   # [N, p_num]
        weight = F.softmax(score, dim=1)     # [N, p_num]
        p      = weight.mm(self.p_list)      # [N, C]

        # 2) 子采样近似中位数阈值
        mask_logits = self.mask_mlp(x).squeeze(1)  # [N]
        N           = mask_logits.size(0)
        s           = min(256, N)                 # 子集大小
        if s < N:
            idx    = torch.randperm(N, device=x.device)[:s]
            sample = mask_logits[idx]
            thresh = sample.median()
        else:
            thresh = mask_logits.median()

        mask = (mask_logits >= thresh).float().unsqueeze(1)  # [N,1]，大致一半为 1

        # 3) 融合
        return x + p * mask
    
class GPF_plus_mask_attn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        p_num: int,
        num_heads: int = 1,
    ):
        """
        in_channels: 特征维度 C
        p_num: p_list 中向量数量
        num_heads: 自注意力头数
        """
        super().__init__()
        # — 原有 p 计算部分 —
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)

        # — 新增：多头自注意力生成 mask_logits —
        # batch_first=True，输入为 (batch, seq_len, embed_dim)
        self.mask_attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True
        )
        # 将 attn 输出投影到标量 logit
        self.mask_proj = nn.Linear(in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()
        # MultiheadAttention 参数已在内部初始化
        self.mask_proj.reset_parameters()

    def add(self, x: torch.Tensor):
        """
        x: [N, C]
        返回: [N, C]
        """
        # 1) 计算 p = softmax(a(x)) @ p_list
        score = self.a(x)                     # [N, p_num]
        weight = F.softmax(score, dim=1)      # [N, p_num]
        p = weight.mm(self.p_list)            # [N, C]

        # 2) 用自注意力生成 mask_logits
        #    MultiheadAttention 要求输入 shape = (batch, seq_len, embed)
        x_batch = x.unsqueeze(0)              # [1, N, C]
        attn_out, _ = self.mask_attn(x_batch, x_batch, x_batch)
                                              # attn_out: [1, N, C]
        attn_out = attn_out.squeeze(0)        # [N, C]
        mask_logits = self.mask_proj(attn_out)  # [N, 1]

        # 3) 用 STE 二值化
        mask = BinarySTE.apply(mask_logits)   # [N, 1] 只有 0/1，梯度直通
        #mask = torch.sigmoid(mask_logits) 

        # 4) 只对 mask==1 的节点加 p
        out = x + p * mask                    # [N, C]
        return out

class SpectralGPF_plus(nn.Module):
    def __init__(self, in_channels: int, p_num: int, eps: float = 1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.p_num = p_num
        self.eps = eps

        # Spectral prompt 所需参数
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a      = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.p_list)
        self.a.reset_parameters()

    # def _process_one(self, x: torch.Tensor,
    #                        edge_index: torch.Tensor
    #                        ) -> torch.Tensor:
    #     """
    #     对单个图 x, edge_index, edge_attr 做谱分解 + prompt + 回到原域
    #     返回 x_p 形状 [n_i, F]
    #     """
    #     n = x.size(0)
    #     # 1) 构造稠密邻接矩阵
    #     A = to_dense_adj(edge_index, max_num_nodes=n)[0]   # [n,n]
    #     deg = A.sum(1) + self.eps                          # [n]
    #     D   = torch.diag(deg)                              # [n,n]
    #     L   = D - A
    #     L   = 0.5 * (L + L.t())                            # 对称化

    #     # 2) 特征分解
    #     eigvals, eigvecs = torch.linalg.eigh(L)            # [n], [n,n]
    #     eigvals = torch.clamp(eigvals, min=0.0)

    #     # 3) 构造 Q_s, Q_inv
    #     sqrtL    = torch.diag(torch.sqrt(eigvals))                 # [n,n]
    #     inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals + self.eps))
    #     Q_s      = eigvecs @ sqrtL                                  # [n,n]
    #     Q_inv    = inv_sqrt @ eigvecs.t()                           # [n,n]

    #     # 4) 谱域提示
    #     x_s   = Q_s @ x                                             # [n, F]
    #     score = self.a(x_s)                                         # [n, p_num]
    #     w     = F.softmax(score, dim=1)                             # [n, p_num]
    #     p     = w.mm(self.p_list)                                   # [n, F]
    #     x_sp  = x_s + p                                              # [n, F]
    #     x_p   = Q_inv @ x_sp                                         # [n, F]

    #     return x_p

    def _process_one(self, x: torch.Tensor,
                       edge_index: torch.Tensor,) -> torch.Tensor:
        """
        对单个图 x, edge_index, edge_attr 做谱分解 + prompt + 回到原域，
        使用对称归一化拉普拉斯 L_sym = I - D^{-1/2} A D^{-1/2}
        返回 x_p 形状 [n_i, F]
        """
        n = x.size(0)
        # 1) 构造稠密邻接矩阵 A
        A = to_dense_adj(edge_index, max_num_nodes=n)[0]  # [n,n]

        # 2) 计算 D^{-1/2}
        deg = A.sum(1)                                                   # [n]
        deg_inv_sqrt = (deg + self.eps).pow(-0.5)                        # [n]
        D_inv_sqrt = torch.diag(deg_inv_sqrt)                            # [n,n]

        # 3) 构造对称归一化拉普拉斯 L_sym = I - D^{-1/2} A D^{-1/2}
        I = torch.eye(n, device=x.device)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt                             # [n,n]
        L_sym = I - A_norm
        # 数值对称化，消除浮点误差
        L_sym = 0.5 * (L_sym + L_sym.t())
     
        # 4) 谱分解
        eigvals, eigvecs = torch.linalg.eigh(L_sym)                     # [n], [n,n]
        # 理论上 eigvals ∈ [0,2]，截断小于零的数值误差
        eigvals = torch.clamp(eigvals, min=0.0)

        # 5) 构造 Q_s = Q · sqrt(Λ) 和 Q_inv = Λ^{-1/2} · Q^T
        sqrt_L   = torch.diag(eigvals.sqrt())                           # [n,n]
        inv_sqrt = torch.diag((eigvals + self.eps).pow(-0.5))           # [n,n]
        Q_s      = eigvecs @ sqrt_L                                      # [n,n]
        Q_inv    = inv_sqrt @ eigvecs.t()                               # [n,n]

        # 6) 谱域 prompt 操作
        x_s   = Q_s @ x                                                  # [n, F]
        score = self.a(x_s)                                              # [n, p_num]
        w     = F.softmax(score, dim=1)                                  # [n, p_num]
        p     = w.mm(self.p_list)                                        # [n, F]
        x_sp  = x_s + p                                                  # [n, F]
        x_p   = Q_inv @ x_sp                                             # [n, F]

        return x_p          



    def add(self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            ptr: torch.Tensor) -> torch.Tensor:
        """
        针对一个 DataBatch：
        - x:         [total_nodes, F]
        - edge_index:[2, total_edges]
        - edge_attr: [total_edges]
        - ptr:       [batch_size+1]，PyG DataBatch.ptr
        返回同形 [total_nodes, F]
        """
        out = torch.zeros_like(x)
        batch_size = ptr.size(0) - 1

        for i in range(batch_size):
            start, end = ptr[i].item(), ptr[i+1].item()
            # 节点特征切片
            x_i = x[start:end]  # [n_i, F]

            # 边切片 & 重索引
            mask = (edge_index[0] >= start) & (edge_index[0] < end) \
                 & (edge_index[1] >= start) & (edge_index[1] < end)
            ei = edge_index[:, mask] - start                       # [2, e_i]
                                      # [e_i]

            # 对第 i 个图做谱分解 + prompt
            x_p_i = self._process_one(x_i, ei)                  # [n_i, F]
            out[start:end] = x_p_i

        return out
    
    def concat(self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            ptr: torch.Tensor) -> torch.Tensor:
        """
        针对一个 DataBatch，逐图做谱分解 + prompt，然后把原始特征 x_i
        和处理后的特征 x_p_i 在 feature 维度上拼接。
        - x:         [total_nodes, F]
        - edge_index:[2, total_edges]
        - edge_attr: [total_edges]
        - ptr:       [batch_size+1]
        返回 out:   [total_nodes, 2*F]
        """
        total_nodes, F = x.size(0), x.size(1)
        out = x.new_zeros((total_nodes, 2 * F))
        batch_size = ptr.size(0) - 1

        for i in range(batch_size):
            start, end = ptr[i].item(), ptr[i+1].item()
            # 1) 切出第 i 个子图的节点特征
            x_i = x[start:end]                   # [n_i, F]

            # 2) 切出并重索引该子图的边
            mask = (edge_index[0] >= start) & (edge_index[0] < end) \
                & (edge_index[1] >= start) & (edge_index[1] < end)
            ei = edge_index[:, mask] - start     # [2, e_i]
                          # [e_i]

            # 3) 对第 i 个图做谱分解 + prompt
            x_p_i = self._process_one(x_i, ei)  # [n_i, F]

            # 4) 拼接原始 x_i 与提示后的 x_p_i
            out[start:end] = torch.cat([x_i, x_p_i], dim=1)  # [n_i, 2*F]

        return out
    

    def concat_fast(self,
           x: torch.Tensor,
           edge_index: torch.Tensor,
           ptr: torch.Tensor) -> torch.Tensor:
        """
        Batch-eigh 版本 concat：
        - x:         [total_nodes, F]
        - edge_index:[2, total_edges]
        - edge_attr: [total_edges]
        - ptr:       [B+1]
        返回 out:   [total_nodes, 2*F]
        """
        B = ptr.size(0) - 1
        sizes = (ptr[1:] - ptr[:-1]).tolist()      # list of n_i
        N_max = max(sizes)
        F = x.size(1)

        # 1) pad x 到 batch
        x_batch = x.new_zeros((B, N_max, F))
        A_batch = x.new_zeros((B, N_max, N_max))
        for i in range(B):
            s, e = int(ptr[i]), int(ptr[i+1])
            n_i = e - s
            x_batch[i, :n_i] = x[s:e]
            # 子图稠密邻接
            mask = ((edge_index[0] >= s) & (edge_index[0] < e) &
                    (edge_index[1] >= s) & (edge_index[1] < e))
            ei = edge_index[:, mask] - s
            A_i = to_dense_adj(ei, max_num_nodes=n_i)[0]
            A_batch[i, :n_i, :n_i] = A_i

        # 2) 构造 L_batch 并对称化
        deg = A_batch.sum(dim=-1)                            # [B, N_max]
        D_batch = torch.diag_embed(deg)                      # [B, N_max, N_max]
        L_batch = D_batch - A_batch
        L_batch = 0.5 * (L_batch + L_batch.transpose(1, 2))

        # 3) 一次性做 Batched eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(L_batch)        # [B, N_max], [B, N_max, N_max]
        eigvals = eigvals.clamp(min=0.0)
        sqrt_e = eigvals.sqrt()                              # [B, N_max]
        inv_e  = 1.0 / (sqrt_e + self.eps)

        sqrt_L = torch.diag_embed(sqrt_e)                     # [B, N_max, N_max]
        inv_L  = torch.diag_embed(inv_e)                      # [B, N_max, N_max]
        Q_s    = torch.matmul(eigvecs, sqrt_L)                # [B, N_max, N_max]
        Q_inv  = torch.matmul(inv_L, eigvecs.transpose(-1, -2))  # [B, N_max, N_max]

        # 4) 谱域 prompt（并行）
        X_s   = torch.matmul(Q_s, x_batch)                    # [B, N_max, F]
        scores= self.a(X_s)                                   # [B, N_max, p_num]
        wght= torch.softmax(scores, dim=-1)                   # [B, N_max, p_num]
        P_b  = torch.matmul(wght, self.p_list)                # [B, N_max, F]
        X_sp = X_s + P_b                                      # [B, N_max, F]
        X_p  = torch.matmul(Q_inv, X_sp)                      # [B, N_max, F]

        # 5) unpad + concat 回原序
        out = x.new_zeros((x.size(0), 2*F))
        for i in range(B):
            s, e = ptr[i].item(), ptr[i+1].item()
            n_i = e - s
            out[s:e] = torch.cat([x_batch[i, :n_i], X_p[i, :n_i]], dim=1)

        return out

    def concat_batch_eigh_timed(model, x, edge_index, ptr):
        """
        针对 Batch-eigh 版 concat，一次性测量每一步的时间。
        返回：out, times_dict
        """
        # 如果在 GPU 上，测前先同步
        if x.is_cuda: torch.cuda.synchronize()
        times = {}

        # — Step 1: pad & 构造 A_batch —
        t0 = time.perf_counter()
        B = ptr.size(0) - 1
        sizes = (ptr[1:] - ptr[:-1]).tolist()
        N_max = max(sizes)
        Fdim  = x.size(1)
        x_batch = x.new_zeros((B, N_max, Fdim))
        A_batch = x.new_zeros((B, N_max, N_max))
        for i in range(B):
            s, e = int(ptr[i]), int(ptr[i+1])
            n_i = e - s
            x_batch[i, :n_i] = x[s:e]
            mask = ((edge_index[0] >= s) & (edge_index[0] < e) &
                    (edge_index[1] >= s) & (edge_index[1] < e))
            ei = edge_index[:, mask] - s
            
            A_i = to_dense_adj(ei, max_num_nodes=n_i)[0]
            A_batch[i, :n_i, :n_i] = A_i
        if x.is_cuda: torch.cuda.synchronize()
        times['pad_and_build_A'] = time.perf_counter() - t0

        # — Step 2: 构造 L_batch & 对称化 —
        t0 = time.perf_counter()
        deg_batch = A_batch.sum(dim=2)
        D_batch   = torch.diag_embed(deg_batch)
        L_batch   = D_batch - A_batch
        L_batch   = 0.5 * (L_batch + L_batch.transpose(1, 2))
        if x.is_cuda: torch.cuda.synchronize()
        times['build_L'] = time.perf_counter() - t0

        # — Step 3: Batched eigh —
        t0 = time.perf_counter()
        eigvals, eigvecs = torch.linalg.eigh(L_batch)
        if x.is_cuda: torch.cuda.synchronize()
        times['eigh'] = time.perf_counter() - t0

        # — Step 4: 谱域 prompt —
        t0 = time.perf_counter()
        eigvals = eigvals.clamp(min=0.0)
        sqrt_e  = eigvals.sqrt()
        inv_e   = 1.0 / (sqrt_e + model.eps)
        Q_s     = torch.matmul(eigvecs, torch.diag_embed(sqrt_e))
        Q_inv   = torch.matmul(torch.diag_embed(inv_e), eigvecs.transpose(-1, -2))
        X_s     = torch.matmul(Q_s, x_batch)
        scores  = model.a(X_s)
        weights = F.softmax(scores, dim=-1)
        P_b     = torch.matmul(weights, model.p_list)
        X_sp    = X_s + P_b
        X_p     = torch.matmul(Q_inv, X_sp)
        if x.is_cuda: torch.cuda.synchronize()
        times['spectral_prompt'] = time.perf_counter() - t0

        # — Step 5: unpad & concat —
        t0 = time.perf_counter()
        out = x.new_zeros((x.size(0), 2 * Fdim))
        for i in range(B):
            s, e = int(ptr[i]), int(ptr[i+1])
            n_i = e - s
            x_i   = x[s:e]
            x_p_i = X_p[i, :n_i]
            out[s:e] = torch.cat([x_i, x_p_i], dim=1)
        if x.is_cuda: torch.cuda.synchronize()
        times['unpad_and_concat'] = time.perf_counter() - t0

        return out, times



class GPF_plus_fft(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPF_plus_fft, self).__init__()
        self.p_list = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)
        self.p_list_fft = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a_fft = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p
    def concat(self, x: torch.Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        score_fft = self.a_fft(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight_fft = F.softmax(score_fft, dim=1)
        p_fft = weight_fft.mm(self.p_list_fft)

        return torch.cat([x, p_fft], dim=1)
    def product(self, x: torch.Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x * p

class IDPromptG(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 1):
        super(IDPromptG, self).__init__()
        # instead of a fixed global_emb, we build a tiny prompt-generator MLP
        self.prompt_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prompt_generator = nn.Linear(hidden_dim, in_channels)
        # self.prompt_mlp = GINConv(
        #                 nn.Sequential(
        #                     nn.Linear(in_channels, hidden_dim),
        #                     nn.BatchNorm1d(hidden_dim),
        #                     nn.ReLU(),
        #                     nn.Linear(hidden_dim, in_channels),
        #                     nn.BatchNorm1d(in_channels),
        #                     nn.ReLU(),
        #                 ),
        #                 train_eps=True,
        #             )
        self.reset_parameters()

    def reset_parameters(self):
        # initialize MLP weights
        for module in self.prompt_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # initialize prompt_generator weights
        nn.init.xavier_uniform_(self.prompt_generator.weight)
        nn.init.zeros_(self.prompt_generator.bias)
        # self.prompt_mlp.reset_parameters()

    def add(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, in_channels]  (or [batch_size, in_channels] if no sequence dim)
        returns: x + prompt  (broadcasted across seq_len if necessary)
        """

        # 2) Generate a prompt embedding for each sample: [B, C]
        prompt = self.prompt_mlp(x)
        prompt = self.prompt_generator(prompt)

        return x + prompt

class IDPromptG_plus(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128, p_num: int = 20):
        super(IDPromptG_plus, self).__init__()
        # instead of a fixed global_emb, we build a tiny prompt-generator MLP
        self.prompt_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels * p_num),
        )
        self.a = torch.nn.Linear(in_channels, p_num)
        self.p_num = p_num
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier 初始化 MLP 和 gating
        for m in self.prompt_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.a.weight)
        nn.init.zeros_(self.a.bias)


    def add(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, in_channels]  (or [batch_size, in_channels] if no sequence dim)
        returns: x + prompt  (broadcasted across seq_len if necessary)
        """

        # 2) Generate a prompt embedding for each sample: [B, C]
        # 1) 生成 p_num 候选 prompt：[N, C * p_num] -> [N, p_num, C]
        prompts = self.prompt_mlp(x) \
                     .view(-1, self.p_num, self.in_channels)  # [N, p_num, C]

        # 2) gating 权重：[N, p_num]
        w = self.a(x)                    # [N, p_num]
        w = F.softmax(w, dim=-1).unsqueeze(-1)     # [N, p_num, 1]

        # 3) 加权求和得到最终 prompt：[N, C]
        prompt = (prompts * w).sum(dim=1)     # [N, C]

        return x + prompt
    
def chebyshev_filter(X, L_tilde, coeffs):
        """
        X:       [N, F]
        L_tilde: [N, N]  重标度拉普拉斯
        coeffs:  [M+1]   Chebyshev 系数 c0..cM
        返回:     [N, F] ≈ ∑ c_m T_m(L_tilde) X
        """
        M = coeffs.size(0) - 1
        T0 = X
        if M == 0:
            return coeffs[0] * T0
        T1 = L_tilde @ X
        out = coeffs[0] * T0 + coeffs[1] * T1

        for m in range(2, M+1):
            T2 = 2 * (L_tilde @ T1) - T0
            out = out + coeffs[m] * T2
            T0, T1 = T1, T2

        return out
import numpy as np
from numpy.polynomial.chebyshev import chebfit
class SpectralPromptCheb(nn.Module):
    def __init__(self,
                 in_channels: int,
                 p_num: int,
                 M: int = 50,
                 lambda_max: float = 2.0,
                 n_samples: int = 2000,
                 eps: float = 1e-8):
        """
        p_num:        prompt 长度
        in_channels:  特征维度 F
        M:            Chebyshev 多项式阶数
        lambda_max:   归一化拉普拉斯的最大特征值
        n_samples:    拟合系数时在 [-1,1] 上采样数量
        eps:          数值稳定项
        """
        super().__init__()
        self.eps = eps
        self.lambda_max = lambda_max

        # 1) 在 [-1,1] 上均匀采样并映射回 [0, λ_max]
        x = np.linspace(-1, 1, n_samples)
        lam = (x + 1) * lambda_max / 2

        # 2) 计算目标函数值
        y_g = np.sqrt(lam)                   # g(λ)=√λ
        y_d = 1.0 / np.sqrt(lam + 1e-12)     # g⁻¹(λ)=1/√λ

        # 3) Chebyshev 拟合，得到 M+1 个系数
        c = chebfit(x, y_g, M)               # shape (M+1,)
        d = chebfit(x, y_d, M)

        # 4) 注册为 buffer，后续不更新
        self.register_buffer('c', torch.from_numpy(c).float())
        self.register_buffer('d', torch.from_numpy(d).float())

        # prompt 参数
        self.p = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)

    def _process_one(self,
            x: torch.Tensor,
            edge_index: torch.Tensor ) -> torch.Tensor:
        """
        x:          [n, F]
        edge_index: [2, e]
        edge_attr:  [e]
        返回 x_p:   [n, F]
        """
        n, F = x.size()

        # — 构造对称归一化拉普拉斯 L_sym —
        A = to_dense_adj(edge_index, max_num_nodes=n)[0]
        deg = A.sum(1)
        deg_inv_sqrt = (deg + self.eps).pow(-0.5)
        D_inv_s = torch.diag(deg_inv_sqrt)
        A_norm = D_inv_s @ A @ D_inv_s
        L_sym = torch.eye(n, device=x.device) - A_norm
        L_sym = 0.5 * (L_sym + L_sym.t())

        # — 重标度到 [-1,1] —
        L_tilde = (2.0 / self.lambda_max) * L_sym - torch.eye(n, device=x.device)

        # — Chebyshev 近似 g(L)·x —
        X_s = chebyshev_filter(x, L_tilde, self.c)

        # — 谱域 prompt —
        score  = self.a(X_s)
        weight = torch.softmax(score, dim=1)
        P      = weight.mm(self.p)
        X_sp   = X_s + P

        # — Chebyshev 近似 g⁻¹(L)·X_sp —
        X_p = chebyshev_filter(X_sp, L_tilde, self.d)
        return X_p

    def add(self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            ptr: torch.Tensor) -> torch.Tensor:
        """
        针对一个 DataBatch：
        - x:         [total_nodes, F]
        - edge_index:[2, total_edges]
        - edge_attr: [total_edges]
        - ptr:       [batch_size+1]，PyG DataBatch.ptr
        返回同形 [total_nodes, F]
        """
        out = torch.zeros_like(x)
        batch_size = ptr.size(0) - 1

        for i in range(batch_size):
            start, end = ptr[i].item(), ptr[i+1].item()
            # 节点特征切片
            x_i = x[start:end]  # [n_i, F]

            # 边切片 & 重索引
            mask = (edge_index[0] >= start) & (edge_index[0] < end) \
                 & (edge_index[1] >= start) & (edge_index[1] < end)
            ei = edge_index[:, mask] - start                       # [2, e_i]
                                      # [e_i]

            # 对第 i 个图做谱分解 + prompt
            x_p_i = self._process_one(x_i, ei)                  # [n_i, F]
            out[start:end] = x_p_i
        
        return out

class SpectralGPF_plusplus(nn.Module):
    def __init__(self, in_channels: int, p_num: int, eps: float = 1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.p_num = p_num
        self.eps = eps

        # Spectral prompt 所需参数
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a      = nn.Linear(in_channels, p_num)
        self.p_list_fft = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a_fft = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.p_list)
        self.a.reset_parameters()

   

    def _process_one(self, x: torch.Tensor,
                       edge_index: torch.Tensor,) -> torch.Tensor:
        """
        对单个图 x, edge_index, edge_attr 做谱分解 + prompt + 回到原域，
        使用对称归一化拉普拉斯 L_sym = I - D^{-1/2} A D^{-1/2}
        返回 x_p 形状 [n_i, F]
        """
        n = x.size(0)
        # 1) 构造稠密邻接矩阵 A
        A = to_dense_adj(edge_index, max_num_nodes=n)[0]  # [n,n]

        # 2) 计算 D^{-1/2}
        deg = A.sum(1)                                                   # [n]
        deg_inv_sqrt = (deg + self.eps).pow(-0.5)                        # [n]
        D_inv_sqrt = torch.diag(deg_inv_sqrt)                            # [n,n]

        # 3) 构造对称归一化拉普拉斯 L_sym = I - D^{-1/2} A D^{-1/2}
        I = torch.eye(n, device=x.device)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt                             # [n,n]
        L_sym = I - A_norm
        # 数值对称化，消除浮点误差
        L_sym = 0.5 * (L_sym + L_sym.t())
     
        # 4) 谱分解
        eigvals, eigvecs = torch.linalg.eigh(L_sym)                     # [n], [n,n]
        # 理论上 eigvals ∈ [0,2]，截断小于零的数值误差
        eigvals = torch.clamp(eigvals, min=0.0)

        # 5) 构造 Q_s = Q · sqrt(Λ) 和 Q_inv = Λ^{-1/2} · Q^T
        sqrt_L   = torch.diag(eigvals.sqrt())                           # [n,n]
        inv_sqrt = torch.diag((eigvals + self.eps).pow(-0.5))           # [n,n]
        Q_s      = eigvecs @ sqrt_L                                      # [n,n]
        Q_inv    = inv_sqrt @ eigvecs.t()                               # [n,n]

        # 6) 谱域 prompt 操作
        x_s   = Q_s @ x                                                  # [n, F]
        score = self.a_fft(x_s)                                              # [n, p_num]
        w     = F.softmax(score, dim=1)                                  # [n, p_num]
        p_fft     = w.mm(self.p_list_fft)                                        # [n, F]
        

        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return p, p_fft      



    def add(self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            ptr: torch.Tensor) -> torch.Tensor:
        """
        针对一个 DataBatch：
        - x:         [total_nodes, F]
        - edge_index:[2, total_edges]
        - edge_attr: [total_edges]
        - ptr:       [batch_size+1]，PyG DataBatch.ptr
        返回同形 [total_nodes, F]
        """
        out = torch.zeros_like(x)
        batch_size = ptr.size(0) - 1

        for i in range(batch_size):
            start, end = ptr[i].item(), ptr[i+1].item()
            # 节点特征切片
            x_i = x[start:end]  # [n_i, F]

            # 边切片 & 重索引
            mask = (edge_index[0] >= start) & (edge_index[0] < end) \
                 & (edge_index[1] >= start) & (edge_index[1] < end)
            ei = edge_index[:, mask] - start                       # [2, e_i]
                                      # [e_i]

            # 对第 i 个图做谱分解 + prompt
            x_p_i = self._process_one(x_i, ei)                  # [n_i, F]
            out[start:end] = x_p_i

        return out
    
    def concat(self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            ptr: torch.Tensor) -> torch.Tensor:
        """
        针对一个 DataBatch，逐图做谱分解 + prompt，然后把原始特征 x_i
        和处理后的特征 x_p_i 在 feature 维度上拼接。
        - x:         [total_nodes, F]
        - edge_index:[2, total_edges]
        - edge_attr: [total_edges]
        - ptr:       [batch_size+1]
        返回 out:   [total_nodes, 2*F]
        """
        total_nodes, F = x.size(0), x.size(1)
        out = x.new_zeros((total_nodes, 3 * F))
        batch_size = ptr.size(0) - 1

        for i in range(batch_size):
            start, end = ptr[i].item(), ptr[i+1].item()
            # 1) 切出第 i 个子图的节点特征
            x_i = x[start:end]                   # [n_i, F]

            # 2) 切出并重索引该子图的边
            mask = (edge_index[0] >= start) & (edge_index[0] < end) \
                & (edge_index[1] >= start) & (edge_index[1] < end)
            ei = edge_index[:, mask] - start     # [2, e_i]
                          # [e_i]

            # 3) 对第 i 个图做谱分解 + prompt
            x_p_i, x_p_i_fft = self._process_one(x_i, ei)  # [n_i, F]

            # 4) 拼接原始 x_i 与提示后的 x_p_i
            out[start:end] = torch.cat([x_i, x_p_i, x_p_i_fft], dim=1)  # [n_i, 2*F]

        return out
    
class SpectralGPF_pre(nn.Module):
    def __init__(self, in_channels: int, p_num: int, eps: float = 1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.p_num = p_num
        self.eps = eps

        # Spectral prompt 所需参数
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a      = nn.Linear(in_channels, p_num)
        self.p_list_fft = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a_fft = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.p_list)
        self.a.reset_parameters()

   

    def _process_one(self, x: torch.Tensor,
                       edge_index: torch.Tensor,eigvals, eigvecs) -> torch.Tensor:
        """
        对单个图 x, edge_index, edge_attr 做谱分解 + prompt + 回到原域，
        使用对称归一化拉普拉斯 L_sym = I - D^{-1/2} A D^{-1/2}
        返回 x_p 形状 [n_i, F]
        """
        
     
        # 4) 谱分解
        # 理论上 eigvals ∈ [0,2]，截断小于零的数值误差
        eigvals = torch.clamp(eigvals, min=0.0)

        # 5) 构造 Q_s = Q · sqrt(Λ) 和 Q_inv = Λ^{-1/2} · Q^T
        sqrt_L   = torch.diag(eigvals.sqrt())                           # [n,n]
        inv_sqrt = torch.diag((eigvals + self.eps).pow(-0.5))           # [n,n]
        Q_s      = eigvecs @ sqrt_L                                      # [n,n]
        Q_inv    = inv_sqrt @ eigvecs.t()                               # [n,n]

        # 6) 谱域 prompt 操作
        x_s   = Q_s @ x                                                  # [n, F]
        score = self.a_fft(x_s)                                              # [n, p_num]
        w     = F.softmax(score, dim=1)                                  # [n, p_num]
        p_fft     = w.mm(self.p_list_fft)                                        # [n, F]
        x_sp = x_s + p_fft
        x_p   = Q_inv @ x_sp                                             # [n, F]

        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        

        return x+p, x_p      
    
    def concat(self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            ptr: torch.Tensor, eigvals_list, eigvecs_list) -> torch.Tensor:
        """
        针对一个 DataBatch，逐图做谱分解 + prompt，然后把原始特征 x_i
        和处理后的特征 x_p_i 在 feature 维度上拼接。
        - x:         [total_nodes, F]
        - edge_index:[2, total_edges]
        - edge_attr: [total_edges]
        - ptr:       [batch_size+1]
        返回 out:   [total_nodes, 2*F]
        """
        total_nodes, F = x.size(0), x.size(1)
        out = x.new_zeros((total_nodes, 2 * F))
        batch_size = ptr.size(0) - 1

        for i in range(batch_size):
            start, end = ptr[i].item(), ptr[i+1].item()
            # 1) 切出第 i 个子图的节点特征
            x_i = x[start:end]                   # [n_i, F]

            # 2) 切出并重索引该子图的边
            mask = (edge_index[0] >= start) & (edge_index[0] < end) \
                & (edge_index[1] >= start) & (edge_index[1] < end)
            ei = edge_index[:, mask] - start     # [2, e_i]
                          # [e_i]

            # 3) 对第 i 个图做谱分解 + prompt
            x_p_i, x_p_i_fft = self._process_one(x_i, ei, eigvals_list[i], eigvecs_list[i])  # [n_i, F]

            # 4) 拼接原始 x_i 与提示后的 x_p_i
            out[start:end] = torch.cat([x_i, x_p_i_fft], dim=1)  # [n_i, 2*F]

        return out

    
# class GPF_plus(torch.nn.Module):
#     def __init__(self, in_channels: int, p_num: int):
#         super(GPF_plus, self).__init__()
        
#         self.p_a = torch.nn.Linear(in_channels, 1)
#         self.p_b = torch.nn.Linear(1, p_num)
        
#         self.reset_parameters()

#     def reset_parameters(self):
        
#         # torch.nn.init.constant_(self.p_list, 0.0)
#         self.a.reset_parameters()

#     def add(self, x: torch.Tensor):
#         score = self.a(x)
#         # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
#         weight = F.softmax(score, dim=1)
#         p = weight.mm(self.p_list)

#         return x + p

class low(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 1):
        super(IDPromptG, self).__init__()
        # instead of a fixed global_emb, we build a tiny prompt-generator MLP
        self.prompt_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels),
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        # initialize MLP weights
        for module in self.prompt_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # self.prompt_mlp.reset_parameters()

    def add(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, in_channels]  (or [batch_size, in_channels] if no sequence dim)
        returns: x + prompt  (broadcasted across seq_len if necessary)
        """

        # 2) Generate a prompt embedding for each sample: [B, C]
        prompt = self.prompt_mlp(x)

        return x + prompt


class EdgePrompt(nn.Module):
    def __init__(self, dim_list):
        super(EdgePrompt, self).__init__()
        self.global_prompt = nn.ParameterList([nn.Parameter(torch.Tensor(1, dim)) for dim in dim_list])
        self.reset_parameters()

    def reset_parameters(self):
        for prompt in self.global_prompt:
            glorot(prompt)

    def get_prompt(self, x, edge_index, layer):
        return self.global_prompt[layer]

class EdgePromptplus(nn.Module):
    def __init__(self, dim_list, num_anchors):
        super(EdgePromptplus, self).__init__()
        self.anchor_prompt = nn.ParameterList([nn.Parameter(torch.Tensor(num_anchors, dim)) for dim in dim_list])
        self.w = nn.ModuleList([nn.Linear(2 * dim, num_anchors) for dim in dim_list])
        self.reset_parameters()

    def reset_parameters(self):
        for anchor in self.anchor_prompt:
            glorot(anchor)

        for w in self.w:
            w.reset_parameters()

    def get_prompt(self, x, edge_index, layer):
        combined_x = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        b = F.softmax(F.leaky_relu(self.w[layer](combined_x)), dim=1)
        prompt = b.mm(self.anchor_prompt[layer])
        return prompt
    
import torch
import torch.nn.functional as F
from torch.nn import Parameter

class GPF_plus_LoRA(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int, rank: int):
        """
        in_channels: 输入特征维度
        p_num:    掩码向量数量（原 p_list 的第一维）
        rank:     分解秩 r，通常取 1~16
        """
        super().__init__()
        self.in_channels = in_channels
        self.p_num = p_num
        self.rank = rank

        # 低秩分解参数
        self.P_left  = Parameter(torch.Tensor(p_num, rank))
        self.P_right = Parameter(torch.Tensor(rank, in_channels))
        # 原来的打分网络
        self.a = torch.nn.Linear(in_channels, p_num)

        self.reset_parameters()

    def reset_parameters(self):
        # # 采用 Glorot/Xavier 初始化
        # nn.init.normal_(self.P_left, mean=0.0, std=0.1)
        # # L2 全部置 0
        # nn.init.zeros_(self.P_right)
        torch.nn.init.xavier_uniform_(self.P_left)
        torch.nn.init.xavier_uniform_(self.P_right)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        """
        x: [batch_size, in_channels]
        返回: x + softmax(a(x)) @ (P_left @ P_right)
        """
        # 1) 计算打分并归一化
        score = self.a(x)                        # [batch, p_num]
        weight = F.softmax(score, dim=1)         # [batch, p_num]

        # 2) 重构低秩 p_list
        #    P = P_left @ P_right  -> [p_num, in_channels]
        P = self.P_left @ self.P_right

        # 3) 加权组合并加回原始 x
        delta = weight @ P                     # [batch, in_channels]
        return x + delta



class GPF_plus_LoRA_Layer(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int, rank: int, a):
        """
        in_channels: 输入特征维度
        p_num:    掩码向量数量（原 p_list 的第一维）
        rank:     分解秩 r，通常取 1~16
        """
        super().__init__()
        self.in_channels = in_channels
        self.p_num = p_num
        self.rank = rank
        self.a = a
        # 低秩分解参数，每一层独立的 P_left 和 P_right
        self.P_left  = Parameter(torch.Tensor(p_num, rank))
        self.P_right = Parameter(torch.Tensor(rank, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Xavier 初始化
        torch.nn.init.xavier_uniform_(self.P_left)
        torch.nn.init.xavier_uniform_(self.P_right)

    def add(self, x: torch.Tensor):
        """
        x: [batch_size, in_channels]
        返回: x + softmax(a(x)) @ (P_left @ P_right)
        """
        # 1) 计算打分并归一化
        score = self.a(x)                        # [batch, p_num]
        weight = F.softmax(score, dim=1)         # [batch, p_num]

        # 2) 重构低秩 p_list
        P = self.P_left @ self.P_right  # [p_num, in_channels]

        # 3) 加权组合并加回原始 x
        delta = weight @ P                     # [batch, in_channels]
        return x + delta


class GPF_plus_LoRA_multi(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int, rank: int, num_layers: int):
        """
        in_channels: 输入特征维度
        p_num:    掩码向量数量（原 p_list 的第一维）
        rank:     分解秩 r，通常取 1~16
        num_layers: 网络层数
        """
        super().__init__()
        self.in_channels = in_channels
        self.p_num = p_num
        self.rank = rank
        self.num_layers = num_layers

        # 打分网络 (共享)
        self.a = torch.nn.Linear(in_channels, p_num)

        # 为每一层创建独立的 GPF_plus_LoRA 模块
        self.layers = nn.ModuleList([GPF_plus_LoRA_Layer(in_channels, p_num, rank, self.a) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        # 重置每一层的参数
        for layer in self.layers:
            layer.reset_parameters()

        # 重置共享的打分网络
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        """
        x: [batch_size, in_channels]
        返回: x + softmax(a(x)) @ (P_left @ P_right) 对所有层的加和
        """
        delta = torch.zeros_like(x)
        
        for layer in self.layers:
            delta += layer.add(x, self.a)

        return x + delta
