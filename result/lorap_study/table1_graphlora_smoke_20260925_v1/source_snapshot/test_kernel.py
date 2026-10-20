import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.profiler import profile, record_function, ProfilerActivity
@triton.jit
def fully_fused_gpf_kernel(
    # 输入/输出指针
    x_ptr,          # [B, C_in]
    y_ptr,          # [B, C_in]
    # Linear 层权重指针
    W_ptr,          # [P_NUM, C_in] (self.a.weight)
    B_ptr,          # [P_NUM]       (self.a.bias)
    # LoRA 权重指针
    Pl_ptr,         # [P_NUM, Rank]
    Pr_ptr,         # [Rank, C_in]
    # Strides
    stride_x_b, stride_x_c,
    stride_W_p, stride_W_c, # Linear Weight strides
    stride_Pl_p, stride_Pl_r,
    stride_Pr_r, stride_Pr_c,
    stride_y_b, stride_y_c,
    # Dimensions
    P_NUM, Rank, C_IN,
    # Block sizes
    BLOCK_P: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C_IN: tl.constexpr # 处理输入通道的分块大小
):
    pid = tl.program_id(0) # 当前处理的 Batch index

    # ===========================================================
    # 1. Fused Linear Layer (GEMM + Bias)
    # Goal: score = x[pid] @ W.T + bias
    # ===========================================================
    
    # 1.1 初始化 score 累加器，直接加载 Bias
    # -----------------------------------------------------------
    offs_p = tl.arange(0, BLOCK_P)
    mask_p = offs_p < P_NUM
    
    # 加载 bias [BLOCK_P], 不需要 stride_bias_p 因为是 1D contiguous
    bias_val = tl.load(B_ptr + offs_p, mask=mask_p, other=0.0)
    # score_acc 将驻留在寄存器中保存 Linear 的结果
    score_acc = bias_val 

    # 1.2 循环计算矩阵乘法 (x @ W.T)
    # -----------------------------------------------------------
    # 我们需要遍历输入的 C_IN 维度。
    # x 的当前行: x_ptr + pid * stride_x_b
    x_row_start_ptr = x_ptr + pid * stride_x_b

    for off_c_base in range(0, C_IN, BLOCK_C_IN):
        offs_c = off_c_base + tl.arange(0, BLOCK_C_IN)
        mask_c = offs_c < C_IN
        
        # 加载 x 的一个块 [BLOCK_C_IN]
        x_chunk = tl.load(x_row_start_ptr + offs_c * stride_x_c, mask=mask_c, other=0.0)
        
        # 加载 W 的对应块。W 是 [P_NUM, C_IN]。
        # 我们需要加载 [BLOCK_P, BLOCK_C_IN]
        w_ptrs = W_ptr + (offs_p[:, None] * stride_W_p + offs_c[None, :] * stride_W_c)
        w_chunk = tl.load(w_ptrs, mask=(mask_p[:, None] & mask_c[None, :]), other=0.0)
        
        # 计算点积并累加。
        # x_chunk 是 [BLOCK_C_IN], w_chunk 是 [BLOCK_P, BLOCK_C_IN]
        # 我们将 x 广播为 [1, BLOCK_C_IN] 然后与 w_chunk 相乘，再在 C_IN 维度求和。
        
        # [BLOCK_C_IN] -> [1, BLOCK_C_IN] -> broadcast to [BLOCK_P, BLOCK_C_IN]
        x_broad = tl.expand_dims(x_chunk, 0) 
        
        # element-wise multiply
        prod = x_broad * w_chunk
        
        # sum over C_IN axis -> [BLOCK_P]
        dot_prod = tl.sum(prod, axis=1)
        
        # 累加到 score
        score_acc += dot_prod

    # *** 同步点 ***
    # 到这里，score_acc 寄存器里已经存好了当前样本完整的 Linear 输出。
    # 由于我们是一个 block 处理一个样本，代码顺序执行自然就同步了。

    # ===========================================================
    # 2. Softmax (输入是寄存器里的 score_acc)
    # ===========================================================
    # s_row 现在就是 score_acc
    s_row = score_acc
    
    # 标准 Softmax 实现
    s_max = tl.max(s_row, axis=0)
    # 注意：这里需要再次 mask，防止计算 padding 部分的 exp 导致隐患
    s_row_masked = tl.where(mask_p, s_row - s_max, -float('inf'))
    s_num = tl.exp(s_row_masked)
    s_denom = tl.sum(s_num, axis=0)
    probs = s_num / s_denom  # [BLOCK_P] 归一化权重

    # ===========================================================
    # 3. 投影到 Rank 空间 (M = probs @ Pl)
    # ===========================================================
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < Rank
    
    # 加载 Pl [BLOCK_P, BLOCK_R]
    pl_ptrs = Pl_ptr + (offs_p[:, None] * stride_Pl_p + offs_r[None, :] * stride_Pl_r)
    pl_block = tl.load(pl_ptrs, mask=(mask_p[:, None] & mask_r[None, :]), other=0.0)
    
    # 加权求和 [BLOCK_P, 1] * [BLOCK_P, BLOCK_R] -> sum -> [BLOCK_R]
    probs_broad = tl.expand_dims(probs, 1)
    weighted_pl = probs_broad * pl_block
    M = tl.sum(weighted_pl, axis=0) # 中间向量 M，驻留寄存器

    # ===========================================================
    # 4. 投影回 Channel 并加残差 (y = x + M @ Pr)
    # ===========================================================
    M_broad = tl.expand_dims(M, 1) # [BLOCK_R, 1]

    # 再次循环 C_IN 维度用于输出
    for off_c_base in range(0, C_IN, BLOCK_C_IN):
        offs_c = off_c_base + tl.arange(0, BLOCK_C_IN)
        mask_c = offs_c < C_IN
        
        # 加载 Pr 块 [BLOCK_R, BLOCK_C_IN]
        pr_ptrs = Pr_ptr + (offs_r[:, None] * stride_Pr_r + offs_c[None, :] * stride_Pr_c)
        pr_chunk = tl.load(pr_ptrs, mask=(mask_r[:, None] & mask_c[None, :]), other=0.0)
        
        # 计算 Delta [BLOCK_C_IN]
        weighted_pr = M_broad * pr_chunk
        delta_chunk = tl.sum(weighted_pr, axis=0)
        
        # 加载原始 x 的对应块
        x_loc_ptr = x_row_start_ptr + offs_c * stride_x_c
        x_chunk = tl.load(x_loc_ptr, mask=mask_c, other=0.0)
        
        # 残差连接
        out_chunk = x_chunk + delta_chunk
        
        # 写回结果
        y_loc_ptr = y_ptr + pid * stride_y_b + offs_c * stride_y_c
        tl.store(y_loc_ptr, out_chunk, mask=mask_c)

class GPF_FullyFused(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int, rank: int):
        super().__init__()
        self.in_channels = in_channels
        self.p_num = p_num
        self.rank = rank
        
        self.P_left = nn.Parameter(torch.empty(p_num, rank))
        self.P_right = nn.Parameter(torch.empty(rank, in_channels))
        # 注意：Triton 里手动算 Linear，需要确保权重内存连续，通常默认就是连续的。
        self.a = nn.Linear(in_channels, p_num)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.P_left)
        nn.init.xavier_uniform_(self.P_right)
        self.a.reset_parameters()

    def forward(self, x):
        # 准备输出 buffer
        y = torch.empty_like(x)
        
        B = x.shape[0]
        C_IN = self.in_channels
        
        # 设置 Block Sizes
        # BLOCK_P 必须大于等于 p_num
        BLOCK_P = triton.next_power_of_2(self.p_num)
        BLOCK_R = triton.next_power_of_2(self.rank)
        # BLOCK_C_IN 用于切分输入维度，根据实际情况调整，例如 128, 256
        BLOCK_C_IN = 128 
        # 如果 C_IN 很小，就用 C_IN 的 next power of 2
        if C_IN < BLOCK_C_IN:
             BLOCK_C_IN = triton.next_power_of_2(C_IN)

        # Grid: 每个 Batch 一个 Block
        grid = (B, )

        # 获取 Linear 层的权重和偏置指针
        W = self.a.weight
        Bias = self.a.bias

        fully_fused_gpf_kernel[grid](
            x, y,
            W, Bias,
            self.P_left, self.P_right,
            # Strides
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1), # Linear weight strides
            self.P_left.stride(0), self.P_left.stride(1),
            self.P_right.stride(0), self.P_right.stride(1),
            y.stride(0), y.stride(1),
            # Dimensions
            self.p_num, self.rank, C_IN,
            # Block constants
            BLOCK_P=BLOCK_P,
            BLOCK_R=BLOCK_R,
            BLOCK_C_IN=BLOCK_C_IN
        )
        
        return y

def run_profile():
    device = torch.device("cuda")
    in_channels = 128
    p_num = 10
    rank = 4
    batch_size = 18000

    model = GPF_FullyFused(in_channels, p_num, rank).to(device)
    x = torch.randn(batch_size, in_channels, device=device)

    # Warmup
    print("Warmup...")
    for _ in range(20):
        model(x)
    torch.cuda.synchronize()

    print("\nStarting Profiler...")
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("triton_fusion_pass"):
            model(x)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.profiler import profile, record_function, ProfilerActivity

# ==========================================
# 1. Triton Kernel 定义
# ==========================================
@triton.jit
def fused_low_rank_gpf_kernel(
    x_ptr, score_ptr, Pl_ptr, Pr_ptr, y_ptr,
    stride_x_b, stride_x_c,
    stride_s_b, stride_s_p,
    stride_Pl_p, stride_Pl_r,
    stride_Pr_r, stride_Pr_c,
    stride_y_b, stride_y_c,
    P_NUM, Rank, C,
    BLOCK_P: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    pid = tl.program_id(0)

    # --- 1. Softmax (SRAM) ---
    s_ptr_start = score_ptr + pid * stride_s_b
    offs_p = tl.arange(0, BLOCK_P)
    mask_p = offs_p < P_NUM
    s_row = tl.load(s_ptr_start + offs_p * stride_s_p, mask=mask_p, other=-float('inf'))
    
    s_max = tl.max(s_row, axis=0)
    s_num = tl.exp(s_row - s_max)
    s_denom = tl.sum(s_num, axis=0)
    probs = s_num / s_denom 

    # --- 2. 投影到 Rank 空间 (Reduce) ---
    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < Rank
    pl_ptrs = Pl_ptr + (offs_p[:, None] * stride_Pl_p + offs_r[None, :] * stride_Pl_r)
    pl_block = tl.load(pl_ptrs, mask=(mask_p[:, None] & mask_r[None, :]), other=0.0)
    
    probs_broad = tl.expand_dims(probs, 1)
    weighted_pl = probs_broad * pl_block
    M = tl.sum(weighted_pl, axis=0) # [BLOCK_R] (寄存器常驻)

    # --- 3. 投影回 Channel 并加残差 ---
    M_broad = tl.expand_dims(M, 1)
    
    for off_c_base in range(0, C, BLOCK_C):
        offs_c = off_c_base + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        
        pr_ptrs = Pr_ptr + (offs_r[:, None] * stride_Pr_r + offs_c[None, :] * stride_Pr_c)
        pr_chunk = tl.load(pr_ptrs, mask=(mask_r[:, None] & mask_c[None, :]), other=0.0)
        
        weighted_pr = M_broad * pr_chunk
        delta_chunk = tl.sum(weighted_pr, axis=0)
        
        x_loc_ptr = x_ptr + pid * stride_x_b + offs_c * stride_x_c
        x_chunk = tl.load(x_loc_ptr, mask=mask_c, other=0.0)
        
        out_chunk = x_chunk + delta_chunk
        
        y_loc_ptr = y_ptr + pid * stride_y_b + offs_c * stride_y_c
        tl.store(y_loc_ptr, out_chunk, mask=mask_c)

# ==========================================
# 2. PyTorch 模型封装
# ==========================================
class GPF_LoRA_Fusion_NoBuffer(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int, rank: int):
        super().__init__()
        self.in_channels = in_channels
        self.p_num = p_num
        self.rank = rank
        
        self.P_left = nn.Parameter(torch.empty(p_num, rank))
        self.P_right = nn.Parameter(torch.empty(rank, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.P_left)
        nn.init.xavier_uniform_(self.P_right)
        self.a.reset_parameters()

    def forward(self, x):
        # Kernel 1: Native Linear (GEMM)
        score = self.a(x)

        # Kernel 2: Custom Triton Fusion
        y = torch.empty_like(x)
        BLOCK_P = triton.next_power_of_2(self.p_num)
        BLOCK_R = triton.next_power_of_2(self.rank)
        BLOCK_C = triton.next_power_of_2(self.in_channels)
        if BLOCK_C > 1024: BLOCK_C = 1024

        grid = (x.shape[0], )

        fused_low_rank_gpf_kernel[grid](
            x, score, self.P_left, self.P_right, y,
            x.stride(0), x.stride(1),
            score.stride(0), score.stride(1),
            self.P_left.stride(0), self.P_left.stride(1),
            self.P_right.stride(0), self.P_right.stride(1),
            y.stride(0), y.stride(1),
            self.p_num, self.rank, self.in_channels,
            BLOCK_P=BLOCK_P, BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C
        )
        return y

# ==========================================
# 3. Profiling 脚本
# ==========================================
def run_profile_without_gemm():
    device = torch.device("cuda")
    in_channels = 128
    p_num = 10
    rank = 4
    batch_size = 18000

    model = GPF_LoRA_Fusion_NoBuffer(in_channels, p_num, rank).to(device)
    x = torch.randn(batch_size, in_channels, device=device)

    # Warmup
    print("Warmup...")
    for _ in range(20):
        model(x)
    torch.cuda.synchronize()

    print("\nStarting Profiler...")
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("triton_fusion_pass"):
            model(x)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.profiler import profile, record_function, ProfilerActivity

# --- 你的类定义 (保持不变) ---
class GPF_plus_LoRA(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int, rank: int):
        super().__init__()
        self.in_channels = in_channels
        self.p_num = p_num
        self.rank = rank

        self.P_left  = Parameter(torch.Tensor(p_num, rank))
        self.P_right = Parameter(torch.Tensor(rank, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.P_left)
        torch.nn.init.xavier_uniform_(self.P_right)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        # 1) 计算打分并归一化
        score = self.a(x)                        # Kernel 1: GEMM (Linear)
        weight = F.softmax(score, dim=1)         # Kernel 2: Softmax

        # 2) 重构低秩 p_list
        # P = P_left @ P_right
        P = self.P_left @ self.P_right           # Kernel 3: GEMM (MatMul)

        # 3) 加权组合并加回原始 x
        delta = weight @ P                       # Kernel 4: GEMM (MatMul)
        return x + delta                         # Kernel 5: Element-wise Add

# --- 测试脚本 ---

def test_efficiency():
    # 1. 设置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("警告: 未检测到 CUDA，将在 CPU 上运行，无法统计 CUDA Kernel。")
        return

    in_channels = 128
    p_num = 10
    rank = 4
    batch_size = 18000

    # 2. 初始化模型和数据
    model = GPF_plus_LoRA(in_channels, p_num, rank).to(device)
    x = torch.randn(batch_size, in_channels, device=device)

    model.eval() # 切换到评估模式

    # 3. GPU Warm-up (预热)
    # CUDA 是异步的，且第一次运行会有初始化开销，必须预热
    print("正在预热 GPU...")
    for _ in range(20):
        _ = model.add(x)
    torch.cuda.synchronize()

    # 4. 测量精确时间 (使用 CUDA Event)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(1000): # 运行 1000 次取平均
            _ = model.add(x)
    end_event.record()
    
    torch.cuda.synchronize() # 等待所有 GPU 任务完成
    elapsed_time_ms = start_event.elapsed_time(end_event) / 1000
    print(f"Add 操作平均耗时: {elapsed_time_ms * 1000:.4f} us (微秒)")

    # 5. 统计 CUDA Kernel 调用 (Profiler)
    print("\n正在分析 CUDA Kernel 调用...")
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_add_pass"):
            with torch.no_grad():
                model.add(x)
    
    # 打印统计结果
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    #test_efficiency()
    #run_profile()
    run_profile_without_gemm()

    