# 这是Flash Attention的Triton实现
import torch
import math
import triton
import triton.language as tl

# Triton Kenel: Flash Attention Forward Pass
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_q, N_k, scale,
    D: tl.constexpr, Q_TITLE_SIZE: tl.constexpr, K_TITLE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Title and batch index
    query_title_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_offset = batch_index * stride_qb + query_title_index * Q_TITLE_SIZE * stride_qq
    k_offset = batch_index * stride_kb
    v_offset = batch_index * stride_vb
    o_offset = batch_index * stride_ob + query_title_index * Q_TITLE_SIZE * stride_oq
    l_offset = batch_index * stride_lb + query_title_index * K_TITLE_SIZE * stride_lq

    # Create block pointers
    Q_block_ptr = tl.make_block_ptr(Q_ptr + q_offset, (N_q, D), (stride_qq, stride_qd), (0, 0), (Q_TITLE_SIZE, D), (1, 0))
    K_base_ptr = K_ptr + k_offset
    V_base_ptr = V_ptr + v_offset
    O_block_ptr = tl.make_block_ptr(O_ptr + o_offset, (N_q, D), (stride_oq, stride_od), (0, 0), (Q_TITLE_SIZE, D), (1, 0))
    L_block_ptr = tl.make_block_ptr(L_ptr + l_offset, (N_q,), (stride_lq,), (0,), (Q_TITLE_SIZE,), (0,))

    # Initilize accumulators
    o_i = tl.zeros((Q_TITLE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TITLE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TITLE_SIZE,), -float('inf'), dtype=tl.float32)

    # Load Q block
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))

    # Loop over key tiles
    T_k = tl.cdiv(N_k, K_TITLE_SIZE)
    for j in range(T_k):
        k_tile_start = j * K_TITLE_SIZE
        K_block_ptr = tl.make_block_ptr(K_base_ptr, (D, N_k), (stride_kd, stride_kk), (0, k_tile_start), (D,K_TITLE_SIZE), (0, 1))
        V_block_ptr = tl.make_block_ptr(V_base_ptr, (N_k, D), (stride_vk, stride_kd), (k_tile_start, 0), (K_TITLE_SIZE, 0), (1, 0))

        K_j = tl.load(K_block_ptr, boundary_check=(1, 0))
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1))
        S_ij = tl.dot(Q_i.to(tl.float32), K_j.to(tl.float32)) * scale
        
        if is_causal:
            q_indices = torch.arange(q_start, q_end, device=Q.device)
            k_indices = torch.arange(k_start, k_end, device=K.device)
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            S_ij = torch.where(causal_mask, S_ij, -float('inf'))
        
        P_ij = torch.exp(S_ij - L_i.unsqueeze(1))
        dV[b, k_start:k_end] += P_ij.T @ dO_i
        dP_ij = dO_i @ V_j.T
        dS_ij = P_ij * (dP_ij - D_i.unsqueeze(1))
        dQ[b, q_start:q_end] += (dS_ij * scale) @ K_j
        dK[b, k_start:k_end] += (dS_ij * scale).T @ Q_i

    return dQ, dK, dV

# Autograd Funtion for Flash Attention


        



    

