import torch
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TITLE_SIZE: tl.constexpr,
    K_TITLE_SIZE: tl.constexpr,
    is_causal: tl.constexpr):
    # 块索引
    query_title_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 块信息
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (query_title_index * Q_TITLE_SIZE, 0),
        block_shape = (Q_TITLE_SIZE, D),
        order = (1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (0, 0),
        block_shape = (K_TITLE_SIZE, D),
        order = (1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (0, 0),
        block_shape = (K_TITLE_SIZE, D),
        order = (1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_ob,
        shape = (N_QUERIES, D),
        strides = (stride_oq, stride_od),
        offsets = (query_title_index * Q_TITLE_SIZE, 0),
        block_shape = (Q_TITLE_SIZE, D),
        order = (1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides = (stride_lq,),
        offsets = (query_title_index * Q_TITLE_SIZE),
        block_shape = (Q_TITLE_SIZE,),
        order = (0,),
    )

    Q_title = tl.load(Q_block_ptr, boundary_check=(0,), padding_option = "zero")
    O_i = tl.zeros((Q_TITLE_SIZE, D), dtype = tl.float32)
    m_i = tl.full((Q_TITLE_SIZE,), float("-inf"), dtype = tl.float32)
    l_i = tl.zeros((Q_TITLE_SIZE,), dtype = tl.float32)

    log2e: tl.constexpr = 1.44269504

    if is_causal:
        q_pos = tl.arange(0, Q_TITLE_SIZE) + query_title_index * Q_TITLE_SIZE
    
    n_k_tiles = tl.cdiv(N_KEYS, K_TITLE_SIZE)
    q_valid_len = min(Q_TITLE_SIZE, N_QUERIES - query_title_index + Q_TITLE_SIZE)
    q_mask = tl.arange(0, Q_TITLE_SIZE) < q_valid_len

    for i in range(n_k_tiles):
        K_title = tl.load(K_block_ptr, boundary_check=(0,), padding_option = "zero")
        V_title = tl.load(V_block_ptr, boundary_check = (0,), padding_option = "zero")
        k_valid_len = min(K_TITLE_SIZE, N_KEYS - i * K_TITLE_SIZE)


        # 计算注意力分数
        K_mask = tl.arange(0, K_TITLE_SIZE) < k_valid_len
        boundary_mask = q_mask[:, None] & K_mask[None, :]

        S_ij = tl.dot(Q_title, tl.trans(K_title)) * scale + tl.where(boundary_mask, 0, -1.0e6)

        if is_causal:
            k_pos = tl.arange(0, K_TITLE_SIZE) + i * K_TITLE_SIZE
            mask = q_pos[:, None] >= k_pos[None, :]
            S_ij = tl.where(mask, S_ij, float('-inf'))
        
        m_curr = tl.maximum(m_i, tl.max(S_ij, axis = -1))

 

    
