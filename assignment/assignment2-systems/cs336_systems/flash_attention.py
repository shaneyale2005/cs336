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

    