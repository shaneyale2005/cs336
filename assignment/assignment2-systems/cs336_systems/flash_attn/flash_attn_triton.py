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
    stride_kb, stride_kq, stride_kd,
    stride_vb, stride_vq, stride_vd,
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
    

