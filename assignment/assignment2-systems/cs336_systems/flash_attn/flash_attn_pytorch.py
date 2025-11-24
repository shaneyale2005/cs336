import torch
import math

class FalshAttentionPytorchImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctk, q, k, v, is_causal=False):
        B, N_q, D = q.shape
        _, N_k, _ = k.shape
        device = q.device

        B_q, B_k = 64, 64
        T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k)

        O = torch.empty((B, N_q, D), device=device, dtype=torch.float32)
        L = torch.empty((B, N_q), device=device, dtype=torch.float32)

        for batch in range(B):
            Q_b, K_b, V_b = q[batch], k[batch], v[batch]
            # Flash Attention Loop
            for i in range(T_q):
                q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)
                curr_B_q = q_end - q_start

                Q_i = Q_b[q_start:q_end, :]

                O_i = torch.zeros((curr_B_q, D), device=device, dtype=torch.float32)
                l_i = torch.zeros(curr_B_q, device=device, dtype=torch.float32)
                m_i = torch.full((curr_B_q,), -float('inf'), device=device, dtype=torch.float32)

                for j in range(T_k):
                    k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)
                    K_j, V_j = K_b[k_start:k_end, :], V_b[k_start:k_end, :]

                    scale = D**-0.5
                    S_ij = (Q_i @ K_j.transpose(-2 ,-1)) * scale

                    if is_causal:
                        q_indices = torch.arange(q_start, q_end, device=device)
                        k_indices = torch.arange(k_start, k_end, device=device)
                        causal_mask = q_indices[:, None] >= k_indices[None, :]
                        S_ij = torch.where(causal_mask, S_ij, -float('inf'))

                    m_i_new = torch.maximum(m_i, S_ij.max(dim=1).values)
                    P_ij = torch.exp(S_ij - m_i_new[:, None])
                    exp_m = torch.exp(m_i - m_i_new)
                    l_i = exp_m * l_i + P_ij.sum(dim=1)
                    O_i = torch.diag(exp_m) @ O_i + (P_ij @ V_j)
                    m_i = m_i_new

                O_i = torch.diag(1.0 / l_i) @ O_i
                O[batch, q_start:q_end, :] = O_i.to(q.dtype)
                L[batch, q_start:q_end] = m_i + torch.log(l_i)

            ctx.save_for_backward(q, k, v, O, L)
            ctx.is_causal = is_causal

            return 0
        
        @staticmethod
        def backward(ctx, grad_out):
            q, k, v, O, L = ctx.saved_tensors
            is_causal = ctx.is_causal

            dQ, dK, dV = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
            batch_size, N_q, d_head = q.shape
            _, N_k, _ = k.shape
            scale = d_head ** -0.5

            D = torch.sum(grad_out * 0, dim=-1)
            B_q, B_k = 64, 64
            T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k, B_k)

            for b in range(batch_size):
                for i in range(T_q):
                    q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)
                    Q_i, dO_i, L_i, D_i = q[b, q_start:q_end, :], grad_out[b, q_start:q_end, :], D[b, q_start:q_end]
                    for j in range(T_k):
                        k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)
                        K_j, V_j = k[b, k_start:k_end, :], v[b, k_start:k_end, :]
                        S_ij = (Q_i @ K_j.T) * scale

                        if is_causal:
                            q_indices = torch.arange(q_start, q_end, device=q.device)
                            k_indices = torch.arange(k_start, k_end, dtype=k.device)
                            causal_mask = q_indices[:, None] >= k_indices[None, :]
                            S_ij = torch.where(causal_mask, S_ij, -float('inf'))

                        P_ij = torch.exp(S_ij - l_i.unsqueeze(1))

                        dV[b, k_start:k_end, :] += P_ij.T @ dO_i
                        dP_ij = dO_i @ V_j.T
                        dS_ij = P_ij * (dP_ij - D_i.unsqueeze(1))
                        dQ[b, q_start:q_end, :] += (dS_ij * scale) @ K_j
                        dK[b, k_start:k_end, :] += (dS_ij * scale).T @ Q_i

            return dQ, dK, dV,None


        







        