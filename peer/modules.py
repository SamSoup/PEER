import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverCompressor(nn.Module):
    """
    Cross-attends a small set of latent queries into a token sequence to
    produce compressed memory/query tokens.
    """

    def __init__(self, d_h: int = 256, m: int = 8, n_heads: int = 8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(m, d_h) * 0.02)
        self.attn = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_h)
        self.ff = nn.Sequential(
            nn.Linear(d_h, 4 * d_h),
            nn.GELU(),
            nn.Linear(4 * d_h, d_h),
        )
        self.ln2 = nn.LayerNorm(d_h)

    def forward(self, T):
        B = T.size(0)
        Q = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, m, d_h]
        M, _ = self.attn(Q, T, T, need_weights=False)  # [B, m, d_h]
        M = Q + M
        M = M + self.ff(self.ln2(M))
        return self.ln1(M)


class KeyReadout(nn.Module):
    """Attention readout from memory tokens using a learned query."""

    def __init__(self, d_h: int = 256, n_heads: int = 8):
        super().__init__()
        self.q_key = nn.Parameter(torch.randn(1, 1, d_h) * 0.02)
        self.attn = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_h)

    def forward(self, M):
        # M: [B, m, d_h]
        B = M.size(0)
        q = self.q_key.expand(B, -1, -1)  # [B,1,d_h]
        out, _ = self.attn(q, M, M, need_weights=False)
        out = self.ln(out + q)
        return out.squeeze(1)  # [B,d_h]


class ScalarLabelEmbedder(nn.Module):
    """Embeds scalar labels with running normalization stats."""

    def __init__(self, d_h: int = 256):
        super().__init__()
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, d_h),
        )

    def set_stats(self, mean: float, std: float):
        self.y_mean.fill_(mean)
        self.y_std.fill_(std)

    def forward(self, y):
        y = y.view(-1, 1)
        yn = (y - self.y_mean) / (self.y_std + 1e-6)
        e = self.mlp(yn).unsqueeze(1)  # [B, 1, d_h]
        return e


class InferenceLayer(nn.Module):
    def __init__(self, d_h: int = 256, n_heads: int = 8):
        super().__init__()
        self.attn_q = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.attn_m = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_h)
        self.ln2 = nn.LayerNorm(d_h)
        self.ff = nn.Sequential(
            nn.Linear(d_h, 4 * d_h),
            nn.GELU(),
            nn.Linear(4 * d_h, d_h),
        )
        self.ln3 = nn.LayerNorm(d_h)

    def forward(self, R, Qq, Mem, return_q_mem_attn: bool = False):
        R2, _ = self.attn_q(R, Qq, Qq, need_weights=False)
        R = self.ln1(R + R2)

        R2, attn_r_mem = self.attn_m(
            R,
            Mem,
            Mem,
            need_weights=True,
            average_attn_weights=False,
        )
        R = self.ln2(R + R2)

        R = self.ln3(R + self.ff(R))
        if return_q_mem_attn:
            # Query-conditioned retrieval attention for explanations.
            _, attn_q_mem = self.attn_m(
                Qq,
                Mem,
                Mem,
                need_weights=True,
                average_attn_weights=False,
            )
            return R, attn_r_mem, attn_q_mem
        return R, attn_r_mem

    def attn_m_with_token_contrib(self, R, Mem):
        """
        Compute R->Mem cross-attention and per-memory-token contribution vectors
        using the exact attn_m weights (no approximation).
        Returns:
          attn: [B, H, Tq, N]
          attn_out: [B, Tq, E]
          contrib_out: [B, Tq, N, E]
        """
        mha = self.attn_m
        B, Tq, E = R.shape
        N = Mem.size(1)
        H = mha.num_heads
        Dh = E // H
        if Dh * H != E:
            raise ValueError(f"d_h={E} not divisible by num_heads={H}")
        if mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn:
            raise ValueError("Unsupported MHA config for token contributions.")

        W = mha.in_proj_weight
        b = mha.in_proj_bias
        bq = b[:E] if b is not None else None
        bk = b[E : 2 * E] if b is not None else None
        bv = b[2 * E :] if b is not None else None

        q_proj = F.linear(R, W[:E, :], bq)
        k_proj = F.linear(Mem, W[E : 2 * E, :], bk)
        v_proj = F.linear(Mem, W[2 * E :, :], bv)

        q = q_proj.view(B, Tq, H, Dh).transpose(1, 2)  # [B,H,Tq,Dh]
        k = k_proj.view(B, N, H, Dh).transpose(1, 2)  # [B,H,N,Dh]
        v = v_proj.view(B, N, H, Dh).transpose(1, 2)  # [B,H,N,Dh]

        scores = torch.matmul(q, k.transpose(-2, -1)) * (Dh ** -0.5)  # [B,H,Tq,N]
        attn = torch.softmax(scores, dim=-1)

        # Per-token head contribution before output projection.
        contrib_head = attn.unsqueeze(-1) * v.unsqueeze(2)  # [B,H,Tq,N,Dh]

        # Merge heads into model dim E per token contribution.
        contrib_E = contrib_head.permute(0, 2, 3, 1, 4).contiguous()
        contrib_E = contrib_E.view(B, Tq, N, E)  # [B,Tq,N,E]

        # Apply the exact out_proj used by MultiheadAttention.
        contrib_out = F.linear(contrib_E, mha.out_proj.weight, mha.out_proj.bias)
        attn_out = contrib_out.sum(dim=2)  # [B,Tq,E]
        return attn, attn_out, contrib_out

    def forward_with_r_mem_contrib(self, R, Qq, Mem):
        """
        Faithful forward pass that additionally returns per-memory-token
        contribution vectors for the R->Mem branch.
        """
        R2, _ = self.attn_q(R, Qq, Qq, need_weights=False)
        R = self.ln1(R + R2)

        attn_r_mem, R2, contrib_r_mem = self.attn_m_with_token_contrib(R, Mem)
        R = self.ln2(R + R2)
        R = self.ln3(R + self.ff(R))
        return R, attn_r_mem, contrib_r_mem


class InferenceHead(nn.Module):
    def __init__(self, d_h: int = 256, n_heads: int = 8, L: int = 3):
        super().__init__()
        self.R0 = nn.Parameter(torch.randn(1, d_h) * 0.02)
        self.layers = nn.ModuleList(
            [InferenceLayer(d_h, n_heads) for _ in range(L)]
        )
        self.out = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.GELU(),
            nn.Linear(d_h, 1),
        )

    def forward(
        self,
        Qq,
        Mem,
        return_q_mem_attn: bool = False,
        return_r_mem_attn_all: bool = False,
    ):
        B = Qq.size(0)
        R = self.R0.unsqueeze(0).expand(B, 1, -1)
        attn_last = None
        attn_r_mem_all = [] if return_r_mem_attn_all else None
        attn_q_mem_all = [] if return_q_mem_attn else None
        for layer in self.layers:
            if return_q_mem_attn:
                R, attn_r_mem, attn_q_mem = layer(
                    R, Qq, Mem, return_q_mem_attn=True
                )
                attn_q_mem_all.append(attn_q_mem)
            else:
                R, attn_r_mem = layer(R, Qq, Mem)
            if return_r_mem_attn_all:
                attn_r_mem_all.append(attn_r_mem)
            attn_last = attn_r_mem
        raw = self.out(R[:, 0, :])
        if return_q_mem_attn:
            return raw, attn_q_mem_all
        if return_r_mem_attn_all:
            return raw, attn_r_mem_all
        return raw, attn_last

    def forward_with_r_mem_contrib(self, Qq, Mem):
        """
        Faithful explanation forward:
        returns prediction, final readout state, per-layer R->Mem attention,
        and per-layer per-token contribution vectors.
        """
        B = Qq.size(0)
        R = self.R0.unsqueeze(0).expand(B, 1, -1)
        attn_r_all = []
        contrib_r_all = []
        for layer in self.layers:
            R, attn_r_mem, contrib_r_mem = layer.forward_with_r_mem_contrib(
                R, Qq, Mem
            )
            attn_r_all.append(attn_r_mem)
            contrib_r_all.append(contrib_r_mem)
        raw = self.out(R[:, 0, :])
        return raw, R, attn_r_all, contrib_r_all


class SlotSelector(nn.Module):
    """
    Differentiable straight-through slot assignment over cached keys/memories.
    """

    def __init__(self, K: int = 128, d_h: int = 256, T: int = 512):
        super().__init__()
        self.K = K
        self.T = T
        self.slot_q = nn.Parameter(torch.randn(K, d_h) * 0.02)

    @staticmethod
    def l2norm(x, eps: float = 1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(self, keys, mem_cache, y_cache, tau: float):
        """
        Masked sequential selection without duplicates (hard picks).
        Returns:
          Mem_sel: [K, m, d_h] (ST-selected memories)
          idx_hard: [K] hard selected indices (no duplicates)
          logits_full: [K, N] raw slot logits
          q_probs: [K, N] soft selection probabilities (for overlap/repulsion)
          q_st: [K, N] straight-through weights (for label tokens)
          exp_keys: [K, d_h] expected keys under q_probs
        """
        device = keys.device
        keys_n = self.l2norm(keys.float())  # [N, d_h]
        slot_n = self.l2norm(self.slot_q.float())  # [K, d_h]
        logits_full = slot_n @ keys_n.T  # [K, N]

        selected_mask = torch.zeros(keys_n.size(0), dtype=torch.bool, device=device)
        mem_list = []
        idx_list = []
        q_prob_list = []
        q_st_list = []
        exp_keys_list = []

        for k in range(self.K):
            mask = selected_mask  # do not mutate logits_full history
            logits = logits_full[k].clone()
            logits = logits.masked_fill(mask, -1e9)
            topv, topi = torch.topk(logits, k=self.T, dim=0)
            g = -torch.log(-torch.log(torch.rand_like(topv) + 1e-9) + 1e-9)
            soft = torch.softmax((topv + g) / tau, dim=0)  # [T]
            hard_idx_in_T = torch.argmax(soft)
            hard = torch.zeros_like(soft)
            hard[hard_idx_in_T] = 1.0
            w = hard - soft.detach() + soft  # ST

            cand_mem = mem_cache[topi]  # [T, m, d_h]
            mem_k = (w.view(-1, 1, 1) * cand_mem.float()).sum(dim=0)  # [m,d_h]
            mem_list.append(mem_k)

            idx_hard = topi[hard_idx_in_T]
            idx_list.append(idx_hard)
            new_mask = selected_mask.clone()
            new_mask[idx_hard] = True
            selected_mask = new_mask

            q_prob_full = torch.zeros_like(logits)
            q_prob_full[topi] = soft
            q_prob_list.append(q_prob_full)

            q_st_full = torch.zeros_like(logits)
            q_st_full[topi] = w
            q_st_list.append(q_st_full)

            exp_key = (soft.view(-1, 1) * keys_n[topi]).sum(dim=0)
            exp_keys_list.append(exp_key)

        Mem_sel = torch.stack(mem_list, dim=0)
        idx_hard = torch.stack(idx_list, dim=0)
        q_probs = torch.stack(q_prob_list, dim=0)
        q_st = torch.stack(q_st_list, dim=0)
        exp_keys = torch.stack(exp_keys_list, dim=0)
        return Mem_sel, idx_hard, logits_full, q_probs, q_st, exp_keys
