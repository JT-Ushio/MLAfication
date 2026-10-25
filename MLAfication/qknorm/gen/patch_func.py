import sys
import torch
import inspect
from transformers.models.qwen2.modeling_qwen2 import rotate_half


# Monkey patching to use custom partial-rope function
def create_custom_apply_rotary_pos_emb(partial_rope_uniform_step):
    def custom_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        step = partial_rope_uniform_step
        for i in range(0, q.size(-1), step):
            q_embed[..., i + 1 : i + step] = q[..., i + 1 : i + step]
            k_embed[..., i + 1 : i + step] = k[..., i + 1 : i + step]
        return q_embed, k_embed

    return custom_apply_rotary_pos_emb

def create_custom_apply_rotary_pos_emb_mha2mla(idx_mat, topk):
    def custom_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # print("uniform pos emb")
        # Get the calling frame
        group_size = q.size(1) // k.size(1)  # num_heads / num_kv_heads

        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        # ---- 构造 head_dim 级 mask ----
        n_freqs, device = idx_mat.size(0), q.device
        head_dim = q.size(-1)
        freq_mask_k = torch.ones(n_freqs, k.size(1), dtype=torch.bool, device=device)
        freq_mask_k[idx_mat[:topk], torch.arange(k.size(1), device=device)] = False
        freq_mask_q = freq_mask_k.repeat_interleave(group_size, dim=1)
        mask_q = freq_mask_q.repeat(2, 1)   # [head_dim, num_heads]
        mask_k = freq_mask_k.repeat(2, 1)   # [head_dim, num_kv_heads]
        q_embed = torch.where(mask_q.t().view(1, q.size(1), 1, head_dim), q, q_embed)
        k_embed = torch.where(mask_k.t().view(1, k.size(1), 1, head_dim), k, k_embed)

        return q_embed, k_embed

    return custom_apply_rotary_pos_emb

def create_custom_sepearte_rope_nope_mha2mla(idx_mat, topk):
    def custom_sepearte_rope_nope(q, k):
        # print("mha2mla pos emb")
        group_size = q.size(1) // k.size(1)

        n_freqs, device = idx_mat.size(0), q.device
        head_dim = q.size(-1)

        freq_mask_k = torch.ones(n_freqs, k.size(1), dtype=torch.bool, device=device)
        freq_mask_k[idx_mat[:topk], torch.arange(k.size(1), device=device)] = False
        freq_mask_q = freq_mask_k.repeat_interleave(group_size, dim=1)

        mask_q = freq_mask_q.repeat(2, 1)
        mask_k = freq_mask_k.repeat(2, 1)

        q_mask_view = mask_q.t().view(1, q.size(1), 1, head_dim)  # True = nope
        k_mask_view = mask_k.t().view(1, k.size(1), 1, head_dim)  # True = nope

        q_rope = torch.where(~q_mask_view, q, torch.zeros_like(q))
        k_rope = torch.where(~k_mask_view, k, torch.zeros_like(k))
        q_nope = torch.where(q_mask_view, q, torch.zeros_like(q))
        k_nope = torch.where(k_mask_view, k, torch.zeros_like(k))

        return q_rope, k_rope, q_nope, k_nope

    return custom_sepearte_rope_nope


def seperate_nope_rope(Wk, idx_mat, topk, num_heads=-1):
    idx_mat = idx_mat.long().clone()
    n_freqs, num_kv_heads = idx_mat.shape
    if num_heads == -1:
        num_heads = num_kv_heads
    idx_mat = idx_mat.repeat_interleave(num_heads // num_kv_heads, dim=1)  # [n_freqs, num_heads]

    freq_mask_k = torch.ones(n_freqs, num_heads, dtype=torch.bool, device=idx_mat.device)
    freq_mask_k[idx_mat[:topk], torch.arange(num_heads, device=idx_mat.device)] = False

    mask_k = freq_mask_k.repeat(2, 1)
    row_mask_flat = mask_k.t().contiguous().reshape(-1).to(Wk.device)  # True = nope

    Wk_rope = Wk[~row_mask_flat]
    Wk_nope = Wk[row_mask_flat]
    return Wk_rope, Wk_nope


def build_mla_layer_tensors(attn, idx_mat, topk, down, up_k, up_v,
                            knorm_mode="calibrated", calibrated_rms=None):
    """Package the same Partial-RoPE/SVD factors with calibrated or exact KNorm.

    Exact mode leaves both key projections unscaled. Its gamma_k follows the
    per-head [RoPE, NoPE] permutation, so RMSNorm can be applied to the complete
    reconstructed key before rotation. Never apply exact KNorm to weights that
    already contain gamma_k / calibrated_rms.
    """
    if knorm_mode not in {"calibrated", "exact"}:
        raise ValueError(f"Unknown knorm_mode: {knorm_mode!r}")
    nq = attn.config.num_attention_heads
    nkv = attn.config.num_key_value_heads
    wq = attn.q_proj.weight.detach()
    wqr, wqc = seperate_nope_rope(wq, idx_mat, topk, num_heads=nq)
    wq = torch.cat([wqr.reshape(nq, -1, wq.size(1)),
                    wqc.reshape(nq, -1, wq.size(1))], dim=1).reshape_as(wq)
    gqr, gqc = seperate_nope_rope(attn.q_norm.weight.detach().repeat(nq),
                                 idx_mat, topk, num_heads=nq)
    gamma_q = torch.cat([gqr.reshape(nq, -1), gqc.reshape(nq, -1)], dim=1)
    wk = attn.k_proj.weight.detach()
    wkr, _ = seperate_nope_rope(wk, idx_mat, topk)
    up_k = up_k.to(device=wk.device, dtype=wk.dtype)
    gkr, gkc = seperate_nope_rope(attn.k_norm.weight.detach().repeat(nkv),
                                 idx_mat, topk)
    gamma_k = torch.cat([gkr.reshape(nkv, -1), gkc.reshape(nkv, -1)], dim=1)
    if knorm_mode == "calibrated":
        if calibrated_rms is None:
            raise ValueError("calibrated KNorm requires one RMS value per KV head")
        rms = calibrated_rms.to(device=wk.device).reshape(nkv, 1)
        if not torch.isfinite(rms).all() or not (rms > 0).all():
            raise ValueError("calibrated RMS values must be finite and positive")
        wkr = (gkr.reshape(nkv, -1) / rms).reshape(-1, 1) * wkr
        up_k = up_k * gkc[:, None]
        up_k = (up_k.reshape(nkv, -1, down.size(0)) / rms[:, :, None]).reshape_as(up_k)
    tensors = {
        "q_proj": wq,
        "gamma_q": gamma_q.flatten(),
        "k_r_proj": wkr,
        "kv_down_weight": down,
        "v_up_weight": up_v,
        "k_up_weight": up_k,
    }
    if knorm_mode == "exact":
        tensors["gamma_k"] = gamma_k.flatten()
    tensors = {name: value.detach().to(dtype=torch.bfloat16) for name, value in tensors.items()}
    tensors["rope_index"] = torch.sort(idx_mat[:topk], dim=0).values.long()
    return tensors

def apply_U(x, repeat, U, parts):
    nh = x.size(-2)
    n_freqs = x.size(-1) // 2
    for h_idx in range(nh):
        for p_idx, (l, r) in enumerate(parts):
            u = U[h_idx // repeat][p_idx]
            x[..., h_idx, l: r + 1] = x[..., h_idx, l: r + 1] @ u # real
            x[..., h_idx, n_freqs + l: n_freqs + r + 1] = x[..., h_idx, n_freqs + l: n_freqs + r + 1] @ u # imag
    return x
