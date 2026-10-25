import torch
import torch.nn as nn
import torch.distributed as dist
from transformers.models.qwen3 import modeling_qwen3


class MultiWeightRMSNorm(nn.Module):
    def __init__(self, gamma, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(gamma.clone().detach())
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LowRankKVLinear(nn.Module):
    """
    A low-rank approximation of a linear layer.
    Instead of storing a full matrix W of shape (out_features, in_features),
    it stores two matrices: down of shape (in_features, low_rank) and
    up of shape (low_rank, out_features), such that W ≈ up.T @ down.T
    """

    def __init__(
        self,
        d_in,
        d_c_mid,
        d_kc_out,
        d_v_out,
        n_head,
        d_head,
        has_down_proj=False,
        has_c_norm=False,
        bias=True,
    ):
        super().__init__()
        if has_down_proj:  # (x * W_q）* （x * down_kv * up_kc）
            self.down_kv = nn.Linear(in_features=d_in, out_features=d_c_mid, bias=False)
        d_kv_in = d_c_mid if has_down_proj else d_in
        self.up_kc = nn.Linear(in_features=d_kv_in, out_features=d_kc_out, bias=False)
        self.up_v = nn.Linear(in_features=d_kv_in, out_features=d_v_out, bias=bias)
        if has_c_norm:
            self.c_norm = modeling_qwen3.Qwen3RMSNorm(d_c_mid, eps=eps)
        self.has_down_proj, self.bias = has_down_proj, bias

    def reset_parameters(
        self,
        down_kv_weight=None,
        up_kc_weight=None,
        up_v_weight=None,
        up_v_bias=None,
    ):
        self.up_kc.weight.data.copy_(up_kc_weight)
        self.up_v.weight.data.copy_(up_v_weight)
        if down_kv_weight is not None:
            self.down_kv.weight.data.copy_(down_kv_weight)
        if up_v_bias is not None:
            self.up_v.bias.data.copy_(up_v_bias)

    def mha_forward(self, x):
        # x: (batch_size, seq_len, in_features)
        if self.has_down_proj:
            x = self.down_kv(x)
        if hasattr(self, "c_norm"):
            kc = self.up_kc(self.c_norm(x))
        else:
            kc = self.up_kc(x)
        v = self.up_v(x)
        return kc, v
    def forward(self, x):
        return self.mha_forward(x)

    def mla_forward(self, x, q_hid, o_hid):
        kv = self.down_kv(x)
        # TODO: Tilelang kernel
        return kv


def patch_mlafication(model, model_args, mha2mla_args):
    global eps
    eps = model_args.rms_norm_eps
    """
    Patch a huggingface model by:
    1. Reordering rows in q_proj and k_proj matrices based on a partial-rope mask
    2. Replacing v_proj with a low-rank approximation

    Args:
        model: The Qwen2ForCausalLM model
        mask: A binary mask for reordering q_proj and k_proj
        low_rank: The rank for the low-rank approximation of v_proj
    """
    n_k_head, n_head = model_args.num_key_value_heads, model_args.num_attention_heads
    d_head = model_args.head_dim
    mla_tensor = torch.load(
        mha2mla_args.mla_tensor_path, map_location="cpu", weights_only=True
    )
    # Legacy tensors absorb gamma_k / E[RMS(K)] into the key projections.
    # Exact tensors instead carry gamma_k and keep those projections unscaled.
    knorm_mode = "exact" if "gamma_k" in mla_tensor else "calibrated"
    for requested in (getattr(mha2mla_args, "knorm_mode", None),
                      getattr(model_args, "mla_knorm_mode", None)):
        if requested is not None and requested != knorm_mode:
            raise ValueError(
                f"KNorm mode mismatch: requested {requested!r}, but MLA tensors "
                f"are {knorm_mode!r}. Regenerate tensors with --knorm_mode {requested}; "
                "calibrated key weights cannot be used with exact KNorm."
            )
    model.config.mla_knorm_mode = knorm_mode
    mha2mla_args.knorm_mode = knorm_mode

    has_down_proj = "kv_down_weight" in mla_tensor
    q_proj_shape = mla_tensor["q_proj"][0].shape
    k_r_proj_shape = mla_tensor["k_r_proj"][0].shape
    d_in = model_args.hidden_size
    d_c_mid, d_kc_out = mla_tensor["k_up_weight"][0].t().shape if has_down_proj else mla_tensor["k_c_proj"][0].t().shape
    _, d_v_out = mla_tensor["v_up_weight"][0].t().shape if has_down_proj else model.model.layers[0].self_attn.v_proj.weight.t().shape
    model.config.rope_indices = mla_tensor["rope_index"].tolist()

    for layer_idx, layer in enumerate(model.model.layers):
        # 1. Setup RoPE part
        q_proj = nn.Linear(q_proj_shape[1], q_proj_shape[0], bias=False)
        k_r_proj = nn.Linear(k_r_proj_shape[1], k_r_proj_shape[0], bias=False)
        q_proj.weight.data.copy_(mla_tensor["q_proj"][layer_idx])
        k_r_proj.weight.data.copy_(mla_tensor["k_r_proj"][layer_idx])

        # 2. Setup NoPE part
        kv_proj = LowRankKVLinear(
            d_in,
            d_c_mid,
            d_kc_out,
            d_v_out,
            n_head,
            d_head,
            has_down_proj=has_down_proj,
            has_c_norm=False,
            bias=False,
        )

        kv_proj.reset_parameters(
            down_kv_weight=mla_tensor["kv_down_weight"][layer_idx]
            if has_down_proj
            else None,
            up_kc_weight=mla_tensor["k_up_weight"][layer_idx]
            if has_down_proj
            else mla_tensor["k_c_proj"][layer_idx],
            up_v_weight=mla_tensor["v_up_weight"][layer_idx]
            if has_down_proj
            else model.model.layers[layer_idx].self_attn.v_proj.weight.data,
            up_v_bias=None,
        )

        # 3. Register proj & Delete original k_proj and v_proj
        layer.self_attn.q_proj = q_proj
        layer.self_attn.k_r_proj = k_r_proj
        layer.self_attn.kv_proj = kv_proj
        delattr(layer.self_attn, "k_proj")
        delattr(layer.self_attn, "v_proj")
        delattr(layer.self_attn, "q_norm")
        delattr(layer.self_attn, "k_norm")
        layer.self_attn.q_norm = MultiWeightRMSNorm(
            mla_tensor["gamma_q"][layer_idx].view(n_head, d_head), eps=eps
        )
        if knorm_mode == "exact":
            layer.self_attn.k_norm = MultiWeightRMSNorm(
                mla_tensor["gamma_k"][layer_idx].view(n_k_head, d_head), eps=eps
            )

        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            print(f"Layer {layer_idx}: Set up q_r_proj, k_r_proj, and kv_proj")

    return model
