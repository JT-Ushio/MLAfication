import types
from typing import Callable, Optional, Tuple

import torch
from transformers import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    eager_attention_forward,
    rotate_half,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)


def _get_attention_interface(module, output_attentions: bool = False):
    if module.config._attn_implementation == "eager":
        return eager_attention_forward

    if module.config._attn_implementation == "sdpa" and output_attentions:
        logger.warning_once(
            "`torch.nn.functional.scaled_dot_product_attention` does not support "
            "`output_attentions=True`. Falling back to eager attention."
        )
        return eager_attention_forward

    if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
        return ALL_ATTENTION_FUNCTIONS.get_interface(
            module.config._attn_implementation, eager_attention_forward
        )
    return ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]


def _create_apply_partial_rope(layer_rope_indices: torch.Tensor):
    # layer_rope_indices: [topk, num_kv_heads]
    def apply_partial_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # q: [B, num_q_heads, L, 2 * topk]
        # k: [B, num_kv_heads, L, 2 * topk]
        if cos.dim() == 2:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        rope_idx_k = layer_rope_indices.to(k.device).t().contiguous()
        group_size = q.size(1) // k.size(1)
        rope_idx_q = rope_idx_k.repeat_interleave(group_size, dim=0).contiguous()

        def gather_per_head(x, head_idx):
            batch_size, seq_len, _ = x.shape
            num_heads, topk = head_idx.shape
            x = x.unsqueeze(1).expand(batch_size, num_heads, seq_len, x.size(-1))
            idx = head_idx.unsqueeze(0).unsqueeze(2).expand(
                batch_size, num_heads, seq_len, topk
            )
            y = torch.gather(x, dim=-1, index=idx)
            return torch.cat([y, y], dim=-1)

        cos_q = gather_per_head(cos, rope_idx_q)
        sin_q = gather_per_head(sin, rope_idx_q)
        cos_k = gather_per_head(cos, rope_idx_k)
        sin_k = gather_per_head(sin, rope_idx_k)

        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
        return q_embed, k_embed

    return apply_partial_rope


def _split_kv_head_weight(weight: torch.Tensor, num_kv_heads: int):
    return weight.reshape(num_kv_heads, weight.shape[0] // num_kv_heads, weight.shape[1])


def _repeat_for_query_heads(weight: torch.Tensor, num_q_heads: int):
    if num_q_heads % weight.size(0) != 0:
        raise ValueError(
            f"num_attention_heads ({num_q_heads}) must be divisible by "
            f"num_key_value_heads ({weight.size(0)})."
        )
    group_size = num_q_heads // weight.size(0)
    return weight.repeat_interleave(group_size, dim=0)


def _project_latent_to_kv(attn, latent_states: torch.Tensor, num_kv_heads: int):
    # latent_states: [B, L, rank]
    if hasattr(attn.kv_proj, "c_norm"):
        latent_for_key = attn.kv_proj.c_norm(latent_states)
    else:
        latent_for_key = latent_states

    input_shape = latent_states.shape[:-1]
    k_c = attn.kv_proj.up_kc(latent_for_key)
    v = attn.kv_proj.up_v(latent_states)
    k_c = k_c.view(*input_shape, num_kv_heads, -1).transpose(1, 2)
    value_states = v.view(*input_shape, num_kv_heads, -1).transpose(1, 2)
    return k_c, value_states


def _latent_decode_attention(
    attn,
    q_r: torch.Tensor,
    q_c: torch.Tensor,
    k_r: torch.Tensor,
    latent_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    **kwargs: Unpack[FlashAttentionKwargs],
):
    # q_r/q_c: [B, num_q_heads, q_len, dim]
    # k_r: [B, num_kv_heads, kv_len, rope_dim]
    # latent_states: [B, kv_len, rank]
    num_q_heads = q_c.size(1)
    num_kv_heads = k_r.size(1)

    up_kc_weight = _split_kv_head_weight(attn.kv_proj.up_kc.weight, num_kv_heads)
    up_kc_weight = _repeat_for_query_heads(up_kc_weight, num_q_heads)
    q_c_latent = torch.einsum("bhld,hdr->bhlr", q_c, up_kc_weight)

    if hasattr(attn.kv_proj, "c_norm"):
        latent_for_key = attn.kv_proj.c_norm(latent_states)
    else:
        latent_for_key = latent_states

    latent_key_states = latent_for_key.unsqueeze(1).expand(-1, num_kv_heads, -1, -1)
    latent_value_states = latent_states.unsqueeze(1).expand(-1, num_kv_heads, -1, -1)
    query_states = torch.cat([q_r, q_c_latent], dim=-1)
    key_states = torch.cat([k_r, latent_key_states], dim=-1)

    latent_output, attn_weights = eager_attention_forward(
        attn,
        query_states,
        key_states,
        latent_value_states,
        attention_mask,
        dropout=0.0 if not attn.training else attn.attention_dropout,
        scaling=attn.scaling,
        **kwargs,
    )

    up_v_weight = _split_kv_head_weight(attn.kv_proj.up_v.weight, num_kv_heads)
    up_v_weight = _repeat_for_query_heads(up_v_weight, num_q_heads)
    attn_output = torch.einsum("blhr,hdr->blhd", latent_output, up_v_weight)

    if attn.kv_proj.up_v.bias is not None:
        up_v_bias = attn.kv_proj.up_v.bias.view(num_kv_heads, -1)
        up_v_bias = _repeat_for_query_heads(up_v_bias, num_q_heads)
        attn_output = attn_output + up_v_bias.view(1, 1, num_q_heads, -1)

    return attn_output, attn_weights


def latent_Qwen3Attention_instance_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    input_shape = hidden_states.shape[:-1]
    num_kv_heads = self.config.num_key_value_heads
    num_q_heads = self.config.num_attention_heads

    q_shape = (*input_shape, num_q_heads, -1)
    k_r_shape = (*input_shape, num_kv_heads, -1)
    use_latent_cache = (
        getattr(self.kv_proj, "has_down_proj", False)
        and hasattr(self.kv_proj, "down_kv")
        and not hasattr(self, "k_norm")
    )

    k_r = self.k_r_proj(hidden_states).view(k_r_shape)
    if hasattr(self, "k_r_norm"):
        k_r = self.k_r_norm(k_r)
    k_r = k_r.transpose(1, 2)

    q = self.q_norm(self.q_proj(hidden_states).view(q_shape))
    q_r = q[:, :, :, : k_r.size(-1)].transpose(1, 2)
    q_c = q[:, :, :, k_r.size(-1) :].transpose(1, 2)

    if use_latent_cache:
        latent_states = self.kv_proj.down_kv(hidden_states)
    else:
        k_c_shape = (*input_shape, num_kv_heads, -1)
        v_shape = (*input_shape, num_kv_heads, -1)
        k_c, v = self.kv_proj(hidden_states)
        k_c = k_c.view(k_c_shape).transpose(1, 2)
        value_states = v.view(v_shape).transpose(1, 2)

        if hasattr(self, "k_norm"):
            k = torch.cat([k_r, k_c], dim=-1)
            k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)
            k_r, k_c = k[..., : k_r.size(-1)], k[..., k_r.size(-1) :]

    cos, sin = position_embeddings
    q_r, k_r = self.mla_special_rope_fn(q_r, k_r, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if use_latent_cache:
            k_r, cached_latent_states = past_key_values.update(
                k_r, latent_states.unsqueeze(1), self.layer_idx, cache_kwargs
            )
            if cached_latent_states.size(1) != 1:
                raise ValueError(
                    "MLA latent cache expects latent states with one cache head."
                )
            latent_states = cached_latent_states.squeeze(1)
        else:
            key_states = torch.cat([k_r, k_c], dim=-1)
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            k_r, k_c = key_states[..., : k_r.size(-1)], key_states[..., k_r.size(-1) :]

    if use_latent_cache and hidden_states.shape[-2] == 1:
        attn_output, attn_weights = _latent_decode_attention(
            self,
            q_r,
            q_c,
            k_r,
            latent_states,
            attention_mask,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    if use_latent_cache:
        k_c, value_states = _project_latent_to_kv(self, latent_states, num_kv_heads)

    query_states = torch.cat([q_r, q_c], dim=-1)
    key_states = torch.cat([k_r, k_c], dim=-1)

    attention_interface: Callable = _get_attention_interface(
        self, kwargs.get("output_attentions", False)
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self, "sliding_window", None),
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def patch_mla_fwd_instance(model):
    if getattr(model.config, "mla_knorm_mode", None) == "exact" or any(
        hasattr(layer.self_attn, "k_norm") for layer in model.model.layers
    ):
        raise ValueError(
            "Exact KNorm requires reconstructed per-head keys. Use patching_mla_fwd "
            "with an ordinary dynamic KV cache; latent-cache/MQA decoding is unsupported."
        )
    model_type = model.config.model_type
    all_rope_indices = torch.as_tensor(model.config.rope_indices, dtype=torch.long)

    for _, module in model.named_modules():
        patch_func = None

        if model_type == "qwen3" or model_type == "qwen3_moe":
            from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
            from transformers.models.qwen3_moe.modeling_qwen3_moe import (
                Qwen3MoeAttention,
            )

            if isinstance(module, Qwen3Attention) or isinstance(
                module, Qwen3MoeAttention
            ):
                patch_func = latent_Qwen3Attention_instance_forward

        if patch_func is not None:
            layer_rope_indices = all_rope_indices[module.layer_idx]
            module.mla_special_rope_fn = _create_apply_partial_rope(layer_rope_indices)
            module.forward = types.MethodType(patch_func, module)

    return model
