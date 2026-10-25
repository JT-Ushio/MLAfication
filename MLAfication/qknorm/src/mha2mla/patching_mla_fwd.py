import types
from typing import Callable, Optional, Tuple

import torch
from transformers import Cache
from transformers.utils import logging
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    rotate_half,
    eager_attention_forward,
)

logger = logging.get_logger(__name__)


def create_custom_apply_rotary_pos_emb(layer_rope_indices: torch.Tensor):
    # layer_rope_indices: [topk, num_kv_heads]
    def custom_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # q: [B, num_q_heads, L, 2*topk]
        # k: [B, num_kv_heads, L, 2*topk]

        if cos.dim() == 2:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        # [topk, nkv] -> [nkv, topk]
        rope_idx_k = layer_rope_indices.to(k.device).t().contiguous()
        group_size = q.size(1) // k.size(1)
        rope_idx_q = rope_idx_k.repeat_interleave(
            group_size, dim=0
        ).contiguous()  # [nq, topk]

        def gather_per_head(x, head_idx):
            # x: [B, L, Dbase]
            # head_idx: [H, topk]
            B, L, _ = x.shape
            H, topk = head_idx.shape
            x = x.unsqueeze(1).expand(B, H, L, x.size(-1))  # [B, H, L, Dbase]
            idx = (
                head_idx.unsqueeze(0).unsqueeze(2).expand(B, H, L, topk)
            )  # [B, H, L, topk]
            y = torch.gather(x, dim=-1, index=idx)  # [B, H, L, topk]
            y = torch.cat([y, y], dim=-1)  # [B, H, L, 2*topk]
            return y

        cos_q = gather_per_head(cos, rope_idx_q)
        sin_q = gather_per_head(sin, rope_idx_q)
        cos_k = gather_per_head(cos, rope_idx_k)
        sin_k = gather_per_head(sin, rope_idx_k)

        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
        return q_embed, k_embed

    return custom_apply_rotary_pos_emb


def custom_Qwen3Attention_instance_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    # Transformers 4.x passes the cache under the singular attention argument.
    # Consume it here instead of silently dropping cached keys during decoding.
    legacy_cache = kwargs.pop("past_key_value", None)
    if past_key_values is None:
        past_key_values = legacy_cache
    elif legacy_cache is not None and legacy_cache is not past_key_values:
        raise ValueError("Conflicting past_key_value and past_key_values caches")
    input_shape = hidden_states.shape[:-1]
    num_kv_heads = self.config.num_key_value_heads
    num_q_heads = self.config.num_attention_heads

    q_shape = (*input_shape, num_q_heads, -1)
    k_r_shape = (*input_shape, num_kv_heads, -1)
    k_c_shape = (*input_shape, num_kv_heads, -1)
    v_shape = (*input_shape, num_kv_heads, -1)

    k_r = self.k_r_proj(hidden_states).view(k_r_shape)
    if hasattr(self, "k_r_norm"):
        k_r = self.k_r_norm(k_r)
    k_r = k_r.transpose(1, 2)

    q = self.q_norm(self.q_proj(hidden_states).view(q_shape))
    q_r = q[:, :, :, : k_r.size(-1)].transpose(1, 2)
    q_c = q[:, :, :, k_r.size(-1) :].transpose(1, 2)

    k_c, v = self.kv_proj(hidden_states)
    k_c = k_c.view(k_c_shape).transpose(1, 2)
    value_states = v.view(v_shape).transpose(1, 2)

    if hasattr(self, "k_norm"):
        # Normalize all reconstructed key dimensions jointly, before Partial-RoPE.
        # Normalizing the RoPE and NoPE parts separately changes the model.
        k = torch.cat([k_r, k_c], dim=-1)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)
        k_r, k_c = k[..., : k_r.size(-1)], k[..., k_r.size(-1) :]

    cos, sin = position_embeddings
    q_r, k_r = self.mla_special_rope_fn(q_r, k_r, cos, sin)

    query_states = torch.cat([q_r, q_c], dim=-1)
    key_states = torch.cat([k_r, k_c], dim=-1)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True`. Falling back to eager attention."
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def patch_mla_fwd_instance(model):
    model_type = model.config.model_type
    all_rope_indices = torch.as_tensor(
        model.config.rope_indices, dtype=torch.long
    )  # [layers, topk, nkv]

    for _, module in model.named_modules():
        patch_func = None

        if model_type == "llama":
            from transformers.models.llama.modeling_llama import LlamaAttention

            if isinstance(module, LlamaAttention):
                patch_func = custom_LlamaAttention_instance_forward
        elif model_type == "qwen2":
            from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

            if isinstance(module, Qwen2Attention):
                patch_func = custom_LlamaAttention_instance_forward
        elif model_type == "qwen3" or model_type == "qwen3_moe":
            from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
            from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

            if isinstance(module, Qwen3Attention) or isinstance(module, Qwen3MoeAttention):
                patch_func = custom_Qwen3Attention_instance_forward

        if patch_func is not None:
            layer_rope_indices = all_rope_indices[module.layer_idx]  # [topk, nkv]
            module.mla_special_rope_fn = create_custom_apply_rotary_pos_emb(
                layer_rope_indices
            )
            module.forward = types.MethodType(patch_func, module)

    return model
