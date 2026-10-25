"""Loss, batching and portable checkpoints for full-parameter EP distillation."""

import json
import math
import os
from pathlib import Path
import shutil
import types
import sys
import traceback
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file


class _ChunkedKL(torch.autograd.Function):
    """Exact full-vocabulary KL; recompute softmax by token blocks in backward."""

    @staticmethod
    def forward(ctx, student, teacher, temperature, block_rows):
        if student.shape != teacher.shape or student.ndim < 2:
            raise ValueError("Student and teacher must have identical [..., vocab] shapes")
        if temperature <= 0 or block_rows <= 0:
            raise ValueError("temperature and block_rows must be positive")
        s = student.reshape(-1, student.shape[-1])
        t = teacher.reshape_as(s)
        if not s.shape[0]:
            raise ValueError("Cannot distill an empty batch")
        ctx.save_for_backward(student, teacher)
        ctx.temperature, ctx.block_rows = temperature, block_rows
        # FP64 inputs are retained for numerical gradient tests; BF16 training
        # accumulates probabilities and KL in FP32, without truncating the vocab.
        dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
        loss = torch.zeros((), device=student.device, dtype=dtype)
        for start in range(0, s.shape[0], block_rows):
            a = s[start:start + block_rows].to(dtype) / temperature
            b = t[start:start + block_rows].to(dtype) / temperature
            log_s, log_t = F.log_softmax(a, -1), F.log_softmax(b, -1)
            loss += (log_t.exp() * (log_t - log_s)).sum()
        return loss * (temperature ** 2 / s.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        student, teacher = ctx.saved_tensors
        s = student.reshape(-1, student.shape[-1])
        t = teacher.reshape_as(s)
        grad = torch.empty_like(s)
        dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
        factor = grad_output * ctx.temperature / s.shape[0]
        for start in range(0, s.shape[0], ctx.block_rows):
            a = s[start:start + ctx.block_rows].to(dtype) / ctx.temperature
            b = t[start:start + ctx.block_rows].to(dtype) / ctx.temperature
            grad[start:start + ctx.block_rows] = (F.softmax(a, -1) - F.softmax(b, -1)) * factor
        return grad.reshape_as(student), None, None, None


def distillation_kl(student, teacher, temperature=1.0, block_rows=128):
    return _ChunkedKL.apply(student, teacher.detach(), temperature, block_rows)


def cast_bf16_parameters(model):
    # Keep rotary-frequency buffers in FP32, including newly loaded HF models.
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype=torch.bfloat16)
    return model


def _qwen_router_forward(self, x, expert_bias=None):
    """Retain Qwen's sorted top-k and BF16 routing weights after AutoEP packing."""
    from deepspeed.moe.ep_router import count_tokens_per_expert
    logits = self.gate(x)
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    selected_weights, selected = torch.topk(probabilities, self.top_k, dim=-1, sorted=True)
    if self.route_norm:
        selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)
    counts = count_tokens_per_expert(selected, self.num_experts)
    return selected_weights.to(logits.dtype), selected, counts


def preserve_qwen_routing(model):
    for layer in model.model.layers:
        router = layer.mlp.router
        if router.score_func != 'softmax' or router.route_scale != 1.0 or router.num_expert_groups is not None:
            raise ValueError('Only the original Qwen softmax top-k router is supported')
        if layer.mlp.expert_bias is not None and torch.count_nonzero(layer.mlp.expert_bias).item():
            raise ValueError('Load-balancing bias would change the distillation control')
        router.forward = types.MethodType(_qwen_router_forward, router)


class FrozenEPTeacher(torch.nn.Module):
    """AutoEP inference without DeepSpeed's optimizer construction path."""

    def __init__(self, model, config, device):
        super().__init__()
        from deepspeed.module_inject.auto_ep import AutoEP
        from deepspeed.module_inject.auto_ep_config import AutoEPConfig
        from deepspeed.utils import groups

        ep_size = config['ep_size']
        if dist.get_world_size() != ep_size:
            raise ValueError('Teacher requires world_size == ep_size')
        groups._create_expert_and_data_parallel(expert_parallel_size_=ep_size)
        group = groups._get_expert_parallel_group(f'ep_size_{ep_size}')
        options = engine_config(config, 1, dist.get_world_size(), teacher=True)['expert_parallel']
        auto_ep = AutoEP(model, AutoEPConfig(**options))
        specs = auto_ep.ep_parser()
        if not specs:
            raise RuntimeError('No teacher MoE layers detected')
        auto_ep.replace_moe_layers(specs, ep_size=ep_size, ep_rank=dist.get_rank(group=group))
        for layer in model.model.layers:
            layer.mlp.set_deepspeed_parallelism()
        preserve_qwen_routing(model)
        self.module = cast_bf16_parameters(model).to(device).requires_grad_(False).eval()
        self.eval()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def accumulation_steps(global_batch, micro_batch, world_size):
    if min(global_batch, micro_batch, world_size) < 1 or global_batch % (micro_batch * world_size):
        raise ValueError("global_batch must be divisible by micro_batch * world_size")
    return global_batch // (micro_batch * world_size)


class GlobalBatchSampler:
    """Keep each optimizer step's sample set fixed across micro-batch choices."""

    def __init__(self, num_samples, global_batch, micro_batch, rank, world_size, seed=42, start_step=0):
        self.accumulation = accumulation_steps(global_batch, micro_batch, world_size)
        if num_samples % global_batch or not 0 <= rank < world_size:
            raise ValueError("Invalid sample count or rank")
        if not 0 <= start_step <= num_samples // global_batch:
            raise ValueError("Invalid start_step")
        self.indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed)).tolist()
        self.global_batch, self.micro_batch = global_batch, micro_batch
        self.rank, self.world_size, self.start_step = rank, world_size, start_step

    def __iter__(self):
        for step in range(self.start_step, len(self.indices) // self.global_batch):
            for micro in range(self.accumulation):
                start = step * self.global_batch + (micro * self.world_size + self.rank) * self.micro_batch
                yield self.indices[start:start + self.micro_batch]

    def __len__(self):
        return (len(self.indices) // self.global_batch - self.start_step) * self.accumulation


def engine_config(config, micro_batch, world_size, teacher=False):
    accumulation = accumulation_steps(config['global_batch_size'], micro_batch, world_size)
    if world_size % config['ep_size']:
        raise ValueError("ep_size must divide world_size")
    result = {
        'train_micro_batch_size_per_gpu': micro_batch,
        'gradient_accumulation_steps': accumulation,
        'train_batch_size': config['global_batch_size'],
        'bf16': {'enabled': True},
        'data_types': {'grad_accum_dtype': 'fp32'},
        'expert_parallel': {
            'enabled': True, 'autoep_size': config['ep_size'], 'preset_model': 'qwen3_moe',
            'use_grouped_mm': True, 'comm_backend': 'comm',
            'route_norm': True, 'score_func': 'softmax', 'score_apply': 'post',
        },
        'zero_optimization': {'stage': 0 if teacher else 2},
        'steps_per_print': 1000000,
        'wall_clock_breakdown': False,
    }
    if not teacher:
        result['zero_optimization'].update({
            'overlap_comm': True, 'contiguous_gradients': True, 'reduce_scatter': True,
            'reduce_bucket_size': 50000000, 'allgather_bucket_size': 50000000,
        })
        result['gradient_clipping'] = config.get('max_grad_norm', 1.0)
        result['optimizer'] = {
            'type': 'AdamW', 'params': {
                'lr': config['learning_rate'], 'betas': config.get('betas', [0.9, 0.95]),
                'eps': config.get('adam_epsilon', 1e-8), 'weight_decay': config.get('weight_decay', 0.0),
                'torch_adam': True,
            },
        }
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def pack_hf_experts(w1, w2, w3, layout):
    """Undo AutoEP packing into the actual source Transformers tensor layout."""
    gate_up = torch.cat([w1, w3], dim=1)
    if layout == 'out_in':
        return gate_up.contiguous(), w2.contiguous()
    if layout == 'in_out':
        return gate_up.transpose(1, 2).contiguous(), w2.transpose(1, 2).contiguous()
    raise ValueError(f"Unknown expert layout: {layout}")


def expert_layout(model):
    experts = model.model.layers[0].mlp.experts
    shape = tuple(experts.gate_up_proj.shape)
    hidden = model.config.hidden_size
    intermediate = model.config.moe_intermediate_size
    if shape[1:] == (2 * intermediate, hidden):
        return 'out_in'
    if shape[1:] == (hidden, 2 * intermediate):
        return 'in_out'
    raise ValueError(f"Unsupported HF expert layout: {shape}")


def export_hf_checkpoint(engine, tokenizer, destination, layout, mla_tensor_path, metadata):
    """All ranks participate; rank 0 streams one complete HF layer at a time.

    Unlike module.state_dict(), this gathers every expert and removes AutoEP's
    wrapper names. ZeRO-2 leaves BF16 weights available without optimizer gathers.
    The completion marker is written only after every shard and index exists.
    """
    if dist.get_world_size() != engine.module.model.layers[0].mlp.ep_size:
        raise ValueError("Portable export currently requires world_size == ep_size")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    final_destination = Path(destination)
    destination = final_destination.with_name(final_destination.name + '.incomplete')
    already_complete = [False]
    if rank == 0:
        marker = final_destination / 'export_complete.json'
        if marker.exists() and json.loads(marker.read_text()) == metadata:
            already_complete[0] = True
        elif final_destination.exists():
            raise FileExistsError(f'Export destination already exists with different or incomplete metadata: {final_destination}')
    dist.broadcast_object_list(already_complete, src=0)
    if already_complete[0]:
        return
    if rank == 0:
        if destination.exists():
            # Preserve an interrupted export for inspection; do not overwrite it.
            destination.rename(destination.with_name(destination.name + f'-previous-{time.time_ns()}'))
        destination.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    weight_map, total_size = {}, 0

    def write_shard(filename, state):
        nonlocal total_size
        if rank != 0:
            return
        state = {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}
        save_file(state, str(destination / filename), metadata={'format': 'pt'})
        for name, tensor in state.items():
            weight_map[name] = filename
            total_size += tensor.numel() * tensor.element_size()

    model = engine.module
    for index, layer in enumerate(model.model.layers):
        prefix = f'model.layers.{index}.'
        state = {}
        if rank == 0:
            state = {prefix + name: value.detach().cpu() for name, value in layer.state_dict().items()
                     if not name.startswith('mlp.')}
            state[prefix + 'mlp.gate.weight'] = layer.mlp.router.gate.weight.detach().cpu()
        gathered = {}
        for name in ('w1', 'w2', 'w3'):
            local = getattr(layer.mlp.experts, name).detach().contiguous()
            pieces = [torch.empty_like(local) for _ in range(world_size)] if rank == 0 else None
            dist.gather(local, gather_list=pieces, dst=0)
            if rank == 0:
                gathered[name] = torch.cat([piece.cpu() for piece in pieces], dim=0)
            del pieces
        if rank == 0:
            gate_up, down = pack_hf_experts(gathered['w1'], gathered['w2'], gathered['w3'], layout)
            state[prefix + 'mlp.experts.gate_up_proj'] = gate_up
            state[prefix + 'mlp.experts.down_proj'] = down
        write_shard(f'model-layer-{index:05d}.safetensors', state)
        del state, gathered
    if rank == 0:
        state = {name: value for name, value in model.state_dict().items()
                 if not name.startswith('model.layers.')}
        write_shard('model-other.safetensors', state)
        atomic_json(destination / 'model.safetensors.index.json', {
            'metadata': {'total_size': total_size}, 'weight_map': weight_map,
        })
        previous = model.config.mha2mla['mla_tensor_path']
        try:
            shutil.copy2(mla_tensor_path, destination / 'mla_tensors.pt')
            model.config.mha2mla['mla_tensor_path'] = 'mla_tensors.pt'
            model.config.save_pretrained(destination)
            tokenizer.save_pretrained(destination)
        finally:
            model.config.mha2mla['mla_tensor_path'] = previous
        atomic_json(destination / 'export_complete.json', metadata)
        destination.rename(final_destination)
    dist.barrier()


def require_finite(value, label):
    if not math.isfinite(float(value)):
        raise RuntimeError(f"Non-finite {label}: {value}")


def distributed_main(function):
    try:
        function()
    except Exception:
        # A failed worker must exit before Python destroys NCCL objects: peers
        # may be in another collective and destruction can otherwise hang.
        # torchrun observes this exit and terminates the remaining workers.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
