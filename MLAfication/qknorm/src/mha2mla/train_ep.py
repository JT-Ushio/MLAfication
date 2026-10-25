"""Full-parameter Qwen3-MoE distillation with AutoEP and ZeRO-2.

Launch with torchrun; use run_ep_distill.py for OOM-aware micro-batch selection.
Teacher and student are independently expert-sharded before moving to CUDA.
"""
import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from ep_distill_utils import (
    FrozenEPTeacher, GlobalBatchSampler, accumulation_steps, atomic_json, cast_bf16_parameters, distillation_kl,
    engine_config, expert_layout, export_hf_checkpoint, preserve_qwen_routing, require_finite, distributed_main,
)
from patching_mlafication import patch_mlafication
from patching_mla_fwd import patch_mla_fwd_instance


def lr_multiplier(step, config):
    warmup = config['warmup_steps']
    decay_start = config['decay_start_step']
    if step <= warmup:
        return step / max(1, warmup)
    if step < decay_start:
        return 1.0
    return max(0.0, 1.0 - ((step - decay_start) / (config['max_steps'] - decay_start)) ** 0.5)


def collate_tokens(samples):
    # Datatrove returns sequence_length + 1 tokens, as in the original Trainer.
    ids = torch.stack([torch.as_tensor(x['input_ids'], dtype=torch.long) for x in samples])
    return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


def make_dataset(config, tokenizer):
    from nanotron.data.nanoset import Nanoset
    return Nanoset(
        dataset_folders=config['dataset_folders'], dataset_weights=config['dataset_weights'],
        sequence_length=config['sequence_length'], token_size=4 if len(tokenizer) > 65536 else 2,
        random_seed=config['seed'],
        train_split_num_samples=config['max_steps'] * config['global_batch_size'],
    )


def validate_engines(teacher, student, config):
    for name, engine in [('teacher', teacher), ('student', student)]:
        for layer in engine.module.model.layers:
            if layer.mlp.ep_size != config['ep_size'] or not layer.mlp.experts.use_grouped_mm:
                raise RuntimeError(f'{name}: EP/grouped GEMM was not activated')
        trainable = [p.requires_grad for p in engine.module.parameters()]
        if (name == 'student' and not all(trainable)) or (name == 'teacher' and any(trainable)):
            raise RuntimeError(f'{name}: incorrect parameter freezing')
    if not student.module.is_gradient_checkpointing:
        raise RuntimeError('Student activation checkpointing is not active')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--micro-batch', type=int, required=True)
    parser.add_argument('--probe-steps', type=int, default=0)
    parser.add_argument('--resume-tag')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend='nccl', timeout=timedelta(minutes=30))
    if world != config['ep_size']:
        raise ValueError('This runner requires world_size == ep_size')
    device = torch.device('cuda', local_rank)
    if config.get('required_gpu') and config['required_gpu'] not in torch.cuda.get_device_name():
        raise RuntimeError(f"Expected {config['required_gpu']}, got {torch.cuda.get_device_name()}")
    set_seed(config['seed'])
    accumulation = accumulation_steps(config['global_batch_size'], args.micro_batch, world)
    run_dir, ckpt_dir = Path(config['run_dir']), Path(config['checkpoint_dir'])
    probe = bool(args.probe_steps)
    prefix = f'probe-mb{args.micro_batch}' if probe else 'train'
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(run_dir / f'{prefix}-status.json', {'phase': 'loading', 'time': time.time()})
        atomic_json(run_dir / 'config.json', config)
    dist.barrier()
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load_base():
        model = AutoModelForCausalLM.from_pretrained(
            config['model_name_or_path'], dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True, low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        model.config.output_router_logits = False
        return model

    teacher_model = load_base().requires_grad_(False).eval()
    teacher = FrozenEPTeacher(teacher_model, config, device)
    teacher.eval()
    del teacher_model
    gc.collect()
    student_model = load_base()
    layout = expert_layout(student_model)
    patch_mlafication(student_model, student_model.config, SimpleNamespace(**config))
    patch_mla_fwd_instance(student_model)
    cast_bf16_parameters(student_model).requires_grad_(True)
    student_model.config.mha2mla = {
        'model_name_or_path': config['model_name_or_path'], 'mla_tensor_path': config['mla_tensor_path'],
        'is_baseline': False, 'checkpoint_type': 'full',
    }
    student_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    student, _, _, _ = deepspeed.initialize(
        model=student_model, config=engine_config(config, args.micro_batch, world),
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_multiplier(step, config)),
    )
    del student_model
    preserve_qwen_routing(student.module)
    student.train()
    validate_engines(teacher, student, config)
    start_step = 0
    if args.resume_tag:
        loaded, state = student.load_checkpoint(str(ckpt_dir / 'native'), tag=args.resume_tag)
        if not loaded or state.get('config_sha256') != digest:
            raise RuntimeError('Resume failed or configuration differs from checkpoint')
        if state['micro_batch'] != args.micro_batch:
            raise RuntimeError('Resume must retain the recorded micro-batch')
        start_step = state['step']
    elif not probe and (ckpt_dir / 'native' / 'latest').exists():
        raise RuntimeError('Existing training checkpoint: specify --resume-tag; refusing to overwrite')

    # Build once per rank. The pinned Nanoset implementation matches the PEFT data mixture.
    dataset = make_dataset(config, tokenizer)
    loader = DataLoader(dataset, batch_sampler=GlobalBatchSampler(
        len(dataset), config['global_batch_size'], args.micro_batch, rank, world,
        config['seed'], start_step), collate_fn=collate_tokens,
        num_workers=config.get('dataloader_workers', 0), pin_memory=True)
    dist.barrier()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step, micro, losses, tokens = start_step, 0, torch.zeros((), device=device), 0
    started = time.monotonic()
    end_step = min(start_step + args.probe_steps, config['max_steps']) if probe else config['max_steps']
    # A final native checkpoint can be resumed solely to finish its HF export.
    previous_metrics = run_dir / 'train-metrics.jsonl'
    metric = {'step': start_step, 'time': time.time()}
    if previous_metrics.exists():
        metric.update(json.loads(previous_metrics.read_text().splitlines()[-1]))
    for inputs in loader:
        inputs = {name: value.to(device, non_blocking=True) for name, value in inputs.items()}
        with torch.no_grad():
            teacher_logits = teacher(**inputs, use_cache=False).logits
        student_logits = student(**inputs, use_cache=False).logits
        loss = distillation_kl(student_logits, teacher_logits, config.get('temperature', 1.0),
                               config.get('kl_block_rows', 128))
        # Check all ranks together to avoid stranding peers in EP collectives.
        bad = (~torch.isfinite(loss)).to(torch.int32)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bad.item():
            raise RuntimeError('Non-finite distillation loss on at least one rank')
        losses += loss.detach()
        tokens += inputs['input_ids'].numel()
        student.backward(loss)
        student.step()
        del loss, teacher_logits, student_logits, inputs
        micro += 1
        if micro % accumulation:
            continue
        step += 1
        torch.cuda.synchronize()
        duration = time.monotonic() - started
        grad_norm = float(student.get_global_grad_norm())
        require_finite(grad_norm, 'gradient norm')
        dist.all_reduce(losses)
        memory = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                              device=device, dtype=torch.float64)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        elapsed = torch.tensor(duration, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        metric = {
            'phase': 'probing' if probe else 'training', 'step': step, 'time': time.time(),
            'loss': losses.item() / (world * accumulation), 'grad_norm': grad_norm,
            'learning_rate': student.get_lr()[0], 'micro_batch': args.micro_batch,
            'gradient_accumulation_steps': accumulation, 'world_size': world,
            'tokens_per_step': tokens * world, 'tokens_per_second': tokens * world / elapsed.item(),
            'step_seconds': elapsed.item(), 'peak_allocated_gib': memory[0].item() / 2**30,
            'peak_reserved_gib': memory[1].item() / 2**30,
            'device_total_gib': torch.cuda.get_device_properties(device).total_memory / 2**30,
        }
        if rank == 0:
            print(json.dumps(metric, allow_nan=False), flush=True)
            with (run_dir / f'{prefix}-metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metric, allow_nan=False) + '\n')
            atomic_json(run_dir / f'{prefix}-status.json', metric)
        if not probe and (step % config['save_steps'] == 0 or step == end_step):
            tag = f'step-{step:06d}'
            state = {'step': step, 'config_sha256': digest, 'micro_batch': args.micro_batch}
            student.save_checkpoint(str(ckpt_dir / 'native'), tag=tag, client_state=state)
            dist.barrier()
            if rank == 0:
                atomic_json(ckpt_dir / 'native' / tag / 'complete.json', state)
                # Only prune this run's own completed native checkpoints, keeping two by default.
                completed = sorted(p.parent for p in (ckpt_dir / 'native').glob('step-*/complete.json'))
                for old in completed[:-config.get('keep_native_checkpoints', 2)]:
                    shutil.rmtree(old)
            dist.barrier()
        losses.zero_()
        tokens = 0
        started = time.monotonic()
        if step == end_step:
            break
    if step != end_step:
        raise RuntimeError(f'Data exhausted at step {step}, expected {end_step}')
    if probe:
        metric.update(phase='probe_complete', config_sha256=digest)
        if rank == 0:
            atomic_json(run_dir / f'{prefix}-result.json', metric)
    else:
        export_hf_checkpoint(student, tokenizer, ckpt_dir / f'hf-step-{step:06d}', layout,
                             config['mla_tensor_path'], {'step': step, 'config_sha256': digest})
        if rank == 0:
            atomic_json(run_dir / 'train-status.json', dict(metric, phase='complete'))
    dist.destroy_process_group()


if __name__ == '__main__':
    distributed_main(main)
