import argparse
import csv
import math
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import load_sharded_checkpoint
from safetensors.torch import load_file
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

current_dir = os.path.dirname(os.path.abspath(__file__))
mha2mla_v2_root = os.path.dirname(current_dir)
project_root = os.path.dirname(mha2mla_v2_root)
target_directory = os.path.join(mha2mla_v2_root, "src", "mha2mla")
gen_directory = os.path.join(mha2mla_v2_root, "gen")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if target_directory not in sys.path:
    sys.path.insert(0, target_directory)
if gen_directory not in sys.path:
    sys.path.insert(0, gen_directory)

from mha2mla_datasets import MHA2MLADataArguments, load_dataset
from patching_mlafication import patch_mlafication
from patching_mla_fwd import patch_mla_fwd_instance
from helpers import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--baseline_step0", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument(
        "--seq_length",
        "--seq_lenght",
        dest="seq_length",
        type=int,
        required=True,
    )
    parser.add_argument("--num_samples", type=int, required=True)
    args = parser.parse_args()
    if args.ckpt_path is None and not args.baseline_step0:
        parser.error("Either --ckpt_path or --baseline_step0 is required.")
    if args.ckpt_path is not None and args.baseline_step0:
        parser.error("--ckpt_path and --baseline_step0 are mutually exclusive.")
    return args


def init_single_process_group():
    if dist.is_available() and not dist.is_initialized():
        if torch.cuda.is_available():
            backend = "nccl"
            torch.cuda.set_device(0)
        else:
            backend = "gloo"

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])

        dist.init_process_group(
            backend=backend,
            rank=0,
            world_size=1,
        )


args = parse_args()
data_seed = 43
init_single_process_group()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    fix_mistral_regex=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if args.baseline_step0:
    print("load baseline step0 model")

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    mla_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
elif args.ckpt_path.endswith(".pt"):
    print("load step0 model")

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    mha_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    mha2mla_args = SimpleNamespace(
        is_baseline=False,
        mla_tensor_path=args.ckpt_path,
        model_name_or_path=args.model_name_or_path,
        output_mla_dir=None,
    )
    mla_model = patch_mlafication(mha_model, model_config, mha2mla_args)
    patch_mla_fwd_instance(mla_model)
else:
    model_config = AutoConfig.from_pretrained(args.ckpt_path)
    if not hasattr(model_config, "mha2mla"):
        raise ValueError("not supported.")
    if model_config.mha2mla["is_baseline"]:
        print("load mha model from ckpt")

        mla_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        mha2mla_args = SimpleNamespace(**model_config.mha2mla)
        if getattr(mha2mla_args, "model_name_or_path", None) is None:
            mha2mla_args.model_name_or_path = args.model_name_or_path
        signle_weight_file = os.path.join(args.ckpt_path, "model.safetensors")
        if os.path.exists(signle_weight_file):
            state_dict = load_file(signle_weight_file)
            mla_model.load_state_dict(state_dict, strict=False)
        else:
            load_sharded_checkpoint(mla_model, args.ckpt_path, strict=False)
    else:
        print("load mla model from ckpt")

        mha_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        mha2mla_args = SimpleNamespace(**model_config.mha2mla)
        if getattr(mha2mla_args, "model_name_or_path", None) is None:
            mha2mla_args.model_name_or_path = args.model_name_or_path

        if not os.path.isabs(mha2mla_args.mla_tensor_path):
            mha2mla_args.mla_tensor_path = os.path.join(
                args.ckpt_path, mha2mla_args.mla_tensor_path
            )
        mla_model = patch_mlafication(mha_model, model_config, mha2mla_args)
        patch_mla_fwd_instance(mla_model)
        load_checkpoint(mla_model, args.ckpt_path, strict=False)
model = mla_model.to(dtype=dtype, device=device).eval()

data_args = MHA2MLADataArguments()
data_args.sequence_length = args.seq_length
data_args.random_seed = data_seed
num_batches = max(1, math.ceil(args.num_samples / data_args.per_gpu_batchsize))
data_args.max_batches = num_batches

ds = load_dataset(tokenizer, data_args)
dataloader = DataLoader(ds, batch_size=data_args.per_gpu_batchsize, shuffle=False)

def build_query_key_states_baseline(module, kwargs):
    hidden_states = kwargs["hidden_states"]
    position_embeddings = kwargs["position_embeddings"]   # (cos, sin)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    query_states = module.q_proj(hidden_states).view(hidden_shape)
    key_states = module.k_proj(hidden_states).view(hidden_shape)

    query_states = module.q_norm(query_states).transpose(1, 2)
    key_states = module.k_norm(key_states).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    return query_states, key_states

def is_baseline_attention():
    if args.ckpt_path is not None and args.ckpt_path.endswith(".pt"):
        return False
    return not hasattr(model_config, "mha2mla") or model_config.mha2mla["is_baseline"]


def build_query_key_states(module, kwargs):
    if is_baseline_attention():
        return build_query_key_states_baseline(module, kwargs)
    hidden_states = kwargs["hidden_states"]
    position_embeddings = kwargs["position_embeddings"]
    input_shape = hidden_states.shape[:-1]
    num_kv_heads = module.config.num_key_value_heads
    num_q_heads = module.config.num_attention_heads

    q_shape = (*input_shape, num_q_heads, -1)
    k_r_shape = (*input_shape, num_kv_heads, -1)
    k_c_shape = (*input_shape, num_kv_heads, -1)

    k_r = module.k_r_proj(hidden_states).view(k_r_shape)
    if hasattr(module, "k_r_norm"):
        k_r = module.k_r_norm(k_r)
    k_r = k_r.transpose(1, 2)

    q = module.q_norm(module.q_proj(hidden_states).view(q_shape))
    q_r = q[:, :, :, : k_r.size(-1)].transpose(1, 2)
    q_c = q[:, :, :, k_r.size(-1) :].transpose(1, 2)

    k_c, _ = module.kv_proj(hidden_states)
    k_c = k_c.view(k_c_shape).transpose(1, 2)

    if hasattr(module, "k_norm"):
        k = torch.cat([k_r, k_c], dim=-1)
        k = module.k_norm(k.transpose(1, 2)).transpose(1, 2)
        k_r, k_c = k[..., : k_r.size(-1)], k[..., k_r.size(-1) :]

    cos, sin = position_embeddings
    q_r, k_r = module.mla_special_rope_fn(q_r, k_r, cos, sin)

    query_states = torch.cat([q_r, q_c], dim=-1)
    key_states = torch.cat([k_r, k_c], dim=-1)
    return query_states, key_states


def get_attn_logits_max(query_states, key_states, scaling, chunk_size=256):
    if key_states.size(1) != query_states.size(1):
        key_states = key_states.repeat_interleave(
            query_states.size(1) // key_states.size(1), dim=1
        )
    middle_head_idx = query_states.size(1) // 2
    # query_states = query_states[:, middle_head_idx : middle_head_idx + 1]
    # key_states = key_states[:, middle_head_idx : middle_head_idx + 1]
    key_states_t = key_states.to(torch.float32).transpose(-1, -2).contiguous()
    mx = torch.tensor(0.0, device=query_states.device, dtype=torch.float32)

    for start in range(0, query_states.size(-2), chunk_size):
        q = query_states[:, :, start : start + chunk_size, :].to(torch.float32)
        logits = torch.matmul(q, key_states_t) * scaling

        q_pos = torch.arange(start, start + q.size(-2), device=logits.device).unsqueeze(-1)   # [q_chunk,1]
        k_pos = torch.arange(key_states.size(-2), device=logits.device).unsqueeze(0)   # [1,k_len_total]
        causal_mask = k_pos <= q_pos                                            # [q_chunk,k_len_total]
        logits = logits.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -100.0)

        mx = torch.maximum(mx, logits.amax())

    return mx.detach()


def hook_collect_attn_logits_max(module, args, kwargs, output):
    query_states, key_states = build_query_key_states(module, kwargs)
    layer_max = get_attn_logits_max(
        query_states,
        key_states,
        scaling=module.scaling,
    )
    if ATTN_LOGITS_MAX[module.layer_idx] is None:
        ATTN_LOGITS_MAX[module.layer_idx] = layer_max
    else:
        ATTN_LOGITS_MAX[module.layer_idx] = torch.maximum(
            ATTN_LOGITS_MAX[module.layer_idx],
            layer_max,
        )


def register_hook(hook_func):
    handles = [
        layer.self_attn.register_forward_hook(hook_func, with_kwargs=True)
        for layer in model.model.layers
    ]
    return handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()
    handles.clear()


def model_forward():
    pbar = tqdm(total=num_batches, desc="fwd")
    for idx, batch in enumerate(dataloader):
        if idx >= num_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        pbar.update(1)
    pbar.close()


n_layers = model.config.num_hidden_layers
ATTN_LOGITS_MAX = [None for _ in range(n_layers)]
handles = register_hook(hook_collect_attn_logits_max)
model_forward()
remove_hooks(handles)

for layer_idx in range(n_layers):
    if ATTN_LOGITS_MAX[layer_idx] is None:
        ATTN_LOGITS_MAX[layer_idx] = torch.tensor(
            float("-inf"), device=device, dtype=torch.float32
        )

real_num_samples = num_batches * data_args.per_gpu_batchsize
print(f"data_seed: {data_seed}")
print(f"model_name_or_path: {args.model_name_or_path}")
print(f"ckpt_path: {args.ckpt_path}")
print(f"seq_length: {args.seq_length}")
print(f"num_samples: {args.num_samples}")
print(f"real_num_samples: {real_num_samples}")
print(f"middle_head_idx: {model.config.num_attention_heads // 2}")
print("ATTN_LOGITS_MAX:")
for layer_idx in range(n_layers):
    print(f"layer_{layer_idx}: {ATTN_LOGITS_MAX[layer_idx].item():.6f}")
global_max = max(x.item() for x in ATTN_LOGITS_MAX)
print(f"global_max: {global_max:.6f}")

if args.output_file is not None:
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    layer_values = [x.item() for x in ATTN_LOGITS_MAX]
    ckpt_label = args.ckpt_path if args.ckpt_path is not None else args.model_name_or_path
    with output_file.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["step", "ckpt_path", "global_max"]
            + [f"layer_{idx}" for idx in range(n_layers)]
        )
        writer.writerow(
            [args.step, ckpt_label, f"{global_max:.6f}"]
            + [f"{value:.6f}" for value in layer_values]
        )

    layer_major_file = output_file.with_name(output_file.stem + "_by_layer.tsv")
    with layer_major_file.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["layer", f"step_{args.step}"])
        for layer_idx, value in enumerate(layer_values):
            writer.writerow([f"layer_{layer_idx}", f"{value:.6f}"])

    print(f"[saved] wide_tsv={output_file}")
    print(f"[saved] layer_major_tsv={layer_major_file}")

if dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()
