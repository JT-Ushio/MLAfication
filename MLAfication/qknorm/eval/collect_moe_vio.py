#!/usr/bin/env python
import argparse
import csv
import gc
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


CURRENT_FILE = Path(__file__).resolve()
EVAL_DIR = CURRENT_FILE.parent
REPO_ROOT = EVAL_DIR.parent
SRC_MHA2MLA_DIR = REPO_ROOT / "src" / "mha2mla"
GEN_DIR = REPO_ROOT / "gen"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_results"

for extra_path in (str(SRC_MHA2MLA_DIR), str(GEN_DIR), str(REPO_ROOT)):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from helpers import cast_model_parameters, load_checkpoint


@dataclass
class ModelSpec:
    label: str
    path: Optional[Path]


class MoeStatsCollector:
    def __init__(
        self,
        model,
        label: str,
        baseline_routes: Optional[dict[int, list[torch.Tensor]]] = None,
        keep_routes: bool = False,
    ):
        self.label = label
        self.baseline_routes = baseline_routes
        self.keep_routes = keep_routes
        self.current_batch_idx = 0
        self.handles = []
        self.layers = []
        self.layer_name_to_idx = {}
        self.route_cache = {}
        self.layer_stats = {}
        self.route_stats = {}

        for layer_idx, layer in enumerate(model.model.layers):
            mlp = getattr(layer, "mlp", None)
            if not is_moe_block(mlp):
                continue
            # Transformers 5 returns selected indices from the router child;
            # older versions return router logits from the whole MoE block.
            target = mlp.gate if hasattr(mlp.gate, 'top_k') else mlp
            self.layer_name_to_idx[id(target)] = layer_idx
            self.layers.append((layer_idx, target))
            self.route_cache[layer_idx] = []
            num_experts = int(target.num_experts)
            self.layer_stats[layer_idx] = {
                "num_experts": num_experts,
                "top_k": int(target.top_k),
                "tokens": 0,
                "counts": torch.zeros(num_experts, dtype=torch.long),
            }
            self.route_stats[layer_idx] = {
                "sum_vio": 0.0,
                "max_vio": 0.0,
                "tokens": 0,
                "exact_match": 0,
            }

        if not self.layers:
            raise ValueError("No MoE layers were found under model.model.layers[*].mlp.")

    def register(self):
        self.handles = [
            mlp.register_forward_hook(self.hook)
            for _, mlp in self.layers
        ]
        return self.handles

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def hook(self, module, inputs, output):
        layer_idx = self.layer_name_to_idx[id(module)]
        top_k = int(module.top_k)
        num_experts = int(module.num_experts)
        if isinstance(output, tuple) and len(output) == 3:
            selected = output[2].detach()
        else:
            router_logits = extract_router_logits(output)
            if router_logits is None:
                raise RuntimeError('MoE hook did not expose router logits or selected expert indices')
            selected = router_logits.detach().float().topk(top_k, dim=-1).indices
        selected = selected.to(torch.long)
        flat_selected = selected.reshape(-1)

        stats = self.layer_stats[layer_idx]
        counts = torch.bincount(
            flat_selected.cpu(),
            minlength=num_experts,
        ).to(torch.long)
        stats["counts"] += counts
        stats["tokens"] += int(selected.shape[0])

        if self.keep_routes:
            self.route_cache[layer_idx].append(selected.cpu().to(torch.int16))

        if self.baseline_routes is not None:
            baseline_selected = self.baseline_routes[layer_idx][self.current_batch_idx]
            baseline_selected = baseline_selected.to(
                device=selected.device,
                dtype=torch.long,
            )
            if baseline_selected.shape != selected.shape:
                raise ValueError(
                    f"Route shape mismatch on {self.label} layer {layer_idx} "
                    f"batch {self.current_batch_idx}: baseline "
                    f"{tuple(baseline_selected.shape)} vs current {tuple(selected.shape)}"
                )
            route_vio, exact_match = compute_route_vio(
                selected,
                baseline_selected,
                num_experts=num_experts,
            )
            route_stats = self.route_stats[layer_idx]
            route_stats["sum_vio"] += float(route_vio.sum().item())
            route_stats["max_vio"] = max(
                route_stats["max_vio"],
                float(route_vio.max().item()),
            )
            route_stats["tokens"] += int(route_vio.numel())
            route_stats["exact_match"] += int(exact_match.sum().item())

    def finish(self):
        per_layer = []
        total_avg_vio = 0.0
        total_max_vio = 0.0
        total_avg_abs_vio = 0.0
        total_max_abs_vio = 0.0
        total_layer_experts = 0

        total_route_sum = 0.0
        total_route_tokens = 0
        total_route_exact = 0
        total_route_max = 0.0

        for layer_idx, _ in self.layers:
            stats = self.layer_stats[layer_idx]
            counts = stats["counts"].to(torch.float32)
            tokens = int(stats["tokens"])
            num_experts = int(stats["num_experts"])
            top_k = int(stats["top_k"])
            expected = tokens * top_k / num_experts
            if expected <= 0:
                raise ValueError(f"No tokens collected for layer {layer_idx}.")

            rel = counts / expected - 1.0
            pos = rel.clamp_min(0.0)
            abs_rel = rel.abs()
            max_count = int(counts.max().item())
            max_expert = int(counts.argmax().item())

            route = self.route_stats[layer_idx]
            route_tokens = int(route["tokens"])
            if route_tokens > 0:
                route_avg = route["sum_vio"] / route_tokens
                route_max = float(route["max_vio"])
                route_exact = route["exact_match"] / route_tokens
            else:
                route_avg = None
                route_max = None
                route_exact = None

            row = {
                "label": self.label,
                "layer": layer_idx,
                "tokens": tokens,
                "num_experts": num_experts,
                "top_k": top_k,
                "expected_per_expert": expected,
                "average_vio": float(pos.mean().item()),
                "max_vio": float(pos.max().item()),
                "avg_abs_vio": float(abs_rel.mean().item()),
                "max_abs_vio": float(abs_rel.max().item()),
                "min_rel_vio": float(rel.min().item()),
                "max_rel_vio": float(rel.max().item()),
                "most_loaded_expert": max_expert,
                "most_loaded_count": max_count,
                "route_average_vio": route_avg,
                "route_max_vio": route_max,
                "route_exact_match": route_exact,
            }
            per_layer.append(row)

            total_avg_vio += row["average_vio"] * num_experts
            total_avg_abs_vio += row["avg_abs_vio"] * num_experts
            total_max_vio = max(total_max_vio, row["max_vio"])
            total_max_abs_vio = max(total_max_abs_vio, row["max_abs_vio"])
            total_layer_experts += num_experts

            if route_tokens > 0:
                total_route_sum += route["sum_vio"]
                total_route_tokens += route_tokens
                total_route_exact += route["exact_match"]
                total_route_max = max(total_route_max, float(route["max_vio"]))

        summary = {
            "label": self.label,
            "num_moe_layers": len(per_layer),
            "num_layer_experts": total_layer_experts,
            "tokens_per_layer": per_layer[0]["tokens"],
            "num_experts": per_layer[0]["num_experts"],
            "top_k": per_layer[0]["top_k"],
            "average_vio": total_avg_vio / total_layer_experts,
            "max_vio": total_max_vio,
            "avg_abs_vio": total_avg_abs_vio / total_layer_experts,
            "max_abs_vio": total_max_abs_vio,
            "route_average_vio": (
                total_route_sum / total_route_tokens if total_route_tokens > 0 else None
            ),
            "route_max_vio": (
                total_route_max if total_route_tokens > 0 else None
            ),
            "route_exact_match": (
                total_route_exact / total_route_tokens
                if total_route_tokens > 0
                else None
            ),
        }
        return summary, per_layer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect Qwen3-MoE expert load violation metrics for the HF baseline "
            "and MLA checkpoints. average_vio/max_vio are positive relative "
            "over-loads: relu(expert_load / uniform_expected_load - 1)."
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HF baseline model path used as the base model for MLA patching.",
    )
    parser.add_argument(
        "--baseline_label",
        type=str,
        default="hf_baseline",
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help=(
            "Checkpoint to evaluate, either CHECKPOINT or LABEL=CHECKPOINT. "
            "Repeat this option to evaluate multiple checkpoints."
        ),
    )
    parser.add_argument("--seq_length", "--seq_lenght", dest="seq_length", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--sample_offset",
        type=int,
        default=0,
        help=(
            "Skip this many deterministic Nanoset samples before evaluation. "
            "Set this past the training stream to avoid exact train/eval overlap."
        ),
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=1234,
        help="Nanoset random_seed. Training configs use trainset_seed=42 by default.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("MHA2MLA_DATA_PATH"),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "qwen3_30b_a3b_moe_vio.tsv"),
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model execution. Use cuda for a single large GPU.",
    )
    parser.add_argument(
        "--no_route_compare",
        action="store_true",
        help="Only report expert load violation; skip top-k route comparison to baseline.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoModel/AutoTokenizer.",
    )
    args = parser.parse_args()
    if not args.data_path:
        parser.error("--data_path or MHA2MLA_DATA_PATH is required")
    return args


def parse_ckpt_specs(raw_ckpts):
    specs = []
    for raw in raw_ckpts:
        if "=" in raw:
            label, path = raw.split("=", 1)
            label = sanitize_label(label)
        else:
            path = raw
            label = sanitize_label(Path(path).parent.name + "_" + Path(path).name)
        specs.append(ModelSpec(label, Path(path)))
    return specs


def sanitize_label(value):
    value = value.strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def is_moe_block(module):
    return (
        module is not None
        and hasattr(module, "gate")
        and ((hasattr(module, "num_experts") and hasattr(module, "top_k"))
             or (hasattr(module.gate, "num_experts") and hasattr(module.gate, "top_k")))
    )


def extract_router_logits(output):
    if isinstance(output, tuple) and len(output) >= 2:
        return output[1]
    return None


def compute_route_vio(selected, baseline_selected, num_experts):
    current_mask = F.one_hot(selected, num_classes=num_experts).sum(dim=1).bool()
    baseline_mask = F.one_hot(baseline_selected, num_classes=num_experts).sum(dim=1).bool()
    overlap = (current_mask & baseline_mask).sum(dim=1).to(torch.float32)
    top_k = selected.shape[1]
    route_vio = (top_k - overlap) / top_k
    exact_match = overlap.eq(top_k)
    return route_vio, exact_match


def torch_dtype_from_arg(dtype_name):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_patch_functions(ckpt_path, mha2mla_args):
    from patching_mla_fwd import patch_mla_fwd_instance
    from patching_mlafication import patch_mlafication
    return patch_mlafication, patch_mla_fwd_instance


def load_model(model_name_or_path, ckpt_path, dtype, device, trust_remote_code):
    if ckpt_path is None:
        print(f"[load] baseline={model_name_or_path}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
        return move_model(model, device=device, dtype=dtype)

    print(f"[load] ckpt={ckpt_path}", flush=True)
    ckpt_config = AutoConfig.from_pretrained(
        ckpt_path,
        trust_remote_code=trust_remote_code,
    )
    mha2mla_cfg = getattr(ckpt_config, "mha2mla", None)
    if mha2mla_cfg is None:
        raise ValueError(f"Checkpoint config has no `mha2mla`: {ckpt_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    mha2mla_args = SimpleNamespace(**mha2mla_cfg)
    if getattr(mha2mla_args, "model_name_or_path", None) is None:
        mha2mla_args.model_name_or_path = model_name_or_path
    tensor_path = getattr(mha2mla_args, 'mla_tensor_path', None)
    if tensor_path and not os.path.isabs(tensor_path):
        mha2mla_args.mla_tensor_path = str(ckpt_path / tensor_path)

    if getattr(mha2mla_args, "is_baseline", False):
        load_checkpoint(model, str(ckpt_path), strict=False)
    else:
        patch_mlafication, patch_mla_fwd_instance = resolve_patch_functions(
            ckpt_path,
            mha2mla_args,
        )
        model = patch_mlafication(model, ckpt_config, mha2mla_args)
        patch_mla_fwd_instance(model)
        load_checkpoint(model, str(ckpt_path), strict=False)

    return move_model(model, device=device, dtype=dtype)


def move_model(model, device, dtype):
    if dtype == "auto":
        return model.to(device).eval()
    return cast_model_parameters(model, dtype).to(device).eval()


def build_dataloader(tokenizer, args):
    from nanotron.data.nanoset import Nanoset
    from mha2mla_datasets import MHA2MLADataArguments

    os.environ["MHA2MLA_DATA_PATH"] = args.data_path.rstrip("/")
    data_args = MHA2MLADataArguments()
    data_args.sequence_length = args.seq_length
    data_args.per_gpu_batchsize = args.batch_size
    data_args.max_batches = max(1, math.ceil(args.num_samples / args.batch_size))
    real_num_samples = data_args.max_batches * args.batch_size
    total_num_samples = args.sample_offset + real_num_samples
    if args.sample_offset < 0:
        raise ValueError("--sample_offset must be non-negative.")

    origin_model_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = data_args.sequence_length
    token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
    tokenizer.model_max_length = origin_model_max_length

    # The pinned Nanoset logger expects a default process group even for a
    # standalone evaluator. Its indexing/data access need no collectives.
    own_group = not torch.distributed.is_initialized()
    if own_group:
        torch.distributed.init_process_group(
            'gloo', store=torch.distributed.HashStore(), rank=0, world_size=1
        )
    try:
        base_dataset = Nanoset(
            dataset_folders=data_args.dataset_folders,
            sequence_length=data_args.sequence_length,
            dataset_weights=data_args.dataset_weights,
            token_size=token_size,
            random_seed=args.data_seed,
            train_split_num_samples=total_num_samples,
        )
    finally:
        if own_group:
            torch.distributed.destroy_process_group()
    if args.sample_offset > 0:
        dataset = Subset(
            base_dataset,
            range(args.sample_offset, args.sample_offset + real_num_samples),
        )
    else:
        dataset = base_dataset
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False), data_args.max_batches


def run_forward(model, tokenizer, dataloader, num_batches, device, collector):
    collector.register()
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc=collector.label)):
            if batch_idx >= num_batches:
                break
            collector.current_batch_idx = batch_idx
            input_ids = batch["input_ids"].to(device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
    finally:
        collector.remove()


def format_float(value):
    if value is None:
        return ""
    return f"{value:.8f}"


def write_summary_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "path",
        "num_moe_layers",
        "tokens_per_layer",
        "num_experts",
        "top_k",
        "average_vio",
        "max_vio",
        "avg_abs_vio",
        "max_abs_vio",
        "route_average_vio",
        "route_max_vio",
        "route_exact_match",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            out = row.copy()
            for key in fieldnames:
                if isinstance(out.get(key), float):
                    out[key] = format_float(out[key])
                elif out.get(key) is None:
                    out[key] = ""
            writer.writerow({key: out.get(key, "") for key in fieldnames})


def write_layer_tsv(path, rows):
    layer_path = path.with_name(path.stem + "_by_layer.tsv")
    fieldnames = [
        "label",
        "layer",
        "tokens",
        "num_experts",
        "top_k",
        "expected_per_expert",
        "average_vio",
        "max_vio",
        "avg_abs_vio",
        "max_abs_vio",
        "min_rel_vio",
        "max_rel_vio",
        "most_loaded_expert",
        "most_loaded_count",
        "route_average_vio",
        "route_max_vio",
        "route_exact_match",
    ]
    with layer_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            out = row.copy()
            for key in fieldnames:
                if isinstance(out.get(key), float):
                    out[key] = format_float(out[key])
                elif out.get(key) is None:
                    out[key] = ""
            writer.writerow({key: out.get(key, "") for key in fieldnames})
    return layer_path


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Pass --device cpu for a CPU smoke test.")

    model_name_or_path = str(Path(args.model_name_or_path))
    dtype = torch_dtype_from_arg(args.dtype)
    output_file = Path(args.output_file)
    ckpt_specs = parse_ckpt_specs(args.ckpt)
    model_specs = [ModelSpec(args.baseline_label, None)] + ckpt_specs

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataloader, num_batches = build_dataloader(tokenizer, args)

    print(f"[info] data_path={args.data_path}", flush=True)
    print(f"[info] seq_length={args.seq_length}", flush=True)
    print(f"[info] num_samples={args.num_samples}", flush=True)
    print(f"[info] real_num_samples={num_batches * args.batch_size}", flush=True)
    print(f"[info] sample_offset={args.sample_offset}", flush=True)
    print(f"[info] data_seed={args.data_seed}", flush=True)
    print(f"[info] output_file={output_file}", flush=True)
    print(
        "[info] average_vio/max_vio = "
        "relu(expert_load / (tokens * top_k / num_experts) - 1)",
        flush=True,
    )
    if not args.no_route_compare:
        print(
            "[info] route_* fields compare each ckpt top-k selected experts "
            "against the HF baseline on the same samples.",
            flush=True,
        )

    summary_rows = []
    layer_rows = []
    baseline_routes = None

    for spec in model_specs:
        if spec.path is not None and not spec.path.is_dir():
            raise FileNotFoundError(f"Checkpoint does not exist: {spec.path}")

        model = load_model(
            model_name_or_path=model_name_or_path,
            ckpt_path=spec.path,
            dtype=dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
        keep_routes = (spec.path is None) and not args.no_route_compare
        compare_routes = baseline_routes if spec.path is not None else None
        collector = MoeStatsCollector(
            model,
            label=spec.label,
            baseline_routes=compare_routes,
            keep_routes=keep_routes,
        )
        run_forward(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            num_batches=num_batches,
            device=args.device,
            collector=collector,
        )
        summary, per_layer = collector.finish()
        summary["path"] = str(spec.path) if spec.path is not None else model_name_or_path
        summary_rows.append(summary)
        layer_rows.extend(per_layer)

        print(
            f"[done] {spec.label} "
            f"average_vio={summary['average_vio']:.8f} "
            f"max_vio={summary['max_vio']:.8f} "
            f"route_average_vio={format_float(summary['route_average_vio'])}",
            flush=True,
        )

        if keep_routes:
            baseline_routes = collector.route_cache

        unload_model(model)

    write_summary_tsv(output_file, summary_rows)
    layer_file = write_layer_tsv(output_file, layer_rows)
    print(f"[saved] summary_tsv={output_file}", flush=True)
    print(f"[saved] layer_tsv={layer_file}", flush=True)


if __name__ == "__main__":
    main()
