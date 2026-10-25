# MLAfication for QKNorm Models

This directory contains the MLAfication implementation for QKNorm-based GQA
and MoE models, including Qwen3 and Qwen3-MoE. The implementation for standard
MHA/GQA models is provided in the sibling `mha_gqa/` directory.

[← MLAfication guide](../README.md)

Run the commands below from this implementation directory (`MLAfication/qknorm/` in the unified repository).

## Repository layout

- `gen/convert.py`: DeltaRoPE selection, QKNorm absorption, and SVDLLM V2
  initialization.
- `gen/patch_func.py` and `gen/svdllm.py`: conversion primitives.
- `src/mha2mla/train.py`: logit-distillation recovery training.
- `src/mha2mla/patching_*.py`: QKNorm-compatible MLA and latent-cache forward
  passes.
- `eval/`: short-context, LongBench, QKNorm, and MoE evaluation entry points.

## Workflow

1. Follow the parent-project [environment installation](../README.md#environment),
   including its security overrides, after selecting the CUDA-enabled PyTorch
   build for the host.
2. Set `MHA2MLA_DATA_PATH` and generate MLA initialization tensors. The same
   entry point detects dense Qwen3 and Qwen3-MoE from the model config:

   ```bash
   torchrun --nproc_per_node 4 gen/convert.py \
     --model_name_or_path Qwen/Qwen3-8B-Base \
     --output_path artifacts/qwen3-8b-rank512.pt \
     --lora_rank 512 --topk 8 --num_samples 128 --sequence_length 4096
   ```

3. Create a training YAML using the fields in `src/mha2mla/arguments.py`.
   Recovery training preserves the original required `--cfg_file` interface:

   ```bash
   torchrun --nproc_per_node 4 src/mha2mla/train.py \
     --cfg_file /path/to/train.yml
   ```

4. Use `eval/eval_v2.py` and `eval/longbench.py` for the main evaluation.
   LongBench likewise requires `--cfg_file`.
   `collect_attn_logits_max.py` and `collect_moe_vio.py` reproduce the QKNorm
   and MoE diagnostics. QKV-only checkpoints bundle their MLA tensors and load
   on top of the original base model; `src/mha2mla/export_mla_checkpoint.py`
   merges a delta into a full portable checkpoint.

Supply credentials for gated Hugging Face resources through the environment
when needed:

```bash
export HF_TOKEN=your_token
```

LongBench downloads `THUDM/LongBench` directly. Quantized and FP8 latent-cache
evaluation are controlled with `cache_implementation`, `backend`, `nbits`, and
`residual_length`; [`benchmark_decode.py`](../benchmark_decode.py) in the parent project directory measures decode
latency, throughput, and peak memory without a YAML file.

Do not commit credentials, model weights, generated tensors, checkpoints,
datasets, logs, virtual environments, or evaluation outputs.
# Full-parameter MoE control

For EP8 + ZeRO-2 + grouped GEMM + activation-checkpointed distillation, see
[the full-parameter MoE guide](EP_DISTILL.md).
