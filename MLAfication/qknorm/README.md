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

For the Qwen3-8B rank512/top-k8 quality control that preserves token-wise KNorm
without its calibrated approximation, see [the exact-KNorm guide](EXACT_KNORM.md).
This baseline supports ordinary expanded K/V caching, but not standard
latent-cache/MQA decoding.

For Table 1, the original `gen/compare_expect.py` compares mean key RMS norms
across 1K–16K contexts after Partial-RoPE/SVD initialization. The same pass now
also prints `Ek_std`, the population standard deviation across valid tokens
for each KV head in the middle layer. It uses the existing calibration dataset
settings; run with `MODEL_NAME_OR_PATH=/path/to/model NUM_SAMPLES=64 torchrun
--standalone --nproc_per_node=1 gen/compare_expect.py` in the original experiment
environment. `NUM_SAMPLES` defaults to the original script's 128 samples.
