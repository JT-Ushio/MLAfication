# Full-parameter Qwen3-MoE distillation

This runner supports the Qwen3-30B-A3B-Base rebuttal control: rank-256 MLA,
top-8 RoPE selection (68.75% KV reduction), and full-parameter distillation.
All student parameters, including routers and experts, are trainable. Teacher
and student have separate weights and both use EP8 with grouped GEMM. The
student additionally uses ZeRO-2 and activation checkpointing. No router
balancing loss or token dropping is added. This control changes more than
router freezing alone; a quality difference does not by itself isolate a
causal effect of router freezing.

The frozen teacher uses AutoEP directly without constructing a DeepSpeed
optimizer. Routers retain Transformers' sorted top-k selection and BF16
routing weights; rotary-frequency buffers retain FP32 precision. Grouped
BF16 reductions need not be bitwise identical to sequential expert reductions:
the smoke test checks output-norm error and independently compares averaged
gradients, including the expert-parallel scaling factor.

Use a separate environment with PyTorch 2.13, Transformers 5.10.4, Accelerate
1.15, and the Nanoset revision in [requirements](../requirements.txt).
The original DeepSpeed 0.15 environment is insufficient: its explicit MoE
layer does not implement this top-8 AutoEP path. Pin AutoEP to the tested source
revision instead of silently using a different version:

```bash
DS_BUILD_OPS=0 pip install --no-build-isolation \
  'deepspeed @ git+https://github.com/deepspeedai/DeepSpeed.git@fc7bc301f8301e72fed37e2caa3c1d40dae3bbb4'
```

Copy [the example configuration](cfgs/qwen3-30b-a3b-full-ep8.example.json) and
set the base model, conversion tensors, five tokenized data folders, and two
**new** run/checkpoint directories. Keep the original PEFT artifacts read-only.
The data mixture, seed, optimizer, LR schedule, global batch 64, and 4,800
updates match the control configuration. Datatrove returns 4,097 tokens when
configured with sequence length 4,096; this runner preserves the original
Trainer's untrimmed input and reports the actual token count. Its deterministic
sampler fixes each global batch across micro-batch choices and resumes at an
optimizer boundary; it does not claim bitwise identity with an earlier
Transformers Trainer's shuffled sample order.

Run on one node with eight H200 GPUs, enough host RAM to load the original
weights on all ranks, and space for two native optimizer checkpoints plus one
portable full model (allow roughly 1 TB):

```bash
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
torchrun --standalone --nproc_per_node=8 src/mha2mla/smoke_ep.py \
  --output /runs/mlafication/smoke-001
python src/mha2mla/run_ep_distill.py --config /runs/mlafication/run.json
```

The smoke test checks EP output and averaged router/expert gradients against
ordinary Transformers, AC backward, router/expert updates, a native checkpoint reload, and a portable checkpoint
reload without AutoEP. The driver then probes micro-batches 8, 4, 2, 1 in
descending order. Each fresh process performs three complete optimizer
updates, including optimizer-state allocation, and must retain 8 GiB of
reserved-memory headroom on every rank. Only CUDA OOM or insufficient headroom
allows a smaller candidate; other errors stop the run. The largest passing
candidate restarts from the original initialization for training. This is
capacity selection, not a claim that the largest candidate maximizes throughput.
Micro-batch 8 is the maximum compatible with global batch 64 on eight ranks.

The full-vocabulary KL is computed in token blocks with softmax recomputation
in backward. This avoids materializing several extra full-vocabulary FP32
tensors; it does not truncate vocabulary or change the temperature objective.
Both full BF16 logits tensors still occupy memory. No `torch.compile`, DeepEP,
CPU offload, or remote tracker is required.

Native checkpoints are saved every 480 steps with optimizer/scheduler state.
Only this run's completed native checkpoints are pruned, retaining the latest
two. Resume explicitly, preserving the configuration and micro-batch:

```bash
python src/mha2mla/run_ep_distill.py --config /runs/mlafication/run.json \
  --resume-tag step-000480
```

After the final native save, all ranks gather their expert weights one layer
at a time into `hf-step-004800`. This directory contains ordinary Transformers
expert keys, MLA attention weights, the tokenizer, bundled conversion tensors,
and `export_complete.json`. A rank-local AutoEP `state_dict` is **not** a full
evaluation checkpoint. Use the portable export with the existing qknorm
Commonsense, LongBench and `collect_moe_vio.py` evaluators. For paper §3.6's
MeanVio, average the per-layer maximum relative overload, not the per-expert
average positive overload. Preserve the original evaluation datasets and seeds.

`train-metrics.jsonl` and `train-status.json` record loss, gradient norm,
throughput, actual tokens, chosen micro-batch, and the maximum CUDA memory
across all ranks. A successful probe alone does not establish a healthy run;
check continued optimizer-step progress and finite metrics over time.

To chain the three evaluations after training, run
`python eval/evaluate_full_ep.py --config /runs/mlafication/eval.json`.
The evaluation JSON supplies `checkpoint`, `output_dir`, `model_name_or_path`,
`peft_checkpoint`, `data_root`, `tasks_file`, `custom_tasks`, and
`longbench_source` (the existing local LongBench dataset script, or its dataset
identifier). It runs the supplied original Commonsense tasks, BF16 LongBench
at 4K, then Vio for the baseline, full model and PEFT checkpoint on the same
128 held-out samples (seed 42, sample offset 512000). `paper_vio.json` computes
the maximum and mean of the per-layer maxima, matching Table 8.

The original local benchmark builders were validated with LightEval 0.6.2 and
Datasets 3.6.0 in the separate experiment environment. Newer Datasets versions
do not support those Python dataset builders; using one requires migrating
the local datasets and task definitions first. Small-model checks exercise all
Commonsense task entries, all 21 LongBench datasets and cached batched decoding,
and standalone baseline/full-checkpoint Vio. These are interface checks, not
benchmark scores for the trained 30B model.

Training and evaluation preserve FP32 RoPE frequency buffers when casting
weights to BF16. Older evaluation code cast these buffers too, which can
change rotations at long positions. Re-evaluate comparison checkpoints with
the same evaluator/runtime when comparing scores; the Vio pipeline does this
for the baseline, PEFT and full-parameter models automatically.
Set `evaluate_peft_reference: true` in the evaluation JSON to also re-evaluate
the existing PEFT checkpoint on Commonsense and LongBench. Its outputs go to
`peft-reference/`, with both models' summaries in
`paper_quality_comparison.json`. This performs no additional training.
