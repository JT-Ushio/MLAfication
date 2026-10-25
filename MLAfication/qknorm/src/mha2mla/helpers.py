import os
import sys
import math
import json
import logging
import shutil
from types import SimpleNamespace
from functools import partial

import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from transformers import get_scheduler
try:
    from transformers.trainer_utils import load_sharded_checkpoint
except ImportError:  # Historical Transformers 4.x evaluation environment.
    from transformers.modeling_utils import load_sharded_checkpoint
from safetensors.torch import load_file
from torch.utils.data import Subset

logger = logging.getLogger(__name__)


def bundle_mla_tensor(model, output_dir):
    metadata = getattr(model.config, "mha2mla", None)
    if not isinstance(metadata, dict):
        return None
    source = metadata.get("mla_tensor_path")
    if not source or not os.path.isfile(source):
        return None
    os.makedirs(output_dir, exist_ok=True)
    target_name = "mla_tensors.pt"
    target = os.path.join(output_dir, target_name)
    if os.path.abspath(source) != os.path.abspath(target):
        shutil.copy2(source, target)
    original = metadata["mla_tensor_path"]
    metadata["mla_tensor_path"] = target_name
    return metadata, original


def restore_mla_tensor_path(bundle_state):
    if bundle_state is not None:
        metadata, original = bundle_state
        metadata["mla_tensor_path"] = original


def _strip_wrapping_prefix(name):
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


def get_trainable_state_dict(model, state_dict=None):
    if state_dict is None:
        state_dict = model.state_dict()

    trainable_param_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    filtered_state_dict = {}
    for name, tensor in state_dict.items():
        normalized_name = _strip_wrapping_prefix(name)
        if normalized_name in trainable_param_names:
            filtered_state_dict[normalized_name] = tensor

    return filtered_state_dict


def _load_indexed_state_dict(path):
    safetensors_index = os.path.join(path, "model.safetensors.index.json")
    pytorch_index = os.path.join(path, "pytorch_model.bin.index.json")

    if os.path.exists(safetensors_index):
        index_path = safetensors_index
        shard_loader = load_file
    elif os.path.exists(pytorch_index):
        index_path = pytorch_index
        shard_loader = lambda shard_path: torch.load(shard_path, map_location="cpu", weights_only=True)
    else:
        return None

    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    state_dict = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = os.path.join(path, shard_name)
        state_dict.update(shard_loader(shard_path))

    return state_dict


def freeze_non_attn_weights(model):
    if dist.get_rank() == 0:
        print("......freeze_non_attn_weights......")
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "attn" in name or "norm" in name:
            param.requires_grad = True

def freeze_non_attn_weights_v2(model):
    if dist.get_rank() == 0:
        print("......freeze_non_attn_weights......")
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "attn" in name and "o_proj" not in name:
            param.requires_grad = True

def load_checkpoint(model, path, strict=True):
    config_path = os.path.join(path, "config.json")
    if os.path.isfile(config_path) and hasattr(model.config, "mla_knorm_mode"):
        with open(config_path, encoding="utf-8") as handle:
            checkpoint_config = json.load(handle)
        checkpoint_mode = checkpoint_config.get("mla_knorm_mode", "calibrated")
        if checkpoint_mode != model.config.mla_knorm_mode:
            raise ValueError(
                f"Checkpoint KNorm mode {checkpoint_mode!r} does not match model "
                f"mode {model.config.mla_knorm_mode!r}"
            )
    single_file = os.path.join(path, "model.safetensors")
    single_bin_file = os.path.join(path, "pytorch_model.bin")
    if os.path.exists(single_file):
        return model.load_state_dict(load_file(single_file), strict=strict)
    if os.path.exists(single_bin_file):
        return model.load_state_dict(
            torch.load(single_bin_file, map_location="cpu", weights_only=True), strict=strict
        )

    if strict:
        try:
            return load_sharded_checkpoint(model, path)
        except (FileNotFoundError, ValueError):
            pass

    indexed_state_dict = _load_indexed_state_dict(path)
    if indexed_state_dict is not None:
        return model.load_state_dict(indexed_state_dict, strict=strict)

    try:
        return load_sharded_checkpoint(model, path, strict=strict)
    except TypeError:
        if not strict:
            raise
        return load_sharded_checkpoint(model, path)


def save_checkpoint(model, path, tokenizer, trainable_only=False):
    if trainable_only:
        state_dict = get_trainable_state_dict(model)
    else:
        state_dict = model.state_dict()

    if not trainable_only and model.config.tie_word_embeddings:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
    bundle_state = bundle_mla_tensor(model, path)
    try:
        model.save_pretrained(path, state_dict=state_dict)
        tokenizer.save_pretrained(path)
    finally:
        restore_mla_tensor_path(bundle_state)


def enable_input_require_grads(model):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def load_optimizer_scheduler(model, train_args):
    """Load optimizer and scheduler from configuration."""
    optimizer_name = train_args.optim
    # TODO: remove weight_decay for bias and norm
    if "adam" in optimizer_name:
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=train_args.learning_rate,
            betas=(
                train_args.adam_beta1,
                train_args.adam_beta2,
            ),
            eps=train_args.adam_epsilon,
            weight_decay=train_args.weight_decay,
            fused=bool(train_args.optim == "adamw_torch_fused"),
        )
    else:
        raise ValueError(f"Unknown optimizer factory {optimizer_name}")

    if train_args.use_constant_with_warmup_decay_scheduler:
        lr_scheduler = lr_scheduler_builder(
            optimizer=optimizer,
            lr_scheduler_args=SimpleNamespace(**train_args.lr_scheduler_kwargs),
            total_training_steps=train_args.max_steps,
        )
    else:
        lr_scheduler = get_scheduler(
            train_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=train_args.warmup_steps,
            num_training_steps=train_args.max_steps,
        )
    return optimizer, lr_scheduler


def load_dataset(data_args, train_args, tokenizer):
    """Load dataset from configuration."""
    tokenizer.model_max_length = data_args.sequence_length

    if data_args.is_nanoset:
        from nanotron.data.nanoset import Nanoset

        token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if train_args is not None:
            global_batch_size = (
                train_args.per_device_train_batch_size
                * world_size
                * train_args.gradient_accumulation_steps
            )

            # 训练器真正要跑的步数
            run_num_steps = train_args.max_steps

            # 数据从哪个 optimizer step 开始取
            data_offset_steps = getattr(data_args, "data_offset_steps", 0)

            # 样本级 offset 和长度
            offset_num_samples = global_batch_size * data_offset_steps
            run_num_samples = global_batch_size * run_num_steps

            # 底层 Nanoset 必须至少覆盖 [0, offset + run) 这段
            total_num_samples = offset_num_samples + run_num_samples

            # RandomSampler shuffles the entire dataset. Shortening this length
            # changes even the first batch, so an early-stop control must retain
            # the original population, not merely the same Nanoset seed.
            sampling_steps = getattr(data_args, "data_sampling_steps", None)
            if sampling_steps is not None:
                if data_offset_steps != 0 or sampling_steps <= 0 or sampling_steps < run_num_steps:
                    raise ValueError(
                        "data_sampling_steps must be positive and >= max_steps, "
                        "with data_offset_steps=0"
                    )
                total_num_samples = global_batch_size * sampling_steps

            random_seed = data_args.trainset_seed
        else:
            offset_num_samples = 0
            run_num_samples = 256
            total_num_samples = 256
            random_seed = data_args.evalset_seed

        base_dataset = Nanoset(
            dataset_folders=data_args.dataset_folders,
            sequence_length=data_args.sequence_length,
            dataset_weights=data_args.dataset_weights,
            token_size=token_size,
            random_seed=random_seed,
            train_split_num_samples=total_num_samples,
        )

        # 对 deterministic 的 Nanoset 做样本级切片
        if offset_num_samples > 0:
            dataset = Subset(
                base_dataset,
                range(offset_num_samples, offset_num_samples + run_num_samples),
            )
        else:
            dataset = base_dataset

    else:
        import datasets

        dataset_name = data_args.hf_dataset_name_or_path
        if not dataset_name:
            raise ValueError(
                "hf_dataset_name_or_path is required when is_nanoset is false"
            )
        if os.path.isdir(dataset_name):
            dataset = datasets.load_from_disk(dataset_name)
            if isinstance(dataset, datasets.DatasetDict):
                dataset = dataset["train"]
        else:
            dataset = datasets.load_dataset(dataset_name, split="train")

        if "input_ids" not in dataset.column_names:
            text_column = os.getenv("MHA2MLA_TEXT_COLUMN", "text")
            if text_column not in dataset.column_names:
                raise ValueError(
                    f"Text column {text_column!r} is absent; available columns: "
                    f"{dataset.column_names}"
                )

            def tokenize(batch):
                return tokenizer(
                    batch[text_column],
                    truncation=True,
                    max_length=data_args.sequence_length,
                )

            dataset = dataset.map(
                tokenize,
                batched=True,
                remove_columns=dataset.column_names,
            )

    return dataset


# Portions adapted from Hugging Face Nanotron v0.5:
# https://github.com/huggingface/nanotron
# Licensed under the Apache License, Version 2.0.
# Modified by MLAfication Authors and Contributors.
# Changes made:
# - Changed optimizer.get_base_optimizer().param_groups to optimizer.param_groups
def lr_scheduler_builder(
    optimizer: Optimizer, lr_scheduler_args, total_training_steps: int
):
    if lr_scheduler_args.lr_decay_steps is None:
        lr_decay_steps = total_training_steps
        if lr_scheduler_args.lr_warmup_steps is not None:
            lr_decay_steps -= lr_scheduler_args.lr_warmup_steps
        if lr_scheduler_args.lr_decay_starting_step is not None:
            lr_decay_steps -= lr_scheduler_args.lr_decay_starting_step
    else:
        lr_decay_steps = lr_scheduler_args.lr_decay_steps

    if lr_scheduler_args.lr_decay_starting_step is None:
        if lr_scheduler_args.lr_warmup_steps is not None:
            lr_decay_starting_step = lr_scheduler_args.lr_warmup_steps
        else:
            lr_decay_starting_step = 0
    else:
        lr_decay_starting_step = lr_scheduler_args.lr_decay_starting_step

    def lr_lambda(current_step: int, initial_lr: float):
        """
        current_step: current training step
        initial_lr: the learning rate of a parameter group

        More info on initial_lr:
        And in standard parameterization, lr_lambda only takes a single learning rate.
        But in µTransfer, each parameter has a custom learning rate (custom_lr = lr_scheduler_args.learning_rate * scaling_factor),
        so each parameter group has a custom lr_lambda function.

        LR Scheduling function, it has from 2 up to 4 phases:
        - warmup,
        - optional: constant (if lr_decay_starting_step is set)
        - decay
        - optional: constant (if lr_decay_steps and/or lr_decay_starting_step are set)
        Warmup starts at lr=0 and ends at `lr=lr`
        Then it stays constant at lr if lr_decay_starting_step is set and larger than lr_warmup_steps
        Then it decays until `min_decay_lr` for lr_decay_steps if set, else: (total_training_steps - lr_warmup_steps or lr_decay_starting_step)
        Then it stays constant at min_decay_lr if lr_decay_starting_step is set and total_training_steps is larger)
        """
        # No warmup or decay
        if lr_scheduler_args.lr_warmup_steps == 0 and lr_decay_steps == 0:
            return initial_lr

        # Warmup phase
        elif (
            lr_scheduler_args.lr_warmup_style is not None
            and current_step <= lr_scheduler_args.lr_warmup_steps
        ):
            if lr_scheduler_args.lr_warmup_style == "linear":
                lmbda = (
                    initial_lr
                    * current_step
                    / max(lr_scheduler_args.lr_warmup_steps, 1)
                )
            elif lr_scheduler_args.lr_warmup_style == "constant":
                lmbda = lr_scheduler_args.learning_rate
            else:
                raise ValueError(
                    f"Unknown warmup style {lr_scheduler_args.lr_warmup_style}"
                )

        # Optional constant phase at learning_rate
        elif current_step < lr_decay_starting_step:
            lmbda = initial_lr

        # Decay phase
        elif (
            lr_scheduler_args.lr_decay_style is not None
            and current_step < lr_decay_starting_step + lr_decay_steps
        ):
            if lr_scheduler_args.lr_decay_style == "cosine":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (initial_lr - lr_scheduler_args.min_decay_lr)
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (current_step - lr_decay_starting_step)
                            / lr_decay_steps
                        )
                    )
                    / 2
                )
            elif lr_scheduler_args.lr_decay_style == "linear":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (initial_lr - lr_scheduler_args.min_decay_lr)
                    * (lr_decay_steps - (current_step - lr_decay_starting_step))
                    / lr_decay_steps
                )
            elif lr_scheduler_args.lr_decay_style == "1-sqrt":
                lmbda = lr_scheduler_args.min_decay_lr + (
                    initial_lr - lr_scheduler_args.min_decay_lr
                ) * (
                    1
                    - math.sqrt(
                        (current_step - lr_decay_starting_step) / lr_decay_steps
                    )
                )
            else:
                raise ValueError(
                    f"Unknown decay style {lr_scheduler_args.lr_decay_style}"
                )

        # Optional constant phase at min_decay_lr
        else:
            lmbda = lr_scheduler_args.min_decay_lr

        lmbda /= initial_lr  # Normalization for pytorch
        return lmbda

    def get_lr_lambda_for_param_group(lr: float):
        return partial(lr_lambda, initial_lr=lr)

    # NOTE: get learning rate scheduler for each param group
    # NOTE: Changes made.
    lr_lambdas = []
    for param_group in optimizer.param_groups:
        lr_lambdas.append(get_lr_lambda_for_param_group(lr=param_group["lr"]))

    assert len(lr_lambdas) == len(optimizer.param_groups), (
        "Custom learning rate functions dont match the number of param groups"
    )

    from nanotron.logging import log_rank

    log_rank(
        f"[Optimizer Building] There are total {len(lr_lambdas)} custom learning rate function for parameter groups",
        logger=logger,
        level=logging.DEBUG,
    )

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)
    return lr_scheduler
