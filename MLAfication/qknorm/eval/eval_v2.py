# Portions of this file are adapted from Hugging Face LightEval v0.6.0.
# Copyright (c) 2024 Hugging Face.
# Licensed under the MIT License; see ../../THIRD_PARTY_NOTICES.md.
# Modifications are licensed under the Apache License, Version 2.0.

import argparse
import os, sys
from typing import TypeVar

import torch.utils.data.distributed as torch_data_distributed

# Lighteval 0.6.x imports this private Torch typing alias, which was removed
# from torch.utils.data.distributed in newer PyTorch releases.
if not hasattr(torch_data_distributed, "T_co"):
    torch_data_distributed.T_co = TypeVar("T_co", covariant=True)

from transformers.trainer_utils import load_sharded_checkpoint

from lighteval.models.utils import _get_dtype, _simplify_name, batched
from lighteval.pipeline import (
    EnvConfig,
)
import transformers
from lighteval.parsers import (
    parser_accelerate,
)
from lighteval.models.model_config import BaseModelConfig
from transformers import AutoModelForCausalLM, AutoConfig
import types
from lighteval.models.model_loader import BaseModel

# Lighteval 0.6.2 can create an empty final dynamic split for small request
# counts. A single split is equivalent when an explicit batch size is used.
BaseModel.DATASET_SPLITS = 1

current_file_path = os.path.abspath(__file__)
target_directory = os.path.join(
    os.path.dirname(os.path.dirname(current_file_path)), "src", "mha2mla"
)
sys.path.append(str(target_directory))

from safetensors.torch import load_file
from helpers import cast_model_parameters, load_checkpoint


def create_load_func(mha2mla_args):
    def _create_auto_model(
        self, config: BaseModelConfig, env_config: EnvConfig
    ) -> transformers.PreTrainedModel:
        config.model_parallel, max_memory, device_map = self.init_model_parallel(
            config.model_parallel
        )
        torch_dtype = _get_dtype(config.dtype, self._config)
        ckpt_path = config.pretrained
        if mha2mla_args is None or mha2mla_args.is_baseline:
            path = ckpt_path if mha2mla_args is None else mha2mla_args.model_name_or_path
            model = AutoModelForCausalLM.from_pretrained(
                path,
                revision=config.revision
                + (f"/{config.subfolder}" if config.subfolder is not None else ""),
                max_memory=max_memory,
                device_map=device_map,
                dtype=torch_dtype,
                trust_remote_code=config.trust_remote_code,
                cache_dir=env_config.cache_dir,
                offload_folder=env_config.cache_dir,
                token=env_config.token,
                quantization_config=config.quantization_config,
            )
            if ckpt_path != path:
                load_checkpoint(model, ckpt_path, strict=False)
        else:
            # Instance
            if (
                not os.path.isabs(mha2mla_args.mla_tensor_path)
                and os.path.isdir(ckpt_path)
            ):
                mha2mla_args.mla_tensor_path = os.path.join(
                    ckpt_path, mha2mla_args.mla_tensor_path
                )
            model_config = AutoConfig.from_pretrained(ckpt_path)
            mha_model = AutoModelForCausalLM.from_pretrained(
                mha2mla_args.model_name_or_path,
                trust_remote_code=config.trust_remote_code,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            from patching_mla_fwd import patch_mla_fwd_instance
            from patching_mlafication import patch_mlafication
            mla_model = patch_mlafication(mha_model, model_config, mha2mla_args)
            patch_mla_fwd_instance(mla_model)
            # Load weights
            load_checkpoint(mla_model, ckpt_path, strict=False)
            model = cast_model_parameters(mla_model, torch_dtype)
        return model

    return _create_auto_model


# Adapted from lighteval.main.
def cli_evaluate():  # noqa: C901
    parser = argparse.ArgumentParser(
        description="CLI tool for lighteval, a lightweight framework for LLM evaluation"
    )

    subparsers = parser.add_subparsers(help="help for subcommand", dest="subcommand")

    # Subparser for the "accelerate" command
    parser_a = subparsers.add_parser(
        "accelerate", help="use accelerate and transformers as backend for evaluation."
    )
    parser_accelerate(parser_a)

    args = parser.parse_args()
    model_args = args.model_args
    model_args = {
        k.split("=")[0]: k.split("=")[1] if "=" in k else True
        for k in model_args.split(",")
    }
    ckpt_path = model_args["pretrained"]
    model_config = AutoConfig.from_pretrained(ckpt_path)
    if hasattr(model_config, "mha2mla"):
        mha2mla_args = types.SimpleNamespace(**model_config.mha2mla)
    else:
        mha2mla_args = None
    BaseModel._create_auto_model = create_load_func(mha2mla_args)

    if args.subcommand == "accelerate":
        from lighteval.main_accelerate import main as original_main_accelerate

        # main_accelerate(args, model, ckpt_path)
        original_main_accelerate(args)
    else:
        print("You need to set the subcommand to 'accelerate'.")


if __name__ == "__main__":
    cli_evaluate()
