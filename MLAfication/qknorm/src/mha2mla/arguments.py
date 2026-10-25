from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments


@dataclass
class MHA2MLAModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model"},
    )
    mla_tensor_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pre-computed MLA tensor file, e.g., 'utils/mla_tensors_135M.pth'"
        },
    )
    knorm_mode: Optional[str] = field(
        default=None,
        metadata={"help": "calibrated or exact; None infers the mode from the MLA tensors"},
    )
    output_mla_dir: str = field(
        default=None,
        metadata={"help": "Directory to save the patched MLA model"},
    )
    is_baseline: bool = field(
        default=False,
        metadata={"help": "Whether to use the baseline MHA model without MLA patching"},
    )

    def __post_init__(self):
        if self.knorm_mode not in {None, "calibrated", "exact"}:
            raise ValueError("knorm_mode must be calibrated, exact, or null")
        if hasattr(super(), "__post_init__"):
            super().__post_init__()


@dataclass
class MHA2MLADataArguments:
    is_nanoset: bool = field(
        default=False,
        metadata={
            "help": "use nanoset dataloader (False means use Huggingface datasets)"
        },
    )
    dataset_folders: Optional[List[str]] = field(
        default=None, metadata={"help": "List of dataset folders to use"}
    )
    dataset_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "Weights for each dataset when mixing multiple datasets"},
    )
    hf_dataset_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Huggingface Dataset name or path"}
    )
    sequence_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length"}
    )
    trainset_seed: int = field(
        default=42, metadata={"help": "Random seed for trainset"}
    )
    evalset_seed: int = field(
        default=1234, metadata={"help": "Random seed for evalset"}
    )
    data_offset_steps: int = field(
        default=0,
        metadata={
            "help": "Start reading the deterministic training data stream from this optimizer-step offset."
        },
    )
    data_sampling_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Keep the original Nanoset/sampler population when stopping a run early. "
                    "Must cover max_steps and cannot be combined with data_offset_steps."
        },
    )


@dataclass
class MHA2MLATrainingArguments(TrainingArguments):
    use_constant_with_warmup_decay_scheduler: bool = field(
        default=False,
        metadata={"help": "Whether to use constant with warmup decay scheduler"},
    )
    is_freeze_non_attn: bool = field(
        default=False,
        metadata={"help": "if the finetuning is freeze non attention parameters"},
    )
    kl_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for KL divergence loss in knowledge distillation"
        },
    )
