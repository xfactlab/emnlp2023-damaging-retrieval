from dataclasses import dataclass, field
from transformers import (
    SchedulerType,
)
from typing import Any, Dict, List, Optional, Union

@dataclass
class CustomTrainingArguments:
    """
    CustomTrainArguments Class for Training without Huggingface Trainer
    """

    do_train: bool = field(
        default=True,
        metadata={"help": "train options"}
    )

    do_eval: bool = field(
        default=True,
        metadata={"help": "eval options"}
    )

    do_predict: bool = field(
        default=False,
        metadata={"help" : "predict options"}
    )

    with_tracking: bool = field(
        default=True,
        metadata={"help": "Tracking Experiment"},
    )

    report_to: str = field(
        default='wandb',
        metadata={
            "help": "wandb-logging"
        },
    )

    wandb_project: str = field(
        default='binary_classifier',
        metadata={
            "help": "wandb_project name"
        },
    )

    run_name: str = field(
        default='testing_from_scratch',
        metadata={
            "help": "run name for wandb project"
        },
    )

    output_dir: str = field(
        default='/data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/testing-scratch',
        metadata={
            "help": "Output directory"
        },
    )

    per_device_train_batch_size: int = field(
        default=8,
        metadata={
            "help": "per_device_train_batch_size"
        },
    )

    per_device_eval_batch_size: int = field(
        default=8,
        metadata={
            "help": "per_device_eval_batch_size"
        },
    )

    seed: int = field(
        default=42,
        metadata={
            "help": "Seed for Training"
        },
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})

    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )

    max_train_steps: int = field(
        default=None,
        metadata={
            "help": "Total number of training steps to perform. If provided, overrides num_train_epochs."
        },
    )

    num_train_epochs: int = field(
        default=3,
        metadata={
            "help": "Total number of training epochs to perform."
        },
    )

    num_warmup_steps: int = field(
        default=0,
        metadata={
            "help": "Number of steps for the warmup in the lr scheduler."
        },
    )

    checkpointing_steps: str = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
        },
    )

    train_loss_steps: int = field(
        default=10,
        metadata={
            "help": "train_loss_steps"
        },
    )

    save_max_limit: int = field(
        default=3,
        metadata={
            "help": "maximum number of models"
        },
    )

    best_metric: str = field(
        default='accuracy',
        metadata={
            "help": "maximum number of models"
        },
    )

    headtype: str = field(
        default='adaptive',
        metadata={
            "help": "classifier head type"
        },
    )

    class_weights: bool = field(
        default='False',
        metadata={
            "help": "class weights"
        },
    )

    num_layers: Optional[int] = field(
        default=None,
        metadata={
            "help": "num_layers for sentence-level classifier"
        },
    )

    drop_out_rate: Optional[float] = field(
        default=None,
        metadata={
            "help": "drop_out_rate for sentence-level classifier"
        },
    )

    padding: Optional[int] = field(
        default=-100,
        metadata={
            "help": "padding value for sentence-level classifier "
        },
    )

    train_num_workers: int = field(
        default=4,
        metadata={
            "help": "Total number of training epochs to perform."
        },
    )

    eval_num_workers: int = field(
        default=4,
        metadata={
            "help": "Total number of training epochs to perform."
        },
    )




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_architecture: str = field(
        default="roberta-large",
        metadata={
            "help": "Base model architecture"
        },
    )

    model_name_or_path: str = field(
        default="roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    config_base_path: str = field(
        default="/data/philhoon-relevance/FiD/pretrained_models/nq_reader_large",
        metadata={
            "help": "config base path only for binary_classifier_encoder_predict.py"
        },
    )

    prediction_model_name_or_path: str = field(
        default="/data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/roberta-decisive_binary_gold_data_trial1",
        metadata={
            "help": "prediction model path"
        },
    )

    prediction_model_step: str = field(
        default="380",
        metadata={
            "help": "prediction model step"
        },
    )

    git_tag: str = field(
        default="v1.1",
        metadata={
            "help": "Git Tag Number"
        },
    )

    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )

    max_seq_length: int = field(
        default=200,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    n_layer: int = field(
        default=6,
        metadata={
            "help": "n_layer of gpt decoder"
        },
    )

    block_size: int = field(
        default=20,
        metadata={
            "help": "n_context for gpt"
        },
    )

    # wandb_project: str = field(
    #     default="binary_classifier",
    #     metadata={
    #         "help": "wandb_project name"
    #     },
    # )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data: str = field(
        default="NQ-DEV-DPR/5-fold/1",
        metadata={"help": "The name of the dataset to use."},
    )

    train_file: Optional[str] = field(
        default="/data/philhoon-relevance/binary-classification/\
NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_train_1_partial.json",
        metadata={"help": "The name of the train dataset to use."},
    )

    # dev_size: Optional[int] = field(
    #     default=3000,
    #     metadata={"help": "development size "},
    # )

    eval_file: Optional[str] = field(
        default="/data/philhoon-relevance/binary-classification/\
NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_train_1_partial.json",
        metadata={"help": "The name of the test dataset to use."},
    )

    ref_eval: Optional[str] = field(
        default="/scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/ctx100id.json",
        metadata={"help": "Ref_eval file for gpt-decoder."},
    )

    intact_eval: Optional[bool] = field(
        default="False",
        metadata={"help": "For prediction, Testing on Intact Test input"},
    )

    num_labels: Optional[int] = field(
        default=2,
        metadata={"help": "default number of labes"},
    )

    # sample_size: int = field(
    #     default=5,
    #     metadata={"help": "number of sample size including positive context"},
    # )
    #
    # position: int = field(
    #     default=1,
    #     metadata={"help": "position of positive sample (range from 1 to sample_size"},
    # )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."
        },
    )

    dataset_class: Optional[str] = field(
        default='BinaryCustomDatasetShuffle',
        metadata={
            "help": "DatasetClass for Training"
        },
    )



