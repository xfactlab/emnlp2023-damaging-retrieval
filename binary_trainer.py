import json
import os
import logging
import sys
import evaluate
from util import utils

import transformers
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from util.arguments import ModelArguments, DataTrainingArguments
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BinaryCustomDataset(torch.utils.data.Dataset):
    def __init__(self, instances, tokenizer, max_length):
        self.instances = instances
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.max_length = max_length

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        input_ = 'question: ' + self.instances[idx]['question'] + \
                 ' title: ' + self.instances[idx]['ctx']['title'] + \
                 ' context : ' + self.instances[idx]['ctx']['text']
        output = self.tokenizer(
            input_,
            # return_tensors="pt", will be applied later through collator
            # padding=True, will be padded later through collate
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length)

        item = {key: val for key, val in output.items()}
        # item['labels'] = torch.tensor(int(self.instances[idx]['em']))
        item['labels'] = int(self.instances[idx]['em'])

        return item


class BinaryCustomDatasetShuffle(torch.utils.data.Dataset):
    def __init__(self, instances, tokenizer, max_length, shuffle = False):
        if shuffle:
            random.shuffle(instances)
        self.instances = instances
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.max_length = max_length

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        input_ = 'question: ' + self.instances[idx]['question'] + \
                 ' title: ' + self.instances[idx]['ctx']['title'] + \
                 ' context : ' + self.instances[idx]['ctx']['text']
        output = self.tokenizer(
            input_,
            # return_tensors="pt", will be applied later through collator
            # padding=True, will be padded later through collate
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length)

        item = {key: val for key, val in output.items()}
        # item['labels'] = torch.tensor(int(self.instances[idx]['em']))
        item['labels'] = int(self.instances[idx]['em'])

        return item


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    os.environ['WANDB_PROJECT'] = model_args.wandb_project
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Change output_dir by datasetname_modelname_tag_position_sample_size_
    out_temp = f'{data_args.data}-{model_args.model_architecture}'
    training_args.output_dir = os.path.join(training_args.output_dir, out_temp)
    logger.info(f"output directory : {training_args.output_dir}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=data_args.num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    if training_args.do_train:
        binary_train = utils.open_json(data_args.train_file)

        train_dataset = BinaryCustomDatasetShuffle(binary_train,
                                      tokenizer,
                                      model_args.max_seq_length)

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        binary_eval = utils.open_json(data_args.eval_file)

        eval_dataset = BinaryCustomDatasetShuffle(binary_eval,
                                     tokenizer,
                                     model_args.max_seq_length)

    metric = evaluate.load("xnli")
    metric_prec = evaluate.load('precision')
    metric_recall = evaluate.load('recall')
    metric_f1 = evaluate.load('f1')

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        accr_ = metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
        prec_ = metric_prec.compute(predictions=preds, references=p.label_ids)["precision"]
        recall_ = metric_recall.compute(predictions=preds, references=p.label_ids)["recall"]
        f1_ = metric_f1.compute(predictions=preds, references=p.label_ids)["f1"]

        return {
            'accuracy' : accr_,
            'precision' : prec_,
            'recall' : recall_,
            'f1' : f1_,
        }

    # Initialize Trainer
    data_collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        # max_train_samples = (
        #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        # )
        max_train_samples = (
            len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(test_dataset)
        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()