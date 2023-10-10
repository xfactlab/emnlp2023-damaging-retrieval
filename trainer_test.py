import json
import os
import logging
import sys
import evaluate
from copy import deepcopy

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


def open_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def preprocessing_data(json_file, sample_size: int, position: int):
    """
    sample_size : one to five
        e.g.)
            positive_sample = 1 positive passage + n-1 negative passage
            negative_sample = n negative passage
    cut_off : number of questions discarded when there is not enough negative passages
    position : position of positive passage (1 ~ n)
        e.g.) n = 2, position = 1
            instance = [negative passage, positive passage]
    """
    cut_off = 0
    instances = []
    sample_size = sample_size
    position = position
    total_questions = len(json_file)

    for samples in json_file:
        answer = samples['answers']
        question = samples['question']
        negative_samples = []

        # 'hard_negative_ctxs' should be at least equal to sample_size
        # 'positive_ctx' which contains the answer should be at least one
        if len(samples['hard_negative_ctxs']) < sample_size or len(samples['positive_ctxs']) < 1:
            cut_off += 1
        else:
            cnt_negative_sample = 0
            for negative_sample in samples['hard_negative_ctxs']:
                if cnt_negative_sample > sample_size - 1:
                    break
                ng_s = negative_sample['text'].replace('\n', ' ')
                negative_samples.append(ng_s)
                cnt_negative_sample += 1

            # 'hard_negative_ctxs' sorted by its score, so shuffle them
            random.shuffle(negative_samples)

            # replace 1 negative_sample with one positive_sample in designated position
            positive_sample = samples['positive_ctxs'][0]['text'].replace('\n', ' ')
            positive_samples = deepcopy(negative_samples)
            positive_samples[position - 1] = positive_sample

            negative_template = {
                'text': negative_samples,
                'labels': 0,
                'answer': answer,
                'question': question,
            }
            positive_template = {
                'text': positive_samples,
                'labels': 1,
                'answer': answer,
                'question': question,
                'pos': position,
            }
            instances.append(negative_template)
            instances.append(positive_template)

    return instances, cut_off, total_questions


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, instances, tokenizer, max_length):
        self.instances = instances
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.max_length = max_length

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        input_ = [' ' + self.instances[idx]['question']] + self.instances[idx]['text']
        input_txt = f' {self.sep_token} '.join(input_) + ' '

        output = self.tokenizer(
            input_txt,
            # return_tensors="pt", will be applied later through collate
            # padding=True, will be padded later through collate
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length)

        item = {key: val for key, val in output.items()}
        item['labels'] = torch.tensor(self.instances[idx]['labels'])

        return item


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(return_remaining_strings=False)

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
    out_temp = f'{data_args.data}-{model_args.model_architecture}-samplesize{data_args.sample_size}-position{data_args.position}-tag{model_args.git_tag}'
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
        nq_dpr_train = open_json(data_args.train_file)
        instances, cut_off, total_questions = preprocessing_data(
            nq_dpr_train,
            data_args.sample_size,
            data_args.position)

        train_instance = instances[data_args.dev_size:]
        dev_instance = instances[:data_args.dev_size]

        train_dataset = CustomDataset(train_instance,
                                      tokenizer,
                                      model_args.max_seq_length)
        dev_dataset = CustomDataset(dev_instance,
                                    tokenizer,
                                    model_args.max_seq_length)

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        nq_dpr_test = open_json(data_args.test_file)
        test_instances, cut_off, total_questions = preprocessing_data(
            nq_dpr_test,
            data_args.sample_size,
            data_args.position)

        test_dataset = CustomDataset(test_instances,
                                     tokenizer,
                                     model_args.max_seq_length)

    metric = evaluate.load("xnli")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Initialize Trainer
    data_collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=dev_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=30)]
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
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=test_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(test_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(test_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()