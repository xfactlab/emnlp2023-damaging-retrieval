from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import heapq
import pathlib
import shutil

from pprint import pprint
from tqdm.auto import tqdm
from src.data import (
    BinaryCustomDatasetShuffle,
    BinarySentenceDataset,
    BinaryCustomDatasetDecisiveBinaryGold,
    BinaryCustomDatasetPredictionShuffle
)

import json
import math
import os
import logging
import sys
import evaluate
from util import utils

import transformers
import torch
import numpy as np
import random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    set_seed,
    get_scheduler,
)
from util.arguments import ModelArguments, DataTrainingArguments, CustomTrainingArguments
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

NEW_LINE = "\n"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATASET_MAPPING = {
    "BinaryCustomDatasetShuffle" : BinaryCustomDatasetShuffle,
    "BinarySentenceDataset" : BinarySentenceDataset,
    'BinaryCustomDatasetDecisiveBinaryGold' : BinaryCustomDatasetDecisiveBinaryGold,
    'BinaryCustomDatasetPredictionShuffle' : BinaryCustomDatasetPredictionShuffle
}

# def eval(model, eval_dataloader, accelerator, metric_acc,
#          metric_pre, metric_re, metric_f1, train_args, epoch, steps, output_dir, logger):
def eval(model_args, data_args, train_args):

    logger = get_logger(__name__)
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if not data_args.intact_eval:
        train_args.output_dir = os.path.join(model_args.prediction_model_name_or_path,
                                             f'step_{model_args.prediction_model_step}', 'partial_prediction')
    else:
        train_args.output_dir = os.path.join(model_args.prediction_model_name_or_path,
                                             f'step_{model_args.prediction_model_step}', 'intact_prediction')

    if accelerator.is_main_process and train_args.output_dir is not None:
        os.makedirs(train_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    config = AutoConfig.from_pretrained(model_args.prediction_model_name_or_path, num_labels=data_args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_args.prediction_model_name_or_path)
    pytorch_model_path = os.path.join(model_args.prediction_model_name_or_path,
                                      f'step_{model_args.prediction_model_step}')
    model = AutoModelForSequenceClassification.from_pretrained(
        pytorch_model_path,
        config=config,
    )

    eval_file = data_args.eval_file
    eval_data = utils.open_json(eval_file)
    DataSetClass = DATASET_MAPPING[data_args.dataset_class]
    eval_dataset = DataSetClass(eval_data, tokenizer=tokenizer,
                                max_length=model_args.max_seq_length, shuffle=False)

    for index in random.sample(range(len(eval_dataset)), 5):
        logger.info(f"Sample {index} of the eval set: {eval_dataset[index]}.")

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    eval_dataloader = DataLoader(eval_dataset,
                                 shuffle=False,
                                 collate_fn=data_collator,
                                 batch_size=train_args.per_device_eval_batch_size,
                                 )

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    # Get the metric function
    metric_acc = evaluate.load("accuracy")
    metric_pre = evaluate.load('precision')
    metric_re = evaluate.load('recall')
    metric_f1 = evaluate.load('f1')

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {train_args.per_device_eval_batch_size}")
    logger.info(f"  Steps = {math.ceil(len(eval_dataset) / train_args.per_device_eval_batch_size) + 1}")


    # Saving model_args, data_args, train_args
    train_dict = vars(train_args)
    logger.info(f"  Saving training_args = {train_dict}")
    with open(os.path.join(train_args.output_dir, "train_args.json"), "w") as f:
        json.dump(train_dict, f)

    model_dict = vars(model_args)
    logger.info(f"  Saving model_args = {model_dict}")
    with open(os.path.join(train_args.output_dir, "model_args.json"), "w") as f:
        json.dump(model_dict, f)

    data_dict = vars(data_args)
    logger.info(f"  Saving data_args = {data_dict}")
    with open(os.path.join(train_args.output_dir, "data_args.json"), "w") as f:
        json.dump(data_dict, f)

    eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

    eval_loss = 0
    model.eval()
    samples_seen = 0
    prediction_lst = []
    reference_lst = []

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
        if train_args.with_tracking:
            eval_loss += loss.detach().float()

        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]

        metric_acc.add_batch(
            predictions=predictions,
            references=references,
        )
        metric_pre.add_batch(
            predictions=predictions,
            references=references,
        )
        metric_re.add_batch(
            predictions=predictions,
            references=references,
        )
        metric_f1.add_batch(
            predictions=predictions,
            references=references,
        )
        eval_progress_bar.update(1)
        prediction_lst.extend(predictions.detach().cpu().tolist())
        reference_lst.extend(references.detach().cpu().tolist())

    eval_metric = metric_acc.compute()
    eval_metric_pre = metric_pre.compute()
    eval_metric_re = metric_re.compute()
    eval_metric_f1 = metric_f1.compute()

    logger.info(f"Accuracy : {eval_metric['accuracy']} Precision : {eval_metric_pre['precision']}")
    logger.info(f"Recall : {eval_metric_re['recall']} F1 : {eval_metric_f1['f1']}")
    logger.info(f"Eval_loss : {eval_loss.item() / len(eval_dataloader)}")

    result_log = {
        "eval_accuracy": eval_metric['accuracy'],
        "eval_precision": eval_metric_pre['precision'],
        "eval_recall": eval_metric_re['recall'],
        "eval_f1": eval_metric_f1['f1'],
        "eval_loss": eval_loss.item() / len(eval_dataloader),
    }

    output_result_path = os.path.join(train_args.output_dir, 'result.json')
    with open(output_result_path, "w") as f:
        json.dump(result_log, f)

    prediction_np = np.array(prediction_lst)
    reference_np = np.array(reference_lst)

    for ins, p_, r_ in zip(eval_data, prediction_np, reference_np):
        if str(r_) != ins['em']:
            logger.info(f"Not Matching Instance")
        ins['binary_inference'] = str(p_)

    predcition_output_path = os.path.join(train_args.output_dir, 'prediction.json')
    with open(predcition_output_path, "w") as f:
        json.dump(eval_data, f)

def train(model_args, data_args, train_args):
    return None

if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    model_args, data_args, train_args = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    # os.environ['WANDB_PROJECT'] = model_args.wandb_project

    # Train & Eval
    if train_args.do_train:
        print('This script does not support train. Use binary_classifier.py for training')
        train(model_args, data_args, train_args)
        exit()
    # Eval & Prediction only
    if not train_args.do_train and train_args.do_eval and train_args.do_predict:
        eval(model_args, data_args, train_args)

