from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import heapq
import pickle
import pathlib
import shutil
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from pprint import pprint
from tqdm.auto import tqdm
from src.data import (
    BinaryCustomDatasetShuffle,
    BinarySentenceDataset,
    BinaryCustomDatasetDecisiveBinaryGold,
    BinaryCustomDatasetPredictionShuffle,
    SentenceClassificationDataset,
    EncoderSentenceClassificationDataset
)

from functools import partial
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
from sentence_transformers import SentenceTransformer
from FiD.src.model import FiDT5
from src.model import SentenceLSTM

NEW_LINE = "\n"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATASET_MAPPING = {
    "BinaryCustomDatasetShuffle" : BinaryCustomDatasetShuffle,
    "BinarySentenceDataset" : BinarySentenceDataset,
    'BinaryCustomDatasetDecisiveBinaryGold' : BinaryCustomDatasetDecisiveBinaryGold,
    'BinaryCustomDatasetPredictionShuffle' : BinaryCustomDatasetPredictionShuffle,
    'SentenceClassificationDataset' : SentenceClassificationDataset,
    'EncoderSentenceClassificationDataset' : EncoderSentenceClassificationDataset
}
EMBEDDING_ARC_MAPPING = {
    "SentenceTransformer" : SentenceTransformer,
     "FiDT5" : FiDT5
}


def eval(model, eval_dataloader, accelerator, metric_acc, metric_pre, metric_re, metric_f1,
         train_args, epoch, steps, output_dir, logger):
    eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

    eval_loss = 0
    model.eval()
    samples_seen = 0
    prediction_lst = []
    reference_lst = []

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            logits = model(batch['inputs'], batch['sequence_len'])
            if train_args.class_weights:
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean', ignore_index=train_args.padding).cuda()
            else:
                criterion = torch.nn.CrossEntropyLoss(ignore_index=train_args.padding).cuda()
            loss = criterion(logits.view(-1, logits.shape[-1]), batch['labels'].view(-1))

        if train_args.with_tracking:
            eval_loss += loss.detach().float()

        predictions = logits.argmax(dim=-1)
        references = batch['labels']

        # Get mask for target values != padding index
        nonpad_mask = references != train_args.padding

        # Slice out non-pad values
        references = references[nonpad_mask]
        predictions = predictions[nonpad_mask]

        predictions, references = accelerator.gather((predictions, references))
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

    logger.info(f"Evaluation at Epoch : {epoch} Total Step : {steps}")
    logger.info(f"Accuracy : {eval_metric['accuracy']} Precision : {eval_metric_pre['precision']}")
    logger.info(f"Recall : {eval_metric_re['recall']} F1 : {eval_metric_f1['f1']}")
    logger.info(f"Epoch : {epoch} Step : {steps}")
    logger.info(f"Eval_loss : {eval_loss.item() / len(eval_dataloader)}")

    result_log = {
        "eval_accuracy": eval_metric['accuracy'],
        "eval_precision": eval_metric_pre['precision'],
        "eval_recall": eval_metric_re['recall'],
        "eval_f1": eval_metric_f1['f1'],
        "eval_loss": eval_loss.item() / len(eval_dataloader),
        "epoch": epoch,
        "step": steps,
    }

    output_result_path = os.path.join(output_dir, f"epoch{epoch}_steps{steps}_results.json")
    with open(output_result_path, "w") as f:
        json.dump(result_log, f)

    if train_args.with_tracking:
        accelerator.log(
            result_log,
            step=steps,
        )

    ## Extra
    prediction_np = np.array(prediction_lst)
    reference_np = np.array(reference_lst)
    y_actu = pd.Series(reference_np, name='Actual')
    y_pred = pd.Series(prediction_np, name='Predicted')

    reversey_pred = y_pred.map(lambda x: 0 if x == 1 else 1)
    reversey_actu = y_actu.map(lambda x: 0 if x == 1 else 1)
    rev_accuracy = accuracy_score(reversey_actu, reversey_pred)
    rev_precision = precision_score(reversey_actu, reversey_pred)
    rev_recall = recall_score(reversey_actu, reversey_pred)
    rev_f1 = f1_score(reversey_actu, reversey_pred)

    logger.info(f"rev Evaluation at Epoch : {epoch} Total Step : {steps}")
    logger.info(f"rev_Accuracy : {rev_accuracy} rev_Precision : {rev_precision}")
    logger.info(f"rev_Recall : {rev_recall} rev_F1 : {rev_f1}")
    logger.info(f"Epoch : {epoch} Step : {steps}")
    logger.info(f"Eval_loss : {eval_loss.item() / len(eval_dataloader)}")

    result_rev_log = {
        "eval_rev_accuracy": rev_accuracy,
        "eval_rev_precision": rev_precision,
        "eval_rev_recall": rev_recall,
        "eval_rev_f1": rev_f1,
        "eval_loss": eval_loss.item() / len(eval_dataloader),
        "epoch": epoch,
        "step": steps,
    }

    output_result_path = os.path.join(output_dir, f"epoch{epoch}_steps{steps}_rev_results.json")
    with open(output_result_path, "w") as f:
        json.dump(result_rev_log, f)

    if train_args.with_tracking:
        accelerator.log(
            result_rev_log,
            step=steps,
        )

    return result_log, output_dir


def custom_collate(batch, padding):
    train_lst = [b['input_embedding'] for b in batch]
    label_lst = [b['em_pattern'] for b in batch]
    seq_len_lst = [b['em_pattern'].shape[0] for b in batch]
    max_seq_len = max(seq_len_lst)

    padding_train_lst = []
    for embedding in train_lst:
        if embedding.shape[0] < max_seq_len:
            post_pad = torch.full(size=(max_seq_len - embedding.shape[0], embedding.shape[1]), fill_value=padding)
            # post_pad = torch.full(size=(max_seq_len - embedding.shape[0], embedding.shape[1]), fill_value=-100)
            padding_train_lst.append(torch.concat([embedding, post_pad]))
        else:
            padding_train_lst.append(embedding)

    inputs = torch.stack(padding_train_lst)

    padding_label_lst = []
    for label in label_lst:
        if label.shape[0] < max_seq_len:
            post_pad = torch.full(size=(max_seq_len - label.shape[0],), fill_value=padding)
            # post_pad = torch.full(size=(max_seq_len - label.shape[0],), fill_value=-100)
            torch.concat([label, post_pad])
            padding_label_lst.append(torch.concat([label, post_pad]))
        else:
            padding_label_lst.append(label)

    labels = torch.stack(padding_label_lst)

    return {
        'inputs': inputs,
        'labels': labels,
        'sequence_len': torch.tensor(seq_len_lst)
    }

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    model_args, data_args, train_args = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    # os.environ['WANDB_PROJECT'] = model_args.wandb_project
    logger = get_logger(__name__)

    accelerator = (
        Accelerator(log_with=train_args.report_to, logging_dir=train_args.output_dir) if train_args.with_tracking else Accelerator()
    )

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

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if accelerator.is_main_process and train_args.output_dir is not None:
        os.makedirs(train_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    model_args.embedding = 1024
    # model_args.max_seq_length = 200

    model = SentenceLSTM(num_layers=train_args.num_layers,
                         embedding_size=model_args.embedding,
                         num_labels=data_args.num_labels,
                         drop_out_rate=train_args.drop_out_rate
                         )

    train_file = data_args.train_file
    eval_file = data_args.eval_file

    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)

    with open(eval_file, 'rb') as f:
        eval_data = pickle.load(f)

    # Shuffled Here
    train_dataset = EncoderSentenceClassificationDataset(train_data)
    eval_dataset = EncoderSentenceClassificationDataset(eval_data)

    for index in random.sample(range(len(train_dataset)), 5):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=False,
                                  collate_fn=partial(custom_collate, padding=train_args.padding),
                                  batch_size=train_args.per_device_train_batch_size,
                                  )
    eval_dataloader = DataLoader(eval_dataset,
                                 shuffle=False,
                                 collate_fn=partial(custom_collate, padding=train_args.padding),
                                 batch_size=train_args.per_device_eval_batch_size,
                                 )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": train_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=train_args.learning_rate,
                                  betas=(train_args.adam_beta1, train_args.adam_beta2),
                                  eps=train_args.adam_epsilon,
                                  )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.gradient_accumulation_steps)
    if train_args.max_train_steps is None:
        train_args.max_train_steps = train_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=train_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=train_args.num_warmup_steps,
        num_training_steps=train_args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        train_args.max_train_steps = train_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    train_args.num_train_epochs = math.ceil(train_args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = train_args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if train_args.with_tracking:
        experiment_config = vars(train_args)
        accelerator.init_trackers(train_args.wandb_project, config=experiment_config,
                                  init_kwargs={"wandb": {"name": train_args.run_name}})

    # Get the metric function
    metric_acc = evaluate.load("accuracy")
    metric_pre = evaluate.load('precision')
    metric_re = evaluate.load('recall')
    metric_f1 = evaluate.load('f1')

    # Train!
    total_batch_size = train_args.per_device_train_batch_size * accelerator.num_processes * train_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {train_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {train_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {train_args.max_train_steps}")

    # Saving model_args, data_args, train_args
    train_dict = vars(train_args)
    logger.info(f"  Saving training_args = {train_dict}")
    with open(os.path.join(train_args.output_dir, f"train_args.json"), "w") as f:
        json.dump(train_dict, f)

    model_dict = vars(model_args)
    logger.info(f"  Saving model_args = {model_dict}")
    with open(os.path.join(train_args.output_dir, f"model_args.json"), "w") as f:
        json.dump(model_dict, f)

    data_dict = vars(data_args)
    logger.info(f"  Saving data_args = {data_dict}")
    with open(os.path.join(train_args.output_dir, f"data_args.json"), "w") as f:
        json.dump(data_dict, f)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(train_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Using heap for limiting number of saved models
    model_heap = []
    heapq.heapify(model_heap)

    for epoch in range(starting_epoch, train_args.num_train_epochs):
        model.train()
        if train_args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            logits = model(batch['inputs'], batch['sequence_len'])
            criterion = torch.nn.CrossEntropyLoss(ignore_index=train_args.padding).cuda()
            loss = criterion(logits.view(-1, logits.shape[-1]), batch['labels'].view(-1))

            # We keep track of the loss at each epoch
            if train_args.with_tracking:
                cur_loss = loss.detach().float()
                total_loss += cur_loss

            loss = loss / train_args.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % train_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % train_args.train_loss_steps == 0 and step % train_args.gradient_accumulation_steps == 0:
                logger.info(f"Train loss {cur_loss} at current step  {completed_steps}")
                train_loss_log = {
                    "train_loss": cur_loss,
                    "step": completed_steps,
                }
                if train_args.with_tracking:
                    accelerator.log(
                        train_loss_log,
                        step=completed_steps,
                    )

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and step % train_args.gradient_accumulation_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if train_args.output_dir is not None:
                        output_dir = os.path.join(train_args.output_dir, output_dir)
                        os.makedirs(output_dir, exist_ok=True)
                    result_log, model_output_path = eval(model, eval_dataloader, accelerator,
                                                         metric_acc, metric_pre, metric_re, metric_f1,
                                                         train_args, epoch, completed_steps, output_dir,
                                                         logger)
                    accelerator.save_state(output_dir)

                    key_best_metric = f'eval_{train_args.best_metric}'
                    best_metric = result_log[key_best_metric]
                    logger.info(f"best_metric : {best_metric}")
                    heapq.heappush(model_heap, (best_metric, completed_steps, result_log, model_output_path))

                    if len(model_heap) > train_args.save_max_limit:
                        _, _, _, delete_path = heapq.heappop(model_heap)
                        logger.info(f"Deleting file for path : {delete_path}")
                        mydir = pathlib.Path(delete_path)
                        shutil.rmtree(mydir)
                    model.train()

            if completed_steps >= train_args.max_train_steps:
                break

        output_dir = f"epoch_{epoch}_step_{completed_steps}"
        if train_args.output_dir is not None:
            output_dir = os.path.join(train_args.output_dir, output_dir)
            os.makedirs(output_dir, exist_ok=True)

        result_log, model_output_path = eval(model, eval_dataloader, accelerator,
                                             metric_acc, metric_pre, metric_re, metric_f1,
                                             train_args, epoch, completed_steps, output_dir,
                                             logger)
        accelerator.save_state(output_dir)

        key_best_metric = f'eval_{train_args.best_metric}'
        best_metric = result_log[key_best_metric]
        logger.info(f"best_metric : {best_metric}")
        heapq.heappush(model_heap, (best_metric, completed_steps, result_log, model_output_path))

        if len(model_heap) > train_args.save_max_limit:
            _, _, _, delete_path = heapq.heappop(model_heap)
            logger.info(f"Deleting file for path : {delete_path}")
            mydir = pathlib.Path(delete_path)
            shutil.rmtree(mydir)

    if train_args.with_tracking:
        accelerator.end_training()

if __name__ == "__main__":
    main()