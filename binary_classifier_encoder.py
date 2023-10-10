from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import heapq
import pathlib
import shutil
from FiD.src.model import FiDT5
from src.model import FiDEncoderForSequenceClassification

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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATASET_MAPPING = {
    "BinaryCustomDatasetShuffle" : BinaryCustomDatasetShuffle,
    "BinarySentenceDataset" : BinarySentenceDataset,
    'BinaryCustomDatasetDecisiveBinaryGold' : BinaryCustomDatasetDecisiveBinaryGold,
    'BinaryCustomDatasetPredictionShuffle' : BinaryCustomDatasetPredictionShuffle
}

def eval(model, eval_dataloader, accelerator, metric_acc,
         metric_pre, metric_re, metric_f1, train_args, epoch, steps, output_dir, logger):

    eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
    eval_completed_steps = 0

    eval_loss = 0
    model.eval()
    samples_seen = 0
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

    return result_log, output_dir


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

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=data_args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model_class = FiDT5
    fid_model = model_class.from_pretrained(model_args.model_name_or_path)
    fid_encoder = fid_model.encoder.encoder
    model = FiDEncoderForSequenceClassification(config, fid_encoder, pooler = train_args.headtype)

    train_file = data_args.train_file
    eval_file = data_args.eval_file

    train_data = utils.open_json(train_file)
    eval_data = utils.open_json(eval_file)

    DataSetClass = DATASET_MAPPING[data_args.dataset_class]

    train_dataset = DataSetClass(train_data, tokenizer=tokenizer,
                                 max_length=model_args.max_seq_length, shuffle=False)

    eval_dataset = DataSetClass(eval_data, tokenizer=tokenizer,
                                max_length=model_args.max_seq_length, shuffle=False)

    for index in random.sample(range(len(train_dataset)), 5):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  collate_fn=data_collator,
                                  batch_size=train_args.per_device_train_batch_size,
                                  )

    eval_dataloader = DataLoader(eval_dataset,
                                  shuffle = True,
                                  collate_fn=data_collator,
                                  batch_size=train_args.per_device_eval_batch_size,
    )

    no_decay = ["bias", "layer_norm.weight"]
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
            outputs = model(**batch)
            loss = outputs.loss
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
                    result_log, model_output_path = eval(model, eval_dataloader, accelerator, metric_acc,
                         metric_pre, metric_re, metric_f1, train_args, epoch, completed_steps, output_dir, logger)
                    accelerator.save_state(output_dir)

                    key_best_metric = f'eval_{train_args.best_metric}'
                    best_metric = result_log[key_best_metric]
                    logger.info(f"best_metric : {best_metric}")
                    heapq.heappush(model_heap, (best_metric, completed_steps, result_log, model_output_path))

                    if len(model_heap) > train_args.save_max_limit:
                        _, _, _ ,delete_path = heapq.heappop(model_heap)
                        logger.info(f"Deleting file for path : {delete_path}")
                        mydir = pathlib.Path(delete_path)
                        shutil.rmtree(mydir)
                    model.train()

            if completed_steps >= train_args.max_train_steps:
                break

        model.eval()
        samples_seen = 0
        eval_loss = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

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

        eval_metric = metric_acc.compute()
        eval_metric_pre = metric_pre.compute()
        eval_metric_re = metric_re.compute()
        eval_metric_f1 = metric_f1.compute()

        logger.info(f"Evaluation at Epoch : {epoch} ")
        logger.info(f"Accuracy : {eval_metric['accuracy']} Precision : {eval_metric_pre['precision']}")
        logger.info(f"Recall : {eval_metric_re['recall']} F1 : {eval_metric_f1['f1']}")
        logger.info(f"Total_Train_loss : {total_loss.item() / len(train_dataloader)}")
        logger.info(f"Eval_loss : {eval_loss.item() / len(eval_dataloader)}")

        result_log = {
            "eval_accuracy": eval_metric['accuracy'],
            "eval_precision": eval_metric_pre['precision'],
            "eval_recall": eval_metric_re['recall'],
            "eval_f1": eval_metric_f1['f1'],
            "total_train_loss": total_loss.item() / len(train_dataloader),
            "eval_loss": eval_loss.item() / len(eval_dataloader),
            "epoch": epoch,
            "step": completed_steps,
        }

        if train_args.with_tracking:
            accelerator.log(
                result_log,
                step=completed_steps,
            )

        if train_args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if train_args.output_dir is not None:
                output_dir = os.path.join(train_args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if train_args.with_tracking:
        accelerator.end_training()

    if train_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            train_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(train_args.output_dir)

    # Final evaluation
    # eval_dataset = BinaryCustomDatasetShuffle(eval_data, tokenizer=tokenizer, \
    #                                           max_length=model_args.max_seq_length, shuffle=False)
    # eval_dataloader = DataLoader(
    #     eval_dataset, collate_fn=data_collator, batch_size=train_args.per_device_eval_batch_size
    # )
    # eval_dataloader = accelerator.prepare(eval_dataloader)
    #
    # model.eval()
    # for step, batch in enumerate(eval_dataloader):
    #     outputs = model(**batch)
    #     predictions = outputs.logits.argmax(dim=-1)
    #
    #     metric_acc.add_batch(
    #         predictions=accelerator.gather(predictions),
    #         references=accelerator.gather(batch["labels"]),
    #     )
    #     metric_pre.add_batch(
    #         predictions=accelerator.gather(predictions),
    #         references=accelerator.gather(batch["labels"]),
    #     )
    #
    #     metric_re.add_batch(
    #         predictions=accelerator.gather(predictions),
    #         references=accelerator.gather(batch["labels"]),
    #     )
    #
    #     metric_f1.add_batch(
    #         predictions=accelerator.gather(predictions),
    #         references=accelerator.gather(batch["labels"]),
    #     )
    #
    # eval_metric = metric_acc.compute()
    # eval_metric_pre = metric_pre.compute()
    # eval_metric_re = metric_re.compute()
    # eval_metric_f1 = metric_f1.compute()
    #
    # logger.info(f"Evaluation at Epoch : {epoch} ")
    # logger.info(f"Accuracy : {eval_metric['accuracy']} Precision : {eval_metric_pre['precision']}")
    # logger.info(f"Recall : {eval_metric_re['recall']} F1 : {eval_metric_f1['f1']}")
    # logger.info(f"Total_Train_loss : {total_loss.item() / len(train_dataloader)}")
    # logger.info(f"Eval_loss : {eval_loss.item() / len(eval_dataloader)}")
    #
    # logger.info(
    #     f"Final\n: Accuracy : {eval_metric['accuracy']}, Precision : {eval_metric_pre['precision']} Recall : {eval_metric_re['recall']}  F1 : {eval_metric_f1['f1']} ")
    #
    # all_results = {
    #     "eval_accuracy": eval_metric['accuracy'],
    #     "eval_precision": eval_metric_pre['precision'],
    #     "eval_recall": eval_metric_re['recall'],
    #     "eval_f1": eval_metric_f1['f1'],
    #     "eval_loss": eval_loss.item() / len(eval_dataloader),
    #     "epoch": train_args.num_train_epochs,
    #     "step": train_args.max_train_steps,
    # }
    #
    # if train_args.output_dir is not None:
    #     with open(os.path.join(train_args.output_dir, f"Final_results_epoch{train_args.num_train_epochs}_steps{train_args.max_train_steps}.json"), "w") as f:
    #         json.dump(all_results, f)

if __name__ == "__main__":
    main()