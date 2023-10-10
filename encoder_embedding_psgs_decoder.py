from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import heapq
import pathlib
import shutil
from FiD.src.model import FiDT5
from src.model import FiDEncoderForSequenceClassification
import argparse
import pickle
from tqdm.auto import tqdm

from pprint import pprint
from tqdm.auto import tqdm
from src.data import BinaryCustomDatasetShuffle

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
from torch.utils.data import DataLoader, SequentialSampler
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

from pprint import pprint
import numpy as np
import torch
from torch import nn
import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers import T5PreTrainedModel
import copy
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from FiD.src.model import FiDT5
import FiD.src.data


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)

def main():

    parser = argparse.ArgumentParser(description='Extracting encoder embeddings for passage-level decoder (GPT2)')

    parser.add_argument('--input_file_path', type=str,
                        default='/data/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1/ctx100id_split_train_1.json')
    parser.add_argument('--output_file_path', type=str,
                        default='/scratch/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1/')
    parser.add_argument('--batch_size', type=int,
                        default=2)
    parser.add_argument('--n_context', type=int,
                        default=5)
    parser.add_argument('--max_seq_length', type=int,
                        default=200)
    parser.add_argument('--model_name_or_path', type=str,
                        default='/data/philhoon-relevance/FiD/pretrained_models/nq_reader_large')

    args = parser.parse_args()

    model_class = FiDT5
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model_encoder = model.encoder.encoder
    tokenizer = AutoTokenizer.from_pretrained('t5-base', return_dict=False)

    eval_data = args.input_file_path
    eval_examples = FiD.src.data.load_data(
        eval_data,
    )

    n_context = args.n_context
    eval_dataset = FiD.src.data.DatasetWithID(
        eval_examples,
        n_context=n_context
    )

    text_maxlength = args.max_seq_length
    collator_function = FiD.src.data.CollatorWithID(text_maxlength, tokenizer, n_context)
    eval_sampler = SequentialSampler(eval_dataset)
    per_gpu_batch_size = args.batch_size
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=per_gpu_batch_size,
        num_workers=8,
        collate_fn=collator_function
    )

    model_encoder.to(device=device)
    model_encoder.eval()
    result_dict = []
    n_passages = args.n_context

    total_cnt = 1
    test_cnt = 0
    for index, id, _, _, passage_ids, passage_masks in tqdm(eval_dataloader):
        # print(index)
        # print(passage_ids.shape)
        # print(passage_masks.shape)

        input_ids = passage_ids.view(passage_ids.size(0), -1)
        attention_mask = passage_masks.view(passage_masks.size(0), -1)

        # print(input_ids.shape)
        # print(attention_mask.shape)

        bsz, total_length = input_ids.shape
        passage_length = total_length // n_passages
        # print(passage_length)

        input_ids = input_ids.view(bsz * n_passages, passage_length)
        # print(input_ids.shape)

        attention_mask = attention_mask.view(bsz * n_passages, passage_length)
        # print(attention_mask.shape)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model_encoder(input_ids, attention_mask)
        outputs = outputs[0].view(bsz, n_passages, passage_length, -1).detach().cpu()
        # outputs = outputs.detach().cpu()

        # print(output_by.shape)
        for i in range(bsz):
            id_ = id[i].item()
            embedding = outputs[i,]
            result_dict.append({
                'id' : id_,
                'embedding' : embedding
            })

        test_cnt += 1
        if test_cnt == 100:
            if 'train' in args.input_file_path:
                output_file_name = args.output_file_path + f'ctx100id_embedding_train_1_{total_cnt}.pickle'
            elif 'dev' in args.input_file_path:
                output_file_name = args.output_file_path + f'ctx100id_embedding_dev_1_{total_cnt}.pickle'
            else:
                output_file_name = args.output_file_path + f'ctx100id_embedding_test_1_{total_cnt}.pickle'


            with open(output_file_name, 'wb') as f:
                pickle.dump(result_dict, f)

            del result_dict
            result_dict = []

            print(f'{output_file_name} Saved')
            print(f'total_cnt : {total_cnt}')
            print(f'test_cnt : {test_cnt}')
            total_cnt += 1
            test_cnt = 0

    if len(result_dict) > 0:
        if 'train' in args.input_file_path:
            output_file_name = args.output_file_path + f'ctx100id_embedding_train_1_{total_cnt}.pickle'
        elif 'dev' in args.input_file_path:
            output_file_name = args.output_file_path + f'ctx100id_embedding_dev_1_{total_cnt}.pickle'
        else:
            output_file_name = args.output_file_path + f'ctx100id_embedding_test_1_{total_cnt}.pickle'

        with open(output_file_name, 'wb') as f:
            pickle.dump(result_dict, f)

if __name__ == "__main__":
    main()