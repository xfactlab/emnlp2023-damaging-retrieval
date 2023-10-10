from pprint import pprint
from util import utils

import json
import math
import os
import logging
import sys
import evaluate
from util import utils
from tqdm.auto import tqdm
import pickle

import argparse
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
    Trainer,
    TrainingArguments,
    set_seed,
    get_scheduler,
)
from util.arguments import ModelArguments, DataTrainingArguments
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from FiD.src.model import FiDT5
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.data import BinaryCustomDatasetShuffle

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)

def main():
    parser = argparse.ArgumentParser(description='sentence_encoder_prepare')

    parser.add_argument('--input_file_path', type=str, required=True)
    parser.add_argument('--output_file_path', type=str, required=True)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--max_seq_length', type=int, default=200)
    parser.add_argument('--model_name_or_path', type=str, required=True)

    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
    embedding_model = FiDT5.from_pretrained(args.model_name_or_path)
    fid_encoder = embedding_model.encoder.encoder
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    testing_data = utils.open_json(args.input_file_path)

    if 'decisive' in args.input_file_path:
        print(f'decisive data')
        seq_train_data = utils.prepare_sequential_decisive_data(testing_data)
    else:
        print(f'sequential data')
        seq_train_data = utils.prepare_sequential_data(testing_data)

    fid_encoder.to(device=device)
    fid_encoder.eval()
    new_instance = []

    for instance in tqdm(seq_train_data):
        input_ = tokenizer(instance['ctx'],
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           add_special_tokens=True,
                           max_length=args.max_seq_length)
        input_ids = input_['input_ids'].to(device)
        attention_mask = input_['attention_mask'].to(device)
        with torch.no_grad():
            embedding = fid_encoder(input_ids, attention_mask)
            token_embeddings = embedding.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

        mean_pooled_cpu = mean_pooled.detach().cpu()
        em_pattern_ = torch.tensor([int(i) for i in instance['em_pattern']])

        result = {
            'input_embedding': mean_pooled_cpu,
            'em_pattern': em_pattern_
        }
        new_instance.append(result)

    with open(args.output_file_path, 'wb') as f:
        pickle.dump(new_instance, f)


if __name__ == "__main__":
    main()