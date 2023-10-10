from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import wandb
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
    EncoderSentenceClassificationDataset,
    DecoderSinlgeDataset,
    DecoderPositiveSinlgeDataset,
    DecoderCombinedSinlgeDataset,
    DecoderCombinedPositiveDataset,
    DecoderCombinedSinlgeFiveLabelDataset,
)

from functools import partial
import json
import math
import os
import logging
import re
import evaluate
from util import utils
from pprint import pformat

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
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, asdict
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
    'EncoderSentenceClassificationDataset' : EncoderSentenceClassificationDataset,
    'DecoderSinlgeDataset' : DecoderSinlgeDataset,
    'DecoderPositiveSinlgeDataset' : DecoderPositiveSinlgeDataset,
    'DecoderCombinedSinlgeDataset' : DecoderCombinedSinlgeDataset,
    'DecoderCombinedPositiveDataset' : DecoderCombinedPositiveDataset,
    'DecoderCombinedSinlgeFiveLabelDataset' : DecoderCombinedSinlgeFiveLabelDataset,
}
EMBEDDING_ARC_MAPPING = {
    "SentenceTransformer" : SentenceTransformer,
     "FiDT5" : FiDT5
}
LABEL_DICT = {
    'definite_pos': 4,
    'initial_zeros': 3,
    'semi-pos': 2,
    'semi-neg': 1,
    'definite_neg': 0
}

CONVERT_DICT = {
    '4': '1', # definite_pos
    '3': '0', # initial_zeros
    '2': '1', # semi-pos
    '1': '0', # semi-neg
    '0': '0' # definite_neg
}



def get_definite_pos_neg(test_em):
    positive_pos = []
    if test_em.startswith('1'):
        positive_pos.append(0)
    iter_ = re.finditer(r'01', test_em)
    for m in iter_:
        pos_ = m.start() + 1
        positive_pos.append(pos_)

    negative_pos = []
    iter_ = re.finditer(r'10', test_em)
    for m in iter_:
        pos_ = m.start() + 1
        negative_pos.append(pos_)

    return positive_pos, negative_pos


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout,
                                                                 is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class SentenceGPTConfig:
    block_size: int = 100 # block_size represents number_passages
    token_length: int = 200 # represents the max_token_length
    n_embd: int = 1024 # used to be 768, switch to 1024, switch to FiD encoder embedding
    num_labels: int = 2 # labels will be either 0 or 1 (EM_pattern)
    n_layer: int = 6
    n_head: int = 16 # n_embd % n_head == 0
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


# Implemented
class SentenceGPT(nn.Module):

    # Compatibility Checked
    def __init__(self, config):
        super().__init__()
        # No need for vocab_size
        # assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # No NEED for token & positional encoding weights
        # n_embd, block_size, n_embde
        self.transformer = nn.ModuleDict(dict(
            # wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            embedding_layer=nn.Linear(config.token_length * config.n_embd, config.n_embd, bias=config.bias),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Instead using vocab_size use self.num_labels
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.num_labels, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate

        # No need for token embedding weights
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    # Compatibility Checked
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())

        # No need for positional embedding weights
        # if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    # Compatibility Checked
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Compatibility Checked
    def forward(self, idx, targets=None):
        # Here idx is embedding
        # idx -> batch, block_size(num_passages), token_length, n_embd
        # device = idx.device
        # b, t = idx.size()
        # device = idx.device # This is for creating positinal embedding
        b, t, _, _ = idx.size()

        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # No need for tok_emb, pos_emb
        # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        # forward the GPT model itself
        # tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        # x = self.transformer.drop(tok_emb + pos_emb)

        x = idx.view(b, t, -1)
        x = self.transformer.embedding_layer(x)
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)

        ## output (batch, num_passages, num_labels)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            # loss = None

            # inference time: going to foward on every position
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    # No need for Compatibility Check
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        # No need for position encoding weights
        # self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # No need for Compatibility Check
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # Need for Compatibility Check
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        # since we SentenceGPT does not use token embedding...
        # decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    # No need for Compatibility Check
    # SenetenceGPT won't use generate function -> Ignore
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def inference(self, idx):
        logits, _ = self(idx)
        return logits


class DecoderSinlgeDataset(Dataset):
    def __init__(self, path, ref_file, n_context):
        self.files = [path + '/' + file for file in os.listdir(path)]
        self.n_context = n_context
        self.ref_file = ref_file

        # init self.ids
        #         self._get_ids()

        # init self.target
        self._get_target()

    #     def _get_ids(self):
    #         self.ids = []
    #         for file in self.files:
    #             id_ = str(file).split('/')[-1]
    #             id_ = id_.split('.')[0]
    #             self.ids.append(int(id_))

    def _get_target(self):
        self.target = {}
        ref_data = utils.open_json(self.ref_file)
        for ins in ref_data:
            self.target[ins['id']] = ins['em_pattern']

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = self.files[index]
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        id_ = data['id']
        embedding_ = data['embedding'][:self.n_context, :, :]
        em_pattern_ = self.target[int(id_)][:self.n_context]

        return {
            'id': id_,
            'embedding': embedding_,
            'em_pattern': em_pattern_,
        }


def custom_collate_decoder(batch):
    # id_lst for later matching
    # embedding -> turn into tensor
    # em_pattern -> turn into tensor
    id_lst = []
    embeddings = []
    em_patterns = []
    for b in batch:
        id_lst.append(b['id'])
        embeddings.append(b['embedding'])
        em_patterns.append(torch.tensor(list(map(float, map(int, b['em_pattern']))), dtype=torch.long))

    embeddings = torch.stack(embeddings)
    em_patterns = torch.stack(em_patterns)

    return {
        'ids': id_lst,
        'embeddings': embeddings,
        'em_patterns': em_patterns
    }


###### From Here Main Script ######


def eval(model, eval_dataloader, accelerator, metric_acc,
         metric_pre, metric_re, metric_f1, train_args, epoch, steps, output_dir, logger, num_labels):

    eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
    eval_completed_steps = 0

    eval_loss = 0
    model.eval()
    samples_seen = 0
    prediction_lst = []
    reference_lst = []

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            logits, loss = model(batch['embeddings'], batch['em_patterns'])
        if train_args.with_tracking:
            eval_loss += loss.detach().float()

        predictions = logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["em_patterns"]))
        batch_size, context_size = predictions.size()

        predictions = predictions.reshape(-1)
        references = references.reshape(-1)

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

        predictions = predictions.reshape(batch_size, context_size)
        references = references.reshape(batch_size, context_size)

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

    ##### Rev #####
    if num_labels <= 2:
        prediction_np = np.array(prediction_lst).reshape(-1)
        reference_np = np.array(reference_lst).reshape(-1)
        y_actu = pd.Series(reference_np, name='Actual')
        y_pred = pd.Series(prediction_np, name='Predicted')

        reversey_pred = y_pred.map(lambda x: 0 if x == 1 else 1)
        reversey_actu = y_actu.map(lambda x: 0 if x == 1 else 1)
        rev_accuracy = accuracy_score(reversey_actu, reversey_pred)
        rev_precision = precision_score(reversey_actu, reversey_pred)
        rev_recall = recall_score(reversey_actu, reversey_pred)
        rev_f1 = f1_score(reversey_actu, reversey_pred)

        logger.info(f"rev_Accuracy : {rev_accuracy} rev_Precision : {rev_precision}")
        logger.info(f"rev_Recall : {rev_recall} rev_F1 : {rev_f1}")

        result_rev_log = {
            "eval_rev_accuracy": rev_accuracy,
            "eval_rev_precision": rev_precision,
            "eval_rev_recall": rev_recall,
            "eval_rev_f1": rev_f1,
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

        def_pos_ref = []
        def_pos_pre = []
        def_neg_ref = []
        def_neg_pre = []

        def_pos_reference_lst = [''.join([str(ref_) for ref_ in em]) for em in reference_lst]
        for em_str, ref, pred in zip(def_pos_reference_lst, reference_lst, prediction_lst):
            pos_ind_lst, neg_ind_lst = get_definite_pos_neg(em_str)
            if pos_ind_lst:
                for pos_ind in pos_ind_lst:
                    def_pos_ref.append(ref[pos_ind])
                    def_pos_pre.append(pred[pos_ind])
            if neg_ind_lst:
                for neg_ind in neg_ind_lst:
                    def_neg_ref.append(ref[neg_ind])
                    def_neg_pre.append(pred[neg_ind])

        accuracy_def_pos = accuracy_score(def_pos_ref, def_pos_pre)
        precision_def_pos = precision_score(def_pos_ref, def_pos_pre)
        recall_def_pos = recall_score(def_pos_ref, def_pos_pre)
        f1_def_pos = f1_score(def_pos_ref, def_pos_pre)

        def_pos_result_log = {
            "def_pos_acc": accuracy_def_pos,
            "def_pos_pre": precision_def_pos,
            "def_pos_rec": recall_def_pos,
            "def_pos_f1": f1_def_pos,
            "epoch": epoch,
            "step": steps,
        }

        output_result_path = os.path.join(output_dir, f"epoch{epoch}_steps{steps}_def_pos_results.json")
        with open(output_result_path, "w") as f:
            json.dump(def_pos_result_log, f)

        if train_args.with_tracking:
            accelerator.log(
                def_pos_result_log,
                step=steps,
            )

        def_neg_ref = list(map(lambda x: 0 if x == 1 else 1, def_neg_ref))
        def_neg_pre = list(map(lambda x: 0 if x == 1 else 1, def_neg_pre))

        accuracy_def_neg = accuracy_score(def_neg_ref, def_neg_pre)
        precision_def_neg = precision_score(def_neg_ref, def_neg_pre)
        recall_def_neg = recall_score(def_neg_ref, def_neg_pre)
        f1_def_neg = f1_score(def_neg_ref, def_neg_pre)

        def_neg_result_log = {
            "def_neg_acc": accuracy_def_neg,
            "def_neg_pre": precision_def_neg,
            "def_neg_rec": recall_def_neg,
            "def_neg_f1": f1_def_neg,
            "epoch": epoch,
            "step": steps,
        }

        output_result_path = os.path.join(output_dir, f"epoch{epoch}_steps{steps}_def_neg_results.json")
        with open(output_result_path, "w") as f:
            json.dump(def_neg_result_log, f)

        if train_args.with_tracking:
            accelerator.log(
                def_neg_result_log,
                step=steps,
            )

    # num_labels > 2
    else:
        def_pos_ref = []
        def_pos_pre = []
        def_neg_ref = []
        def_neg_pre = []

        def_pos_reference_lst = [''.join([str(ref_) for ref_ in em]) for em in reference_lst]
        for em_str, ref, pred in zip(def_pos_reference_lst, reference_lst, prediction_lst):
            converted_em_str = ''
            for x in em_str:
                converted_em_str += CONVERT_DICT[x]

            pos_ind_lst, neg_ind_lst = get_definite_pos_neg(converted_em_str)
            if pos_ind_lst:
                for pos_ind in pos_ind_lst:
                    def_pos_ref.append(1)

                    if pred[pos_ind] == 4 or pred[pos_ind] == '4':
                        def_pos_pre.append(1)
                    else:
                        def_pos_pre.append(0)

            if neg_ind_lst:
                for neg_ind in neg_ind_lst:
                    def_neg_ref.append(0)

                    if pred[neg_ind] == 0 or pred[neg_ind] == '0':
                        def_neg_pre.append(0)
                    else:
                        def_neg_pre.append(1)

        accuracy_def_pos = accuracy_score(def_pos_ref, def_pos_pre)
        precision_def_pos = precision_score(def_pos_ref, def_pos_pre)
        recall_def_pos = recall_score(def_pos_ref, def_pos_pre)
        f1_def_pos = f1_score(def_pos_ref, def_pos_pre)

        def_pos_result_log = {
            "def_pos_acc": accuracy_def_pos,
            "def_pos_pre": precision_def_pos,
            "def_pos_rec": recall_def_pos,
            "def_pos_f1": f1_def_pos,
            "epoch": epoch,
            "step": steps,
        }

        output_result_path = os.path.join(output_dir, f"epoch{epoch}_steps{steps}_def_pos_results.json")
        with open(output_result_path, "w") as f:
            json.dump(def_pos_result_log, f)

        if train_args.with_tracking:
            accelerator.log(
                def_pos_result_log,
                step=steps,
            )

        def_neg_ref = list(map(lambda x: 0 if x == 1 else 1, def_neg_ref))
        def_neg_pre = list(map(lambda x: 0 if x == 1 else 1, def_neg_pre))

        accuracy_def_neg = accuracy_score(def_neg_ref, def_neg_pre)
        precision_def_neg = precision_score(def_neg_ref, def_neg_pre)
        recall_def_neg = recall_score(def_neg_ref, def_neg_pre)
        f1_def_neg = f1_score(def_neg_ref, def_neg_pre)

        def_neg_result_log = {
            "def_neg_acc": accuracy_def_neg,
            "def_neg_pre": precision_def_neg,
            "def_neg_rec": recall_def_neg,
            "def_neg_f1": f1_def_neg,
            "epoch": epoch,
            "step": steps,
        }

        output_result_path = os.path.join(output_dir, f"epoch{epoch}_steps{steps}_def_neg_results.json")
        with open(output_result_path, "w") as f:
            json.dump(def_neg_result_log, f)

        if train_args.with_tracking:
            accelerator.log(
                def_neg_result_log,
                step=steps,
            )


    return result_log, output_dir


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    model_args, data_args, train_args = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    model_args.num_labels = data_args.num_labels

    model_args.config_base_path = None
    model_args.config_name = None
    model_args.git_tag = None
    model_args.max_seq_length = None
    # model_args.block_size = 20
    model_args.token_length = 200
    model_args.n_embd = 1024
    # if data_args.dataset_class == "DecoderCombinedSinlgeFiveLabelDataset":
    #     model_args.num_labels = 5
    # else:
    #     model_args.num_labels = 2
    # model_args.n_layer = 6
    model_args.n_head = 16
    model_args.dropout = 0.1
    model_args.bias = True
    model_args.model_architecture = 'gpt2'
    model_args.model_name_or_path = None
    model_args.prediction_model_name_or_path = None
    model_args.prediction_model_step = None
    model_args.tokenizer_name = None

    data_args.data = 'decoder-classification/NQ-DEV-DPR/5-fold/1'
    # data_args.dataset_class = 'DecoderSinlgeDataset'
    data_args.train_file = '/scratch/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1/embedding/train'
    data_args.eval_file = '/scratch/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1/embedding/dev'
    data_args.ref_train = '/data/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1/ctx100id_split_train_1.json'
    data_args.ref_eval = '/data/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1/ctx100id_split_dev_1.json'

    data_args.train_file2 = '/scratch/philhoon-relevance/decoder-classification/TQA-DEV-DPR/5-fold/1/embedding/train'
    data_args.eval_file2 = '/scratch/philhoon-relevance/decoder-classification/TQA-DEV-DPR/5-fold/1/embedding/dev'
    data_args.ref_train2 = '/scratch/philhoon-relevance/decoder-classification/TQA-DEV-DPR/5-fold/1/ctx100id_split_train_1.json'
    data_args.ref_eval2 = '/scratch/philhoon-relevance/decoder-classification/TQA-DEV-DPR/5-fold/1/ctx100id_split_dev_1.json'

    data_args.intact_eval = None
    data_args.num_labels = None
    data_args.overwrite_cache = None
    data_args.pad_to_max_length = None

    # train_args.report_to = 'wandb'
    # train_args.wandb_project = 'decoder-sequential-classifier'
    # train_args.with_tracking = True
    # train_args.run_name = 'sequential-decoder-classifier-testing'
    # train_args.output_dir = '/scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/decoder-sequentail-classifier-testing'

    train_args.seed = 42
    train_args.best_metric = 'f1'
    train_args.num_train_epochs = 5
    train_args.train_loss_steps = 10
    train_args.checkpointing_steps = '50'
    train_args.save_max_limit = 10
    # train_args.gradient_accumulation_steps = 1
    # train_args.per_device_eval_batch_size=32
    # train_args.per_device_train_batch_size=32

    # train_args.learning_rate = 6e-4  # 6e-5 :minimum learning rate, 5e-5 ,6e-4 : max learning rate
    train_args.lr_scheduler_type = 'linear'
    train_args.adam_beta1 = 0.9
    train_args.adam_beta2 = 0.95
    train_args.weight_decay = 1e-1
    train_args.num_warmup_steps = 100

    train_args.num_layers = None
    train_args.adam_epsilon = None
    train_args.max_train_steps = None
    train_args.padding = None
    train_args.headtype = None
    train_args.class_weights = None
    train_args.drop_out_rate = None
    train_args.do_eval = None
    train_args.do_predict = None
    train_args.do_train = None


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

    config_gpt = SentenceGPTConfig(
        block_size=model_args.block_size,
        token_length=model_args.token_length,
        n_embd=model_args.n_embd,
        num_labels=model_args.num_labels,
        n_layer=model_args.n_layer,
        n_head=model_args.n_head,
        dropout=model_args.dropout,
        bias=model_args.bias
    )
    model = SentenceGPT(config_gpt)

    if data_args.dataset_class == 'DecoderCombinedSinlgeDataset' or data_args.dataset_class == "DecoderCombinedPositiveDataset" \
        or  data_args.dataset_class == "DecoderCombinedSinlgeFiveLabelDataset":
        path_train = data_args.train_file # '/scratch/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1/embedding/train'
        path_train2 = data_args.train_file2 # '/scratch/philhoon-relevance/decoder-classification/TQA-DEV-DPR/5-fold/1/embedding/train'
        ref_train = data_args.ref_train # '/data/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1/ctx100id_split_train_1.json'
        ref_train2 = data_args.ref_train2 # '/scratch/philhoon-relevance/decoder-classification/TQA-DEV-DPR/5-fold/1/ctx100id_split_train_1.json'

        path_dev = data_args.eval_file # '/scratch/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1/embedding/dev'
        path_dev2 = data_args.eval_file2 # '/scratch/philhoon-relevance/decoder-classification/TQA-DEV-DPR/5-fold/1/embedding/dev'
        ref_dev = data_args.ref_eval # '/data/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1/ctx100id_split_dev_1.json'
        ref_dev2 = data_args.ref_eval2 # '/scratch/philhoon-relevance/decoder-classification/TQA-DEV-DPR/5-fold/1/ctx100id_split_dev_1.json'

    else:
        ref_train = data_args.ref_train
        ref_dev = data_args.ref_eval

        path_train = data_args.train_file
        path_dev = data_args.eval_file

    n_context = model_args.block_size
    DataSetClass = DATASET_MAPPING[data_args.dataset_class]


    if data_args.dataset_class == 'DecoderCombinedSinlgeDataset' or data_args.dataset_class == "DecoderCombinedPositiveDataset" or \
        data_args.dataset_class == 'DecoderCombinedSinlgeFiveLabelDataset':
        train_dataset = DataSetClass(path_train, path_train2, ref_train, ref_train2, n_context)
        dev_dataset = DataSetClass(path_dev, path_dev2, ref_dev, ref_dev2, n_context)
    else:
        train_dataset = DataSetClass(path_train, ref_train, n_context)
        dev_dataset = DataSetClass(path_dev, ref_dev, n_context)

    for index in random.sample(range(len(train_dataset)), 5):
        logger.info(f"Sample {index} of the training index: {index}.")
        logger.info(f"Sample {index} of the training set id: {train_dataset[index]['id']}.")
        logger.info(f"Sample {index} of the training set embedding shape: {train_dataset[index]['embedding'].shape}.")
        logger.info(f"Sample {index} of the training set em pattern: {train_dataset[index]['em_pattern']}.")

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  collate_fn=custom_collate_decoder,
                                  batch_size=train_args.per_device_train_batch_size,
                                  num_workers=train_args.train_num_workers
                                  )

    eval_dataloader = DataLoader(dev_dataset,
                                 shuffle=False,
                                 collate_fn=custom_collate_decoder,
                                 batch_size=train_args.per_device_eval_batch_size,
                                 num_workers=train_args.eval_num_workers
                                 )

    # optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    optimizer = model.configure_optimizers(
        train_args.weight_decay,
        train_args.learning_rate,
        (train_args.adam_beta1, train_args.adam_beta2),
        device_type=None)

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

        init_kwargs = {"wandb": {"name": train_args.run_name,
                                 "settings": {"console": "off"}}}
        accelerator.init_trackers(train_args.wandb_project, config=experiment_config,
                                  init_kwargs=init_kwargs)

        # accelerator.init_trackers(train_args.wandb_project, config=experiment_config,
        #                           init_kwargs={"wandb": {"name": train_args.run_name}})

    # Get the metric function
    if model_args.num_labels > 2:
        metric_acc = evaluate.load("accuracy")
        metric_pre = evaluate.load('precision', average='micro')
        metric_re = evaluate.load('recall', average='micro')
        metric_f1 = evaluate.load('f1', average='micro')
    else:
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
    logger.info(f"  Saving training_args = {pformat(train_dict)}")
    with open(os.path.join(train_args.output_dir, f"train_args.json"), "w") as f:
        json.dump(train_dict, f)

    model_dict = vars(model_args)
    logger.info(f"  Saving model_args = {pformat(model_dict)}")
    with open(os.path.join(train_args.output_dir, f"model_args.json"), "w") as f:
        json.dump(model_dict, f)

    data_dict = vars(data_args)
    logger.info(f"  Saving data_args = {pformat(data_dict)}")
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
            ids_ = batch['ids']
            logits, loss = model(batch['embeddings'], batch['em_patterns'])

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
                         metric_pre, metric_re, metric_f1, train_args, epoch, completed_steps, output_dir, logger, model_args.num_labels)
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



    if train_args.output_dir is not None:
        output_dir = f"fianl_step_{completed_steps}"
        output_dir = os.path.join(train_args.output_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        result_log, model_output_path = eval(model, eval_dataloader, accelerator, metric_acc,
                                             metric_pre, metric_re, metric_f1, train_args, epoch, completed_steps,
                                             output_dir, logger)
        accelerator.save_state(output_dir)

    if train_args.with_tracking:
        accelerator.end_training()
        accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(model)
        # unwrapped_model.save_pretrained(
        #     train_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        # )

if __name__ == "__main__":
    main()