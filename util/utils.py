import json
import torch
import numpy as np
import random
from copy import deepcopy

def open_json(file):
    with open(file , 'r') as f: 
        data = json.load(f)
    return data

def save_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp, ensure_ascii=True)
        
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

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

def prepare_sequential_data(data):
    output = []
    for instance in data:
        question_ = instance['question']
        ctxs_ = instance['ctxs']
        em_pattern_ = instance['em_pattern']
        id_ = instance['id']

        ctx_lst = []
        for context in ctxs_:
            input_ = 'question: ' + question_ + ', '\
                     ' title: ' + context['title'] + ', '\
                     ' context : ' + context['text']
            ctx_lst.append(input_)

        template = {
            'id' : id_,
            'em_pattern' : em_pattern_,
            'ctx' : ctx_lst

        }
        output.append(template)
    return output

def prepare_sequential_decisive_data(data):
    output = []
    for instance in data:
        question_ = instance['question']
        decisive_ctxs_ = instance['decisive_ctxs']
        decisive_em_pattern_ = instance['decisive_em_pattern']
        id_ = instance['id']

        ctx_lst = []
        for decisive_ctxs in decisive_ctxs_:
            input_ = 'question: ' + question_ + ', '\
                     ' title: ' + decisive_ctxs['title'] + ', '\
                     ' context : ' + decisive_ctxs['text']
            ctx_lst.append(input_)

        template = {
            'id' : id_,
            'em_pattern' : decisive_em_pattern_,
            'ctx' : ctx_lst

        }
        output.append(template)
    return output