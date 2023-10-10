import torch
import random
import os
from util import utils
import pickle
import re

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


def get_semi_pos(test_em):
    semi_pos = []
    iter_ = re.finditer(r'(?=(11))', test_em)
    for m in iter_:
        semi_pos_ = m.start() + 1
        semi_pos.append(semi_pos_)

    return semi_pos

def get_semi_neg(test_em, num_undecisive):
    semi_neg = []
    test_em_temp = test_em[num_undecisive:]
    iter_ = re.finditer(r'(?=(00))', test_em_temp)
    for m in iter_:
        semi_neg_ = m.start() + 1
        semi_neg.append(semi_neg_)
    semi_neg = [i + num_undecisive for i in semi_neg]
    return semi_neg


def get_new_label(em_pattern, n_context):
    def_pos, def_neg, semi_pos, ini_zeros, semi_neg = (None,) * 5

    em_pattern = em_pattern[:n_context]

    def_pos, def_neg = get_definite_pos_neg(em_pattern)
    semi_pos = get_semi_pos(em_pattern)
    num_undecisive = len(em_pattern) - len(em_pattern.lstrip('0'))
    ini_zeros = [_ for _ in range(0, num_undecisive)]
    semi_neg = get_semi_neg(em_pattern, num_undecisive)

    label_em = [None] * n_context

    for indice in def_pos:
        label_em[indice] = 4

    for indice in ini_zeros:
        label_em[indice] = 3

    for indice in semi_pos:
        label_em[indice] = 2

    for indice in semi_neg:
        label_em[indice] = 1

    for indice in def_neg:
        label_em[indice] = 0

    if any(x is None for x in label_em):
        raise ValueError("There is None in new_label.")

    new_em_pattern = ''.join([str(x) for x in label_em])
    return new_em_pattern

class BinarySentenceDataset(torch.utils.data.Dataset):
    def __init__(self, instances, tokenizer, max_length, shuffle=False):
        if shuffle:
            random.shuffle(instances)
        self.instances = instances
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.max_length = max_length

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        text1_ = 'question: ' + self.instances[idx]['question']

        text2_ = 'title: ' + self.instances[idx]['ctx']['title'] + \
                 ' context : ' + self.instances[idx]['ctx']['text']
        output = self.tokenizer(
            text1_, text2_,
            # return_tensors="pt", will be applied later through collator
            # padding=True, will be padded later through collate
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length)

        item = {key: val for key, val in output.items()}
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
        input_ = 'question: ' + self.instances[idx]['question'] + ', '\
                 ' title: ' + self.instances[idx]['ctx']['title'] + ', '\
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

# Same as BinaryCustomDatasetShuffle
# self.instances[idx]['em'] -> self.instances[idx]['binary_inference']
class BinaryCustomDatasetPredictionShuffle(torch.utils.data.Dataset):
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
        input_ = 'question: ' + self.instances[idx]['question'] + ', '\
                 ' title: ' + self.instances[idx]['ctx']['title'] + ', '\
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
        item['labels'] = torch.tensor(int(self.instances[idx]['binary_inference']))

        return item


class BinaryCustomDatasetDecisiveBinaryGold(torch.utils.data.Dataset):
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
        input_ = 'question: ' + self.instances[idx]['question'] + ', '\
            ' answer: ' + self.instances[idx]['gold'] + ', '\
            ' title: ' + self.instances[idx]['ctx']['title'] + ', '\
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


class SentenceClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, instances, model, shuffle=True):
        if shuffle:
            random.shuffle(instances)
        self.instances = instances
        self.model = model

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        input_ = torch.from_numpy(self.model.encode(self.instances[idx]['ctx'], show_progress_bar=False))
        em_pattern_ = torch.tensor([int(i) for i in self.instances[idx]['em_pattern']])

        result = {
            'input_embedding': input_,
            'em_pattern': em_pattern_
        }

        return result


class EncoderSentenceClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, instances, shuffle=True):
        if shuffle:
            random.shuffle(instances)
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        input_embedding_ = self.instances[idx]['input_embedding']
        em_pattern_ = self.instances[idx]['em_pattern']

        result = {
            'input_embedding': input_embedding_,
            'em_pattern': em_pattern_
        }

        return result

class BinaryCustomDatasetDecisiveBinaryInference(torch.utils.data.Dataset):
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
        input_ = 'question: ' + self.instances[idx]['question'] + ', '\
            ' inference: ' + self.instances[idx]['inference'] + ', '\
            ' title: ' + self.instances[idx]['ctx']['title'] + ', '\
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


class DecoderSinlgeDataset(torch.utils.data.Dataset):
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


class DecoderPositiveSinlgeDataset(torch.utils.data.Dataset):
    def __init__(self, path, ref_file, n_context):
        self.path = path
        self.ref_data = utils.open_json(ref_file)
        self.n_context = n_context
        self._get_ids()
        self._get_files()

    def _get_files(self):
        self.files = []
        candidate_files = set(os.listdir(self.path))
        for id_ in self.ids_lst:
            temp_file = f'{id_}.pickle'
            if temp_file in candidate_files:
                temp_path = self.path + '/' + temp_file
                self.files.append(temp_path)

    def _get_ids(self):
        self.ids_lst = []
        self.target = {}
        for ref in self.ref_data:
            if '1' in ref['em_pattern'][:self.n_context]:
                self.ids_lst.append(ref['id'])
                self.target[ref['id']] = ref['em_pattern']

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


class DecoderCombinedSinlgeDataset(torch.utils.data.Dataset):
    """
    DecoderSinlgeDataset type merging NQ and TQA
    """

    def __init__(self, nq_path, tqa_path, nq_ref, tqa_ref, n_context):
        self.nq_path = nq_path
        self.tqa_path = tqa_path
        self.nq_ref = utils.open_json(nq_ref)
        self.tqa_ref = utils.open_json(tqa_ref)
        self.n_context = n_context
        self._get_file_ids_lst()
        self._get_target()

    def _get_file_ids_lst(self):
        self.ids_lst = []
        self.files = []

        for nq_file in os.listdir(self.nq_path):
            if 'pickle' in nq_file:
                nq_id = 'nq-' + nq_file.split('.')[0]
                self.ids_lst.append(nq_id)
                self.files.append(self.nq_path + '/' + nq_file)

        for tqa_file in os.listdir(self.tqa_path):
            if 'pickle' in tqa_file:
                tqa_id = 'tqa-' + tqa_file.split('.')[0]
                self.ids_lst.append(tqa_id)
                self.files.append(self.tqa_path + '/' + tqa_file)

        print(f'self.files : {len(self.files)}')
        print(f'self.ids_lst : {len(self.ids_lst)}')

    def _get_target(self):
        self.target = {}
        for ref in self.nq_ref:
            new_id = 'nq-' + str(ref['id'])
            self.target[new_id] = ref['em_pattern']

        for ref in self.tqa_ref:
            new_id = 'tqa-' + str(ref['id'])
            self.target[new_id] = ref['em_pattern']

        print(f'self.target {len(self.target)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = self.files[index]
        id_ = self.ids_lst[index]

        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        embedding_ = data['embedding'][:self.n_context, :, :]
        em_pattern_ = self.target[id_][:self.n_context]

        return {
            'id': id_,
            'embedding': embedding_,
            'em_pattern': em_pattern_,
        }


class DecoderCombinedPositiveDataset(torch.utils.data.Dataset):
    """
    DecoderSinlgeDataset type merging NQ and TQA
    """

    def __init__(self, nq_path, tqa_path, nq_ref, tqa_ref, n_context):
        self.nq_path = nq_path
        self.tqa_path = tqa_path
        self.nq_ref = utils.open_json(nq_ref)
        self.tqa_ref = utils.open_json(tqa_ref)
        self.n_context = n_context
        self._get_target()
        self._get_file_ids_lst()

    def _get_target(self):
        self.target = {}
        for ref in self.nq_ref:
            if '1' in ref['em_pattern'][:self.n_context]:
                new_id = 'nq-' + str(ref['id'])
                self.target[new_id] = ref['em_pattern']

        for ref in self.tqa_ref:
            if '1' in ref['em_pattern'][:self.n_context]:
                new_id = 'tqa-' + str(ref['id'])
                self.target[new_id] = ref['em_pattern']

        print(f'self.target {len(self.target)}')

    def _get_file_ids_lst(self):
        self.ids_lst = []
        self.files = []

        for nq_file in os.listdir(self.nq_path):
            if 'pickle' in nq_file:
                nq_id = 'nq-' + nq_file.split('.')[0]
                if nq_id in self.target:
                    self.ids_lst.append(nq_id)
                    self.files.append(self.nq_path + '/' + nq_file)

        for tqa_file in os.listdir(self.tqa_path):
            if 'pickle' in tqa_file:
                tqa_id = 'tqa-' + tqa_file.split('.')[0]
                if tqa_id in self.target:
                    self.ids_lst.append(tqa_id)
                    self.files.append(self.tqa_path + '/' + tqa_file)

        print(f'self.files : {len(self.files)}')
        print(f'self.ids_lst : {len(self.ids_lst)}')
        print(f'len(self.files) == len(self.ids_lst) : {len(self.files) == len(self.ids_lst)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = self.files[index]
        id_ = self.ids_lst[index]

        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        embedding_ = data['embedding'][:self.n_context, :, :]
        em_pattern_ = self.target[id_][:self.n_context]

        return {
            'id': id_,
            'embedding': embedding_,
            'em_pattern': em_pattern_,
        }


class DecoderCombinedSinlgeFiveLabelDataset(torch.utils.data.Dataset):
    """
    DecoderCombinedSinlgeFourLabelDataset type merging NQ and TQA
    'definite_pos' : 4
    'initial_zeros' : 3
    'semi-pos' : 2
    'sem-neg' : 1
    'definite_neg' : 0
    """

    def __init__(self, nq_path, tqa_path, nq_ref, tqa_ref, n_context):
        self.nq_path = nq_path
        self.tqa_path = tqa_path
        self.nq_ref = utils.open_json(nq_ref)
        self.tqa_ref = utils.open_json(tqa_ref)
        self.n_context = n_context
        self._get_file_ids_lst()
        self._get_target()

    def _get_file_ids_lst(self):
        self.ids_lst = []
        self.files = []

        for nq_file in os.listdir(self.nq_path):
            if 'pickle' in nq_file:
                nq_id = 'nq-' + nq_file.split('.')[0]
                self.ids_lst.append(nq_id)
                self.files.append(self.nq_path + '/' + nq_file)

        for tqa_file in os.listdir(self.tqa_path):
            if 'pickle' in tqa_file:
                tqa_id = 'tqa-' + tqa_file.split('.')[0]
                self.ids_lst.append(tqa_id)
                self.files.append(self.tqa_path + '/' + tqa_file)

        print(f'self.files : {len(self.files)}')
        print(f'self.ids_lst : {len(self.ids_lst)}')

    def _get_target(self):
        self.target = {}
        self.new_target = {}

        for ref in self.nq_ref:
            new_id = 'nq-' + str(ref['id'])
            self.target[new_id] = ref['em_pattern']
            self.new_target[new_id] = get_new_label(ref['em_pattern'], self.n_context)

        for ref in self.tqa_ref:
            new_id = 'tqa-' + str(ref['id'])
            self.target[new_id] = ref['em_pattern']
            self.new_target[new_id] = get_new_label(ref['em_pattern'], self.n_context)

        print(f'self.target {len(self.target)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = self.files[index]
        id_ = self.ids_lst[index]

        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        embedding_ = data['embedding'][:self.n_context, :, :]
        em_pattern_ = self.new_target[id_][:self.n_context]
        ori_em_pattern = self.target[id_][:self.n_context]

        return {
            'id': id_,
            'embedding': embedding_,
            'em_pattern': em_pattern_,
            'or_em_pattern': ori_em_pattern
        }