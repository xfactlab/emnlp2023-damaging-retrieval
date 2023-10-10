import numpy as np
import pathlib
from pprint import pprint
import pandas as pd
from util import utils
import re
import os
from pathlib import Path
import argparse


method_option_dict = {
    'op3' : 'remove_damage_relevant',
    'op4' : 'remove_damage_irrelevant_relevant',
}

option_p_dict = {
    'strict' : 'strict_positive',
    'naive' : 'naive_positive',
}


option_d_dict = {
    'strict' : 'strict_damaging',
    'naive' : 'naive_damaging',
}


name_dict = {
 'naive_positive_naive_damaging_remove_damage_irrelevant_relevant.json' : 'probe1.json',
 'strict_positive_naive_damaging_remove_damage_irrelevant_relevant.json' : 'probe2.json',
 'naive_positive_naive_damaging_remove_damage_relevant.json' : 'probe3.json',
 'strict_positive_naive_damaging_remove_damage_relevant.json' : 'probe4.json',
 'naive_positive_strict_damaging_remove_damage_relevant.json' : 'probe5.json',
 'strict_positive_strict_damaging_remove_damage_relevant.json' : 'probe6.json',
}

# build_data_from_prediction(input_, option, option_p, option_d, block_size)
def build_data_from_prediction(input_file, option, option_p, option_d, block_size):
    '''
    input_file : incremental inference result from FiD from KILT-5-1
        path : /data/philhoon-relevance/FiD/results/KILT_BM25_NQ/incremental_result/pos1_ctx5.json

    output : FiD input json format

    option(required) : removing strategies
        op1 : removes damages only
        op2 : removes damaging + irrelevant
        op3 : removes damaging + relevant
        op4 : removes damaging + irrelevant + relevant

    option_p(required) : positive passage selection options
        strict : strict positive
            e.g.) 11 pattern
                1st '1' is positive, 2nd '1' is relevant
        naive : naive positive
            e.g.) 11 pattern
                1st '1' is positive, 2nd '1' is positive

    option_d(required) : damaging passage selection options
        strict : strict negative
            e.g.) A00 pattern
                if there is at least one '1' occurred in A, 2nd '0' is irrelevant
        naive : naive damaging
            e.g.) A00 pattern
                if there is at least one '1' occurred in A, 2nd '0' is damaging

    '''

    output_format = []
    null_em = '0' * block_size

    # 'strict', 'naive'
    # option_p = 'naive'
    # option_d = 'naive'
    # option = 'op4'

    for id_, instance in enumerate(input_file, 1):
        template_dict = {}
        if 'id' in instance.keys():
            template_dict['id'] = instance['id']
        else:
            template_dict['id'] = str(id_)
        template_dict['answers'] = instance['answers']
        template_dict['question'] = instance['question']
        template_dict['em_pattern'] = instance['pred_em_pattern']

        # Block_size check
        if not len(infer_result[0]['pred_em_pattern']) == block_size:
            print('Block Size does not match')
            return None

        em_pattern = instance['pred_em_pattern']

        # when there is at least one EM in the accumulated inference
        if em_pattern != null_em:
            new_ctx = []

            # relevant vs positive
            positve_ctx_lst = []
            relevant_ctx_lst = []

            # irrelevant vs damaging
            damaging_ctx_lst = []
            irrelevant_ctx_lst = []

            for idx_, ctx in enumerate(instance['ctxs'][:block_size]):

                # checking current em
                cur_em = em_pattern[idx_]
                pre_em_pattern = em_pattern[:idx_]

                # first 1 : positive
                if not pre_em_pattern and cur_em == '1':
                    positve_ctx_lst.append(ctx)

                # first 0 : irrelevant
                elif not pre_em_pattern and cur_em == '0':
                    irrelevant_ctx_lst.append(ctx)

                # 01 pattern : positive
                elif pre_em_pattern and pre_em_pattern[-1] == '0' and cur_em == '1':
                    positve_ctx_lst.append(ctx)

                # 10 pattern : damaging
                elif pre_em_pattern and pre_em_pattern[-1] == '1' and cur_em == '0':
                    damaging_ctx_lst.append(ctx)

                # 11 pattern : Strict Positive(relevant) or Naive Positive(positive)
                elif pre_em_pattern and pre_em_pattern[-1] == '1' and cur_em == '1':
                    if option_p == 'strict':
                        relevant_ctx_lst.append(ctx)

                    elif option_p == 'naive':
                        positve_ctx_lst.append(ctx)

                    else:
                        # print('option_p should be either \'strict\' or \'naive\'')
                        return

                        # 00 pattern : Strict Damaging(irrelevant) or Naive Damaging(damaging)
                elif pre_em_pattern and pre_em_pattern[-1] == '0' and cur_em == '0':
                    # if '1' does not occured in A, currnet passage is irrelevant
                    if not '1' in pre_em_pattern:
                        irrelevant_ctx_lst.append(ctx)

                    # if '1' occurred in A,
                    else:
                        # strict : consider it as irrelevnat
                        if option_d == 'strict':
                            irrelevant_ctx_lst.append(ctx)

                        # naive : consider it as damaging
                        elif option_d == 'naive':
                            damaging_ctx_lst.append(ctx)

                        else:
                            # print('option_p should be either \'strict\' or \'naive\'')
                            return

                            # op1 removes damages only
            if option == 'op1':
                new_ctx.extend(positve_ctx_lst)
                new_ctx.extend(relevant_ctx_lst)
                new_ctx.extend(irrelevant_ctx_lst)


            # op2 removes damaging + irrelevant
            elif option == 'op2':
                new_ctx.extend(positve_ctx_lst)
                new_ctx.extend(relevant_ctx_lst)

            # op3 : Removes damaging + relevant
            elif option == 'op3':
                new_ctx.extend(positve_ctx_lst)
                new_ctx.extend(irrelevant_ctx_lst)

            # op4 : Removes damaging + irrelevant + relevant
            elif option == 'op4':
                new_ctx.extend(positve_ctx_lst)

            else:
                # print('option should be op1, op2, op3, op4')
                return

            template_dict['ctxs'] = new_ctx
            output_format.append(template_dict)

        # when there is no EM in the accumulated inference
        else:
            template_dict['ctxs'] = instance['ctxs']
            output_format.append(template_dict)

    # print('==============instance finished======================')
    return output_format


def build_probes(input_, output_path, name_dict, block_size):
    for o_ in method_option_dict.keys():
        for op in option_p_dict.keys():
            for od in option_d_dict.keys():
                option = o_
                option_p = op
                option_d = od

                if option == 'op4' and option_p == 'strict' and option_d == 'strict':
                    continue
                if option == 'op4' and option_p == 'naive' and option_d == 'strict':
                    continue

                filename = f'{option_p_dict[option_p]}_{option_d_dict[option_d]}_{method_option_dict[option]}.json'
                n_filename = name_dict[filename]

                output_file = os.path.join(output_path, n_filename)
                output_format = build_data_from_prediction(input_, option, option_p, option_d, block_size)

                utils.save_json(output_format, output_file)
                print(f'{n_filename} save on \n {output_path}')


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description='Example usage of argparse.')

    # add an argument
    parser.add_argument('--model_path', type=str,
                        help='Input file path.')

    # parse the arguments
    args = parser.parse_args()

    # access the input argument
    model_path = args.model_path # '/scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/TEST'
    model_args = utils.open_json(os.path.join(model_path, 'model_args.json'))
    block_size = model_args['block_size']

    path = Path(model_path)
    result_lst = path.rglob('*/*ctx*_pred.json')

    for f in result_lst:
        print(f)
        # get the pred_json
        infer_result = utils.open_json(f)

        # get the paretn path
        parent = f.parent.absolute()

        # create 'Probes' directory
        probe_path = parent / 'Probes'
        os.makedirs(probe_path, exist_ok=True)

        infer_result = utils.open_json(f)
        build_probes(infer_result, probe_path, name_dict, block_size)
