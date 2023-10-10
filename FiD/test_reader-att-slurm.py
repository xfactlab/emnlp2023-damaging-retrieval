# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler


import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model

def evaluate(model, dataset, dataloader, tokenizer, opt):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        # model.overwrite_forward_crossattention()
        # model.reset_score_storage()
        model.overwrite_forward_crossattention_token()
        model.reset_score_storage_token()
        model.reset_score_storage()

    total = 0
    exactmatch = []
    opt.global_rank = 1
    att_score_lst = []

    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt'%opt.global_rank), 'a')
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            if opt.write_crossattention_scores:
                model.reset_score_storage_token()
                model.reset_score_storage()

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50,
            )

            if opt.write_crossattention_scores:
                # crossattention_scores = model.get_crossattention_scores(context_mask.cuda())
                att_score_by_token = model.get_crossattention_scores_token(context_mask.cuda())

            # outputs are batched
            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                if 'answers' in example:
                    score = src.evaluation.ems(ans, example['answers'])
                    exactmatch.append(score)

                if opt.write_results:
                    fw.write(str(example['id']) + "\t" + ans  + "\t" + str(score) + '\n')
                if opt.write_crossattention_scores:
                    # for j in range(context_ids.size(1)):
                    #     example['ctxs'][j]['score'] = crossattention_scores[k, j].item()
                    att_score_on_k = att_score_by_token[k].detach().cpu()
                    att_score_lst.append(
                        {
                            'id': idx[k],
                            'attention_score': att_score_on_k,
                        }
                    )

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                if len(exactmatch) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average = {np.mean(exactmatch):.3f}'
                logger.warning(log)

    logger.warning(f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    
    return score, total, att_score_lst


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    # src.slurm.init_distributed_mode(opt)
    # src.slurm.init_signal_handler()
    # opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)
    opt.train_batch_size = opt.per_gpu_batch_size * 1

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    # if opt.is_distributed:
    #     torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)

    opt.is_main = True
    opt.is_distributed = False
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')

    # dir_path was executed before creating directory on the designated path
    # there might be reasons for authors to do this way.
    # but 1 executing test_reader.py will always throw an error :(
    # after 1st execution, script is running without any interruptions
    # better way to do this would be
    # if not directory_exists and opt.is_main:
    if not dir_path.exists() and opt.is_main:
        options.print_options(opt)


    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer, opt.n_context)
    # eval_examples = src.data.load_data(
    #     opt.eval_data,
    #     global_rank=opt.global_rank,
    #     # use the global rank and world size attibutes to split the eval set on multiple gpus
    #     world_size=opt.world_size
    # )
    eval_examples = src.data.load_data(
        opt.eval_data, 
        # global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        # world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context, 
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=10,
        collate_fn=collator_function
    )
    
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    device = torch.device("cuda")
    # model = model.to(opt.device)
    model = model.to(device)

    logger.info("Start eval")
    exactmatch, total, att_score_lst = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        src.util.write_output(glob_path, write_path) 
    if opt.write_crossattention_scores:
        # src.util.save_distributed_dataset(eval_dataset.data, opt)
        src.util.save_distributed_att_score(att_score_lst, opt)

