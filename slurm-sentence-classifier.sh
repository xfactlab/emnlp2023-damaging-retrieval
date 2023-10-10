#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

## decisive_binary_data
CUDA_VISIBLE_DEVICES="$gpu_" python sentence_classifier.py \
--with_tracking True \
--report_to wandb \
--wandb_project sequence_classifier \
--run_name sentencebert-lstm-12layers-sequence_exclude_no_answer_include_indecisve_trial \
--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/sentencebert-lstm-12layers-sequence_exclude_no_answer_include_indecisve_trial \
--seed 42 \
--num_layers 12 \
--drop_out_rate 0.2 \
--padding -100 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 128 \
--num_train_epochs 30 \
--checkpointing_steps 20 \
--train_loss_steps 10 \
--save_max_limit 10 \
--best_metric f1 \
--weight_decay 0.0 \
--adam_beta1 0.9 \
--adam_beta2 0.999 \
--adam_epsilon 1e-8 \
--gradient_accumulation_steps 1 \
--lr_scheduler_type linear \
--num_warmup_steps 0 \
--class_weights False \
--model_architecture SentenceTransformer \
--model_name_or_path all-MiniLM-L6-v2 \
--data NQ-DEV-DPR/5-fold/1 \
--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer/sequence_exclude_no_answer_ctx100id_split_train_1.json \
--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer/sequence_exclude_no_answer_ctx100id_split_dev_1.json \
--dataset_class SentenceClassificationDataset \
--num_labels 2





# desktop setting
#CUDA_VISIBLE_DEVICES=1 python sentence_classifier.py \
#--with_tracking True \
#--report_to wandb \
#--wandb_project sequence_classifier \
#--run_name sentencebert-lstm-sequence_exclude_no_answer_exclude_indecisve-trial1 \
#--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1//data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/sentencebert-lstm-sequence_exclude_no_answer_exclude_indecisve-trial1 \
#--seed 42 \
#--num_layers 2 \
#--drop_out_rate 0.2 \
#--padding -100 \
#--per_device_train_batch_size 32 \
#--per_device_eval_batch_size 64 \
#--num_train_epochs 1 \
#--checkpointing_steps 20 \
#--train_loss_steps 10 \
#--save_max_limit 10 \
#--best_metric f1 \
#--weight_decay 0.0 \
#--adam_beta1 0.9 \
#--adam_beta2 0.999 \
#--adam_epsilon 1e-8 \
#--gradient_accumulation_steps 1 \
#--lr_scheduler_type linear \
#--num_warmup_steps 0 \
#--class_weights False \
#--model_architecture SentenceTransformer \
#--model_name_or_path all-MiniLM-L6-v2 \
#--data NQ-DEV-DPR/5-fold/1 \
#--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer_exclude_indecisve/sequence_exclude_no_answer_exclude_indecisve_ctx100id_split_train_1.json \
#--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer_exclude_indecisve/sequence_exclude_no_answer_exclude_indecisve_ctx100id_split_dev_1.json \
#--dataset_class SentenceClassificationDataset \
#--num_labels 2




