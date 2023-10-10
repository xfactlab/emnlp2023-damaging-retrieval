#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

#echo "CUDA_VISIBLE_DEVICES="$gpu_" python binary_trainer.py \
#--wandb_project binary_classifier \
#--report_to wandb \
#--run_name testing-binary-nq-dev-5-fold-1 \
#--model_architecture roberta-large \
#--model_name_or_path roberta-large \
#--git_tag v1.2 \
#--max_seq_length 200 \
#--data binary-nq-dev-5-fold-1 \
#--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_train_1.json \
#--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_dev_1.json \
#--num_labels 2 \
#--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1 \
#--do_train True \
#--do_eval True \
#--logging_steps 10 \
#--save_steps 100 \
#--eval_steps 100 \
#--evaluation_strategy steps \
#--save_total_limit 10 \
#--load_best_model_at_end True \
#--learning_rate 5e-5 \
#--metric_for_best_model accuracy \
#--per_device_train_batch_size 48 \
#--per_device_eval_batch_size 96 \
#--num_train_epochs 3 \
#--seed 42
#"

#CUDA_VISIBLE_DEVICES="$gpu_" python binary_trainer.py \
#--wandb_project binary_classifier \
#--report_to wandb \
#--run_name testing-binary-nq-dev-5-fold-1 \
#--model_architecture roberta-large \
#--model_name_or_path roberta-large \
#--git_tag v1.2 \
#--max_seq_length 200 \
#--data binary-nq-dev-5-fold-1 \
#--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_train_1.json \
#--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_dev_1.json \
#--num_labels 2 \
#--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1 \
#--do_train True \
#--do_eval True \
#--logging_steps 10 \
#--save_steps 1000 \
#--eval_steps 1000 \
#--evaluation_strategy steps \
#--save_total_limit 5 \
#--load_best_model_at_end True \
#--learning_rate 5e-5 \
#--metric_for_best_model accuracy \
#--per_device_train_batch_size 64 \
#--per_device_eval_batch_size 128 \
#--num_train_epochs 3 \
#--seed 42

#CUDA_VISIBLE_DEVICES="$gpu_" python binary_trainer.py \
#--wandb_project binary_classifier \
#--report_to wandb \
#--run_name testing-binary-nq-dev-5-fold-1-trial2 \
#--model_architecture roberta-large \
#--model_name_or_path roberta-large \
#--git_tag v1.2 \
#--max_seq_length 200 \
#--data binary-nq-dev-5-fold-1 \
#--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_train_1.json \
#--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_dev_1.json \
#--num_labels 2 \
#--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/trial2 \
#--do_train True \
#--do_eval True \
#--logging_steps 10 \
#--save_steps 1000 \
#--eval_steps 1000 \
#--evaluation_strategy steps \
#--save_total_limit 5 \
#--load_best_model_at_end True \
#--learning_rate 3e-4 \
#--metric_for_best_model accuracy \
#--per_device_train_batch_size 64 \
#--per_device_eval_batch_size 128 \
#--num_train_epochs 3 \
#--seed 42

CUDA_VISIBLE_DEVICES="$gpu_" python binary_trainer.py \
--wandb_project binary_classifier \
--report_to wandb \
--run_name testing-binary-nq-dev-5-fold-1-trial-3 \
--model_architecture roberta-large \
--model_name_or_path roberta-large \
--git_tag v1.2 \
--max_seq_length 200 \
--data binary-nq-dev-5-fold-1 \
--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_train_1.json \
--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_dev_1.json \
--num_labels 2 \
--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/trial-3 \
--do_train True \
--do_eval True \
--logging_steps 10 \
--save_steps 1000 \
--eval_steps 1000 \
--evaluation_strategy steps \
--save_total_limit 5 \
--load_best_model_at_end True \
--learning_rate 3e-4 \
--metric_for_best_model accuracy \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 128 \
--num_train_epochs 5 \
--seed 42
