#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

#echo "CUDA_VISIBLE_DEVICES="$gpu_" binary_classifier.py \
#--with_tracking True \
#--report_to wandb \
#--wandb_project binary_classifier \
#--run_name testing-from_scratch \
#--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/testing-from_scratch \
#--per_device_train_batch_size 48 \
#--seed 42
#--learning_rate 5e-5 \
#--weight_decay 0.0 \
#--adam_beta1 0.9 \
#--adam_beta2 0.999 \
#--adam_epsilon 1e-8\
#--lr_scheduler_type linear \
#--gradient_accumulation_steps 1 \
#--num_train_epochs 5 \
#--num_warmup_steps 0 \
#--checkpointing_steps 100 \
#--model_name_or_path roberta-large \
#--max_seq_length 200 \
#--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_train_1_partial.json \
#--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_train_1_partial.json \
#--num_labels 2 \

#CUDA_VISIBLE_DEVICES="$gpu_" python binary_classifier_encoder.py \
#--with_tracking True \
#--report_to wandb \
#--wandb_project binary_classifier \
#--run_name fid-encoder-adpative-trial2 \
#--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/fid-encoder-adpative-trial2 \
#--per_device_train_batch_size 64 \
#--per_device_eval_batch_size 128 \
#--seed 42 \
#--learning_rate 5e-5 \
#--weight_decay 0.0 \
#--adam_beta1 0.9 \
#--adam_beta2 0.999 \
#--adam_epsilon 1e-8 \
#--lr_scheduler_type linear \
#--gradient_accumulation_steps 2 \
#--num_train_epochs 5 \
#--num_warmup_steps 0 \
#--checkpointing_steps 100 \
#--train_loss_steps 5 \
#--save_max_limit 10 \
#--best_metric accuracy \
#--model_architecture FiDT5 \
#--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
#--max_seq_length 200 \
#--data NQ-DEV-DPR/5-fold/1 \
#--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_train_1.json \
#--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_dev_2_partial.json \
#--num_labels 2 \
#--dataset_class BinaryCustomDatasetShuffle \
#--headtype adaptive \

# binary_data
#CUDA_VISIBLE_DEVICES="$gpu_" python binary_classifier_encoder.py \
#--with_tracking True \
#--report_to wandb \
#--wandb_project binary_classifier \
#--run_name fid-encoder-adpative-trial2 \
#--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/fid-encoder-adpative-trial2 \
#--per_device_train_batch_size 64 \
#--per_device_eval_batch_size 128 \
#--seed 42 \
#--learning_rate 5e-5 \
#--weight_decay 0.0 \
#--adam_beta1 0.9 \
#--adam_beta2 0.999 \
#--adam_epsilon 1e-8 \
#--lr_scheduler_type linear \
#--gradient_accumulation_steps 2 \
#--num_train_epochs 5 \
#--num_warmup_steps 0 \
#--checkpointing_steps 100 \
#--train_loss_steps 5 \
#--save_max_limit 10 \
#--best_metric accuracy \
#--model_architecture FiDT5 \
#--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
#--max_seq_length 200 \
#--data NQ-DEV-DPR/5-fold/1 \
#--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_train_1.json \
#--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/binary_data/binary_ex_ctx100id_split_dev_2_partial.json \
#--num_labels 2 \
#--dataset_class BinaryCustomDatasetShuffle \
#--headtype adaptive \

## decisive_binary_data
CUDA_VISIBLE_DEVICES="$gpu_" python binary_classifier_encoder.py \
--with_tracking True \
--report_to wandb \
--wandb_project binary_classifier \
--run_name fid-encoder-linear-decisive-trial6 \
--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/fid-encoder-linear-decisive-trial6 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 128 \
--seed 42 \
--learning_rate 5e-5 \
--weight_decay 0.0 \
--adam_beta1 0.9 \
--adam_beta2 0.999 \
--adam_epsilon 1e-8 \
--lr_scheduler_type linear \
--gradient_accumulation_steps 2 \
--num_train_epochs 30 \
--num_warmup_steps 0 \
--checkpointing_steps 10 \
--train_loss_steps 5 \
--save_max_limit 10 \
--best_metric accuracy \
--model_architecture FiDT5 \
--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
--max_seq_length 200 \
--data NQ-DEV-DPR/5-fold/1 \
--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/decisive_binary_data/binary_decisive_ctx100id_split_train_1.json \
--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/decisive_binary_data/binary_decisive_ctx100id_split_dev_1.json \
--num_labels 2 \
--dataset_class BinaryCustomDatasetShuffle \
--headtype linear \

# decisive_binary_gold_data
#CUDA_VISIBLE_DEVICES="$gpu_" python binary_classifier_encoder.py \
#--with_tracking True \
#--report_to wandb \
#--wandb_project binary_classifier \
#--run_name fid-encoder-adpative-decisive_binary_gold_data_trial1 \
#--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/fid-encoder-adpative-decisive_binary_gold_data_trial1 \
#--per_device_train_batch_size 64 \
#--per_device_eval_batch_size 128 \
#--seed 42 \
#--learning_rate 5e-5 \
#--weight_decay 0.0 \
#--adam_beta1 0.9 \
#--adam_beta2 0.999 \
#--adam_epsilon 1e-8 \
#--lr_scheduler_type linear \
#--gradient_accumulation_steps 2 \
#--num_train_epochs 10 \
#--num_warmup_steps 0 \
#--checkpointing_steps 10 \
#--train_loss_steps 5 \
#--save_max_limit 10 \
#--best_metric accuracy \
#--model_architecture FiDT5 \
#--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
#--max_seq_length 200 \
#--data NQ-DEV-DPR/5-fold/1 \
#--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/decisive_binary_gold_data/binary_decisive_gold_ctx100id_split_train_1.json \
#--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/decisive_binary_gold_data/binary_decisive_gold_ctx100id_split_dev_1.json \
#--num_labels 2 \
#--dataset_class BinaryCustomDatasetDecisiveBinaryGold \
#--headtype adpative \
