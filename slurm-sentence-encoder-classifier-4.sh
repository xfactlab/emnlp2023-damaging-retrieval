#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

## decisive_binary_data
CUDA_VISIBLE_DEVICES="$gpu_" python sentence_encoder_classifier.py \
--with_tracking True \
--report_to wandb \
--wandb_project sequence_classifier \
--run_name Trial2-FiD-Encoder-lstm-12layers-sequence_include_all \
--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/Trial2-FiD-Encoder-lstm-12layers-sequence_include_all \
--seed 42 \
--num_layers 12 \
--drop_out_rate 0.2 \
--padding -100 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 128 \
--num_train_epochs 60 \
--checkpointing_steps 20 \
--train_loss_steps 10 \
--save_max_limit 20 \
--best_metric accuracy \
--learning_rate 5e-5 \
--adam_beta1 0.9 \
--adam_beta2 0.999 \
--adam_epsilon 1e-8 \
--gradient_accumulation_steps 1 \
--lr_scheduler_type linear \
--weight_decay 0.0 \
--num_warmup_steps 70 \
--class_weights False \
--data NQ-DEV-DPR/5-fold/1 \
--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_include_all/ctx100id_split_train_1.pickle \
--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_include_all/ctx100id_split_dev_1.pickle \
--dataset_class EncoderSentenceClassificationDataset \
--num_labels 2




## desktop setting
#CUDA_VISIBLE_DEVICES=1 python sentence_encoder_classifier.py \
#--with_tracking True \
#--report_to wandb \
#--wandb_project sequence_classifier \
#--run_name Trial2-FiD-Encoder-lstm-12layers-sequence_include_all \
#--output_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/Trial2-FiD-Encoder-lstm-12layers-sequence_include_all \
#--seed 42 \
#--num_layers 12 \
#--drop_out_rate 0.2 \
#--padding -100 \
#--per_device_train_batch_size 16 \
#--per_device_eval_batch_size 32 \
#--num_train_epochs 60 \
#--checkpointing_steps 20 \
#--train_loss_steps 10 \
#--save_max_limit 10 \
#--best_metric accuracy \
#--learning_rate 5e-5 \
#--adam_beta1 0.9 \
#--adam_beta2 0.999 \
#--adam_epsilon 1e-8 \
#--gradient_accumulation_steps 4 \
#--lr_scheduler_type linear \
#--weight_decay 0.0 \
#--num_warmup_steps 70 \
#--class_weights False \
#--data NQ-DEV-DPR/5-fold/1 \
#--train_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_include_all/ctx100id_split_train_1.pickle \
#--eval_file /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_include_all/ctx100id_split_dev_1.pickle \
#--dataset_class EncoderSentenceClassificationDataset \
#--num_labels 2




