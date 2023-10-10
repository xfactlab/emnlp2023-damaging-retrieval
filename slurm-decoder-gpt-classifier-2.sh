#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

## decisive_binary_data
CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_classifier.py \
--with_tracking True \
--report_to wandb \
--wandb_project decoder-sequential-classifier \
--run_name sequential-decoder-classifier-batch32X4-lr6e-5-n_layer12-combdata-block_size60 \
--output_dir /scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/sequential-decoder-classifier-batch32X4-lr6e-5-n_layer12-combdata-block_size60 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 4 \
--train_num_workers 4 \
--eval_num_workers 4 \
--learning_rate 6e-5 \
--dataset_class DecoderCombinedSinlgeDataset \
--n_layer 12 \
--block_size 60

# sbatch --gpus 1 --cpus-per-gpu=8 slurm-decoder-gpt-classifier-2.sh

#CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_classifier.py \
#--with_tracking True \
#--report_to wandb \
#--wandb_project decoder-sequential-classifier \
#--run_name sequential-decoder-classifier-testing-batch64-gradac2-lr5e-5 \
#--output_dir /scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/decoder-sequentail-classifier-testing-batch64-gradac2-lr5e-5 \
#--per_device_train_batch_size 64 \
#--per_device_eval_batch_size 32 \
#--gradient_accumulation_steps 2 \
#--train_num_workers 8 \
#--eval_num_workers 8 \
#--learning_rate 5e-5

#CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_classifier.py \
#--with_tracking True \
#--report_to wandb \
#--wandb_project decoder-sequential-classifier \
#--run_name sequential-decoder-classifier-testing-batch64-gradac2 \
#--output_dir /scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/decoder-sequentail-classifier-testing-batch64-gradac2 \
#--per_device_train_batch_size 64 \
#--per_device_eval_batch_size 32 \
#--gradient_accumulation_steps 2 \
#--train_num_workers 8 \
#--eval_num_workers 8 \
#--learning_rate 6e-4

# DESKTOP
#CUDA_VISIBLE_DEVICES=1 python decoder_gpt_classifier.py \
#--with_tracking True \
#--report_to wandb \
#--wandb_project decoder-sequential-classifier \
#--run_name sequential-decoder-classifier-batch-64 \
#--output_dir /scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/decoder-sequentail-classifier-batch-64 \
#--per_device_train_batch_size 64 \
#--per_device_eval_batch_size 12 \
#--gradient_accumulation_steps 1 \
#--train_num_workers 8 \
#--eval_num_workers 8


