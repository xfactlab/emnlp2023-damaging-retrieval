#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES


CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_predict.py \

# sbatch --gpus 1 --cpus-per-gpu=8 slurm-decoder-gpt-predict.sh