#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
#export gpu_=$CUDA_VISIBLE_DEVICES

echo "python decoder_gpt_get_probes.py \
--model_path /scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/TEST
"

python decoder_gpt_get_probes.py \
--model_path /scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/TEST

# sbatch --gpus 1 --cpus-per-gpu=8 decoder_gpt_get_probes.py
# bash slurm-decoder-gpt-get-probes.sh

