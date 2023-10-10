#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

data="dev"
dataset="${data}.json"
attempt="NQ_${data}_${i}_context"

val1=$((768 / 1))

echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
  --model_path /scratch/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
  --eval_data /scratch/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
  --write_results \
  --write_crossattention_scores \
  --per_gpu_batch_size "$val1" \
  --n_context 1 \
  --name "$attempt" \
  --checkpoint_dir /scratch/philhoon-relevance/FiD/results/NQ_DPR/DEV_ATT
"
CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
--model_path /scratch/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
--eval_data /scratch/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
--write_results \
--write_crossattention_scores \
--per_gpu_batch_size "$val1" \
--n_context 1 \
--name "$attempt" \
--checkpoint_dir /scratch/philhoon-relevance/FiD/results/NQ_DPR/DEV_ATT
