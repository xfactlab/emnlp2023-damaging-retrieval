#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

data="dev"
dataset="${data}.json"
attempt="NQ_${data}_${i}_context"

#echo "
#data "$data"
#dataset "$dataset"
#attempt "$attempt"
#"

j=1
val1=$((640 / j))

echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
    --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
    --eval_data /data/philhoon-relevance/SEAL/NQ/"$dataset" \
    --write_results \
    --per_gpu_batch_size "$val1" \
    --n_context "$i" \
    --name "$attempt" \
    --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_SEAL/DEV
  "
  CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
  --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
  --eval_data /data/philhoon-relevance/SEAL/NQ/"$dataset" \
  --write_results \
  --per_gpu_batch_size "$val1" \
  --n_context "$i" \
  --name "$attempt" \
  --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_SEAL/DEV

