#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

data="dev"
dataset="${data}.json"
attempt="NQ_${data}_${i}_context"


if ((i >= 1 && i <= 10)); then
    val1=$((768 / i))

    echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
      --write_results \
      --write_crossattention_scores \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/DEV_ATT
    "
    CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
    --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
    --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
    --write_results \
    --write_crossattention_scores \
    --per_gpu_batch_size "$val1" \
    --n_context "$i" \
    --name "$attempt" \
    --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/DEV_ATT

elif ((i >= 11 && i <= 20)); then
    val1=$((832 / i))

    echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
      --write_results \
      --write_crossattention_scores \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/DEV_ATT
    "
    CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
    --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
    --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
    --write_results \
    --write_crossattention_scores \
    --per_gpu_batch_size "$val1" \
    --n_context "$i" \
    --name "$attempt" \
    --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/DEV_ATT

elif ((i >= 21 && i <= 33)); then
    val1=$((896 / i))

    echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
      --write_results \
      --write_crossattention_scores \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/DEV_ATT
    "
    CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
    --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
    --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
    --write_results \
    --write_crossattention_scores \
    --per_gpu_batch_size "$val1" \
    --n_context "$i" \
    --name "$attempt" \
    --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/DEV_ATT

else
    val1=$((992 / i))

    echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
      --write_results \
      --write_crossattention_scores \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/DEV_ATT
    "
    CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
    --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
    --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
    --write_results \
    --write_crossattention_scores \
    --per_gpu_batch_size "$val1" \
    --n_context "$i" \
    --name "$attempt" \
    --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/DEV_ATT

fi