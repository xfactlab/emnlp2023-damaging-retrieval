#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

data="test"
dataset="${data}.json"
attempt="TQA_${data}_${i}_context"

#echo "
#data "$data"
#dataset "$dataset"
#attempt "$attempt"
#"

if ((i >= 1 && i <= 10)); then
    val1=$((768 / i))

#    echo " part 1 : "$i" \
#    CUDA_VISIBLE_DEVICES : "$CUDA_VISIBLE_DEVICES" \
#    768 \
#    val1 : "$val1"
#    "

    echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/TQA/"$dataset" \
      --write_results \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/TQA_DPR/TEST
    "
    CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
    --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
    --eval_data /data/philhoon-relevance/FiD/open_domain_data/TQA/"$dataset" \
    --write_results \
    --per_gpu_batch_size "$val1" \
    --n_context "$i" \
    --name "$attempt" \
    --checkpoint_dir /data/philhoon-relevance/FiD/results/TQA_DPR/TEST


elif ((i >= 11 && i <= 20)); then
    val1=$((832 / i))

#    echo " part 2 : "$i" \
#    CUDA_VISIBLE_DEVICES : "$gpu_" \
#    832 \
#    val1 : "$val1"
#    "

    echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/TQA/"$dataset" \
      --write_results \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/TQA_DPR/TEST
    "
    CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
    --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
    --eval_data /data/philhoon-relevance/FiD/open_domain_data/TQA/"$dataset" \
    --write_results \
    --per_gpu_batch_size "$val1" \
    --n_context "$i" \
    --name "$attempt" \
    --checkpoint_dir /data/philhoon-relevance/FiD/results/TQA_DPR/TEST

elif ((i >= 21 && i <= 33)); then
    val1=$((896 / i))

#    echo " part 3 : "$i" \
#    CUDA_VISIBLE_DEVICES : "$gpu_" \
#    896 \
#    val1 : "$val1"
#    "

    echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/TQA/"$dataset" \
      --write_results \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/TQA_DPR/TEST
    "
    CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
    --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
    --eval_data /data/philhoon-relevance/FiD/open_domain_data/TQA/"$dataset" \
    --write_results \
    --per_gpu_batch_size "$val1" \
    --n_context "$i" \
    --name "$attempt" \
    --checkpoint_dir /data/philhoon-relevance/FiD/results/TQA_DPR/TEST
else
    val1=$((992 / i))

#    echo " part 4 : "$i" \
#    CUDA_VISIBLE_DEVICES : "$CUDA_VISIBLE_DEVICES" \
#    992 \
#    val1 : "$val1"
#    "

    echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/TQA/"$dataset" \
      --write_results \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/TQA_DPR/TEST
    "
    CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
    --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
    --eval_data /data/philhoon-relevance/FiD/open_domain_data/TQA/"$dataset" \
    --write_results \
    --per_gpu_batch_size "$val1" \
    --n_context "$i" \
    --name "$attempt" \
    --checkpoint_dir /data/philhoon-relevance/FiD/results/TQA_DPR/TEST
fi