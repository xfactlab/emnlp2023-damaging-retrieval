#!/bin/bash

dataArray=("test")

for data in ${dataArray[@]}; do
  for i in {1..10}; do
      val1=$((768 / i))
      dataset="${data}.json"
      attempt="NQ_${data}_${i}_context"

      echo "test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
      --write_results \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/TEST
      "
      python test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
      --write_results \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/TEST
  done
    
  for i in {11..20}; do
      val1=$((832 / i))
      dataset="${data}.json"
      attempt="NQ_${data}_${i}_context"
      echo "test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
      --write_results \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/TEST
      "
      python test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
      --write_results \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/TEST
  done

  for i in {21..33}; do
      val1=$((896 / i))
      dataset="${data}.json"
      attempt="NQ_${data}_${i}_context"
      echo "test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
      --write_results \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/TEST
      "
      python test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
      --write_results \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/TEST
  done

  for i in {34..100}; do
      val1=$((992 / i))
      dataset="${data}.json"
      attempt="NQ_${data}_${i}_context"
      echo "test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
      --write_results \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/TEST
      "
      python test_reader-slurm.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
      --write_results \
      --per_gpu_batch_size "$val1" \
      --n_context "$i" \
      --name "$attempt" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/TEST
  done
done

# sbatch --gpus 1 depreciated_run-real-incremental-nq-qa-test-slurm-100.sh
