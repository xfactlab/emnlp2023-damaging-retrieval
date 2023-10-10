#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

# desktop
#export gpu_=0

# Find directories recursively in the current directory containing "step" in their names
path="/scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/TEST"
dirs=( $(find "$path" -type d -name "*step*" -print) )
# Save absolute paths to dirs array
for i in "${!dirs[@]}"; do
    dirs[$i]=$(realpath "${dirs[$i]}")
done

# Print out the parent directory and integer part of the directories found one by one
echo "Parent directory and integer part of directories containing 'step' recursively in the current directory:"
for dir in "${dirs[@]}"
do
  dir_name=$(basename "$dir")
  integer=$(echo "$dir_name" | grep -oE '[0-9]+')
  parent_dir=$(dirname "$dir")
#  echo "$parent_dir"
#  echo "$integer"

  echo "CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_predict.py \
  --config_base_path "$parent_dir" \
  --prediction_model_step "$integer" \
  --dataset_class DecoderSinlgeDataset \
  --eval_file /scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/embedding/test \
  --ref_eval /scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/ctx100id.json \
  --per_device_eval_batch_size 32 \
  --eval_num_workers 8
  "

  CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_predict.py \
  --config_base_path "$parent_dir" \
  --prediction_model_step "$integer" \
  --dataset_class DecoderSinlgeDataset \
  --eval_file /scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/embedding/test \
  --ref_eval /scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/ctx100id.json \
  --per_device_eval_batch_size 32 \
  --eval_num_workers 8

  echo "$dir_name Finished" >> "$dir/finished.txt"
done

# sbatch --gpus 1 --cpus-per-gpu=8 slurm-decoder-gpt-predict.sh

