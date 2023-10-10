#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=0

# Find directories recursively in the current directory containing "step" in their names

root_path="/scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1"
root_dirs=( $(find "$root_path" -maxdepth 1 -mindepth 1 -type d) )
total_cnt=0
count=0
count2=0
dataset=DecoderSinlgeDataset

for root_dir in "${root_dirs[@]}";
do
  dirs=( $(find "$root_dir" -type d -name "*step*" -print) )

  # Save absolute paths to dirs array
  for i in "${!dirs[@]}"; do
      dirs[$i]=$(realpath "${dirs[$i]}")
  done

  # Print out the parent directory and integer part of the directories found one by one
  echo "Parent directory and integer part of directories containing 'step' recursively in the current directory:"
  for dir in "${dirs[@]}";
  do
    total_cnt=$((total_cnt+1))
    dir_name=$(basename "$dir")
    integer=$(echo "$dir_name" | grep -oE '[0-9]+')
    parent_dir=$(dirname "$dir")

    if [ -e "$dir/finished.txt" ]
    then
      echo "$dir/finished.txt exists"
      # echo "$parent_dir"
      count=$((count+1))
    else
      # echo "$dir_name"
      # echo "$integer"
      echo "$dir/finished.txt does not exists"
      # echo "$parent_dir"

      echo "CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_predict.py \
      --config_base_path "$parent_dir" \
      --prediction_model_step "$integer" \
      --dataset_class "$dataset" \
      --eval_file /scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/embedding/test \
      --ref_eval /scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/ctx100id.json \
      --per_device_eval_batch_size 12 \
      --eval_num_workers 8
      "

      CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_predict.py \
      --config_base_path "$parent_dir" \
      --prediction_model_step "$integer" \
      --dataset_class "$dataset" \
      --eval_file /scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/embedding/test \
      --ref_eval /scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/ctx100id.json \
      --per_device_eval_batch_size 12 \
      --eval_num_workers 8

      echo "$dir_name Finished" >> "$dir/finished.txt"
      count2=$((count2+1))
    fi
  done
done

total=$((count+count2))
echo "Total $total_cnt"
echo "Completion $count"
echo "Process $count2"
echo "Total Completed $total"

# sbatch -p rtx6000 --gpus 1 --cpus-per-gpu=8 slurm-decoder-gpt-predict-3.sh
#sbatch -p desktop2 test.sh
