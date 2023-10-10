#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

# FiD setting
model_path="/data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large"

# Given a Model Path
# Find directories recursively in the current directory containing "blocksize40" &"Probes" in their names
path="/scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1"
block_size=20
# e.g.) /scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/decoder-seq-classifier-layer24-batch64Xgr2-lr6e-5-combdata-blocksize20/step_100/result/NQ-TEST/Probes
dirs=( $(find "$path" -type d -name "*Probes" -print) )
block_dirs=()

for dir in "${dirs[@]}";
do
#    echo "$dir"
    if [[ $dir =~ blocksize$block_size ]]; then
        block_dirs+=("$dir")
    fi
done

new_block_dirs=()
for block_dir in "${block_dirs[@]}";
do
    new_block_dir=$(echo "$block_dir" | grep -oE "/.*blocksize[0-9]+")
    new_block_dirs+=("$new_block_dir")
    cnt=$((cnt+1))
done

cnt = 0
for path in "${new_block_dirs[@]}";
do
  ## Get the block_size for n_context
  model_args_path="$path/model_args.json"
  # Read the JSON file into a variable
  json=$(cat "$model_args_path")
  # Extract the value of 'block_size'
  n_context=$(echo "$json" | sed -n 's/.*"block_size":[[:space:]]*\([0-9]*\).*/\1/p')

  if [ $n_context -eq 20 ]; then
    per_gpu_batch_size=42
  elif [ $n_context -eq 40 ]; then
    per_gpu_batch_size=24
  elif [ $n_context -eq 60 ]; then
    per_gpu_batch_size=16
  elif [ $n_context -eq 80 ]; then
    per_gpu_batch_size=12
  elif [ $n_context -eq 100 ]; then
    per_gpu_batch_size=8
  else
    echo "Invalid value of bs: $$n_context"
    exit 1
  fi

  echo "$path"
  echo "$model_args_path"
  echo "$n_context"
  echo "$per_gpu_batch_size"
  break
done
echo "$cnt"










#for dir in "${dirs[@]}";
#do
#    echo "$dir"
#    echo "model_args : $model_args_path"
#    echo "n_context : $n_context"
#    echo "bastch_size : $per_gpu_batch_size"
#done
#
#count=0
#
#for dir in "${dirs[@]}"; do
#  echo "$dir"
#  echo "model_args : $model_args_path"
#  echo "n_context : $n_context"
#  echo "bastch_size : $per_gpu_batch_size"
#  for i in {1..6}; do
#    probe_name="probe$i"
#    probes_file="$dir/probe$i.json"
#    probe_check="$dir/$probe_name/inferencefinished.txt"
#
#    # checking redundancy
#    if [ -f "$probe_check" ]; then
#      echo "Result file exists for ${probes_file}"
#
#    # checking probes exists
#    elif [ -f "$probes_file" ]; then
#      eval_data=$probes_file
#      echo "Found probes file in directory: $eval_data"
#
#      echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
#            --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
#            --eval_data "$eval_data" \
#            --write_results \
#            --per_gpu_batch_size "$per_gpu_batch_size" \
#            --n_context "$n_context" \
#            --name "$probe_name" \
#            --checkpoint_dir "$dir"
#        "
##      CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
##          --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
##          --eval_data "$file" \
##          --write_results \
##          --per_gpu_batch_size 9 \
##          --n_context 100 \
##          --name "$base" \
##          --checkpoint_dir /data/philhoon-relevance/FiD/results/TQA_DEV_DPR_SELECTION/
#      echo "$probe_check"
##      echo "$probes_file Finished" >> "$probe_check"
#      count=$((count+1))
#    # if not exist
#    else
#      echo "Could not find probes file in directory: $dir"
#    fi
#  done
#done
#
#echo "total probe files : $count"
#
### Print out the parent directory and integer part of the directories found one by one
##echo "Parent directory and integer part of directories containing 'step' recursively in the current directory:"
##for dir in "${dirs[@]}"
##do
##  dir_name=$(basename "$dir")
##  integer=$(echo "$dir_name" | grep -oE '[0-9]+')
##  parent_dir=$(dirname "$dir")
###  echo "$parent_dir"
###  echo "$integer"
##
##  echo "CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_predict.py \
##  --config_base_path "$parent_dir" \
##  --prediction_model_step "$integer" \
##  --dataset_class DecoderSinlgeDataset \
##  --eval_file /scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/embedding/test \
##  --ref_eval /scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/ctx100id.json \
##  --per_device_eval_batch_size 32 \
##  --eval_num_workers 8
##  "
##
##  CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_predict.py \
##  --config_base_path "$parent_dir" \
##  --prediction_model_step "$integer" \
##  --dataset_class DecoderSinlgeDataset \
##  --eval_file /scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/embedding/test \
##  --ref_eval /scratch/philhoon-relevance/decoder-classification/NQ-TEST-DPR/ctx100id.json \
##  --per_device_eval_batch_size 32 \
##  --eval_num_workers 8
##
##  echo "$dir_name Finished" >> "$dir/finished.txt"
##done
#
## sbatch -p rtx6000 --gpus 1 --cpus-per-gpu=8 slurm-decoder-gpt-FiD-inference-by-model.sh

