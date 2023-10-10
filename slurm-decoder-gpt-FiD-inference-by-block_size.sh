#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

# FiD setting
model_path="/data/philhoon-relevance/FiD/pretrained_models/nq_reader_large"

# Given a Model Path
# Find directories recursively in the current directory containing "blocksize40" &"Probes" in their names
path="/scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1"
block_size=20
count=0
# e.g.) /scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/decoder-seq-classifier-layer24-batch64Xgr2-lr6e-5-combdata-blocksize20/step_100/result/NQ-TEST/Probes
dirs=( $(find "$path" -type d -name "*blocksize$block_size" -print) )
block_dirs=()

for path in "${dirs[@]}";
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

  probe_paths=( $(find "$path" -type d -name "*Probes" -print) )

#  echo "Path : $path"

  for dir in "${probe_paths[@]}";
  do
#    echo "dir : $dir"
    for i in 3; do
      probe_name="probe$i"
      probes_file="$dir/probe$i.json"
      probe_check="$dir/$probe_name/inferencefinished.txt"

#      echo "n_context : $n_context"
#      echo "per_gpu_batch_size : $per_gpu_batch_size"
#      echo "probes_name : $probe_name"
#      echo "probes_file : $probes_file"
#      echo "probe_check : $probe_check"


      # checking redundancy
      if [ -f "$probe_check" ]; then
        echo "Result file exists for ${probes_file}"
      # checking probes exists
      elif [ -f "$probes_file" ]; then
        eval_data=$probes_file
#        echo "eval _ file: $eval_data"

        echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
            --model_path "$model_path" \
            --eval_data "$eval_data" \
            --write_results \
            --per_gpu_batch_size "$per_gpu_batch_size" \
            --n_context "$n_context" \
            --name "$probe_name" \
            --checkpoint_dir "$dir"
        "
#        CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
#        --model_path "$model_path" \
#        --eval_data "$eval_data" \
#        --write_results \
#        --per_gpu_batch_size "$per_gpu_batch_size" \
#        --n_context "$n_context" \
#        --name "$probe_name" \
#        --checkpoint_dir "$dir"

        echo "$probe_check"
#        echo "$probes_file Finished" >> "$probe_check"
        count=$((count+1))
      else
        echo "Could not find probes file in directory: $dir"
      fi
    done
  done
done
echo "$count"


### sbatch -p rtx6000 --gpus 1 --cpus-per-gpu=8 slurm-decoder-gpt-FiD-inference-by-model.sh

