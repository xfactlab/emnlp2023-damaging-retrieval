#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

# desktop
#export gpu_=0

#!/bin/bash

#export gpu_=$CUDA_VISIBLE_DEVICES
#
#for file in /data/philhoon-relevance/FiD/open_domain_data/TQA_DEV_DPR_SELECTION/*.json
#do
#
#    if [[ $file == *"strict_positive_"* ]] && [[ $file == *"damaging_remove_damage_relevant.json" ]]; then
#        name=${file##*/}
#        base=${name%.json}
#
#        echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
#            --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
#            --eval_data "$file" \
#            --write_results \
#            --per_gpu_batch_size 9 \
#            --n_context 100 \
#            --name "$base" \
#            --checkpoint_dir /data/philhoon-relevance/FiD/results/TQA_DEV_DPR_SELECTION/
#        "
#        CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
#            --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
#            --eval_data "$file" \
#            --write_results \
#            --per_gpu_batch_size 9 \
#            --n_context 100 \
#            --name "$base" \
#            --checkpoint_dir /data/philhoon-relevance/FiD/results/TQA_DEV_DPR_SELECTION/
#    fi
#done

# FiD setting
model_path="/data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large"


# Given a Model Path
# Find directories recursively in the current directory containing "Probes" in their names
path="/scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/TEST"
dirs=( $(find "$path" -type d -name "*Probes" -print) )

# Get the block_size for n_context
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

# Save absolute paths to dirs array
for i in "${!dirs[@]}"; do
    dirs[$i]=$(realpath "${dirs[$i]}")
done

#for dir in "${dirs[@]}";
#do
#    echo "$dir"
#    echo "model_args : $model_args_path"
#    echo "n_context : $n_context"
#    echo "bastch_size : $per_gpu_batch_size"
#done

count=0

for dir in "${dirs[@]}"; do
  echo "$dir"
  echo "model_args : $model_args_path"
  echo "n_context : $n_context"
  echo "bastch_size : $per_gpu_batch_size"
  for i in {1..6}; do
    probe_name="probe$i"
    probes_file="$dir/probe$i.json"
    probe_check="$dir/$probe_name/inferencefinished.txt"

    # checking redundancy
    if [ -f "$probe_check" ]; then
      echo "Result file exists for ${probes_file}"

    # checking probes exists
    elif [ -f "$probes_file" ]; then
      eval_data=$probes_file
      echo "Found probes file in directory: $eval_data"

      echo "CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
            --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
            --eval_data "$eval_data" \
            --write_results \
            --per_gpu_batch_size "$per_gpu_batch_size" \
            --n_context "$n_context" \
            --name "$probe_name" \
            --checkpoint_dir "$dir"
        "
#      CUDA_VISIBLE_DEVICES="$gpu_" python test_reader-slurm.py \
#          --model_path /data/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \
#          --eval_data "$file" \
#          --write_results \
#          --per_gpu_batch_size 9 \
#          --n_context 100 \
#          --name "$base" \
#          --checkpoint_dir /data/philhoon-relevance/FiD/results/TQA_DEV_DPR_SELECTION/
      echo "$probe_check"
#      echo "$probes_file Finished" >> "$probe_check"
      count=$((count+1))
    # if not exist
    else
      echo "Could not find probes file in directory: $dir"
    fi
  done
done

echo "total probe files : $count"
