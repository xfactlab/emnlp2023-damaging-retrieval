#!/bin/bash

root_path="/scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1"
root_dirs=( $(find "$root_path" -maxdepth 1 -mindepth 1 -type d) )

for root_dir in "${root_dirs[@]}";
do

  echo "python decoder_gpt_get_probes.py \
  --model_path "$root_dir"
  "

  python decoder_gpt_get_probes.py \
  --model_path "$root_dir"

done
