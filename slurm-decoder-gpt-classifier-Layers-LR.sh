#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

layers=(6 12 18 24)
bs=20
lrs=(6e-4 6e-5)
dataset=DecoderCombinedSinlgeDataset

path="/scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1"

for layer in "${layers[@]}"
do
  for lr in "${lrs[@]}"
  do
#    echo $layer
#    echo $lr
#    sequential-decoder-classifier-batch64X2-lr6e-5-n_layer12-combdata-block_size40
    rn="decoder-seq-classifier-layer$layer-batch128-lr$lr-combdata-blocksize$bs"
    outpath="$path/$rn"

#    echo $rn
#    echo $outpath

    echo "CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_classifier.py \
    --with_tracking True \
    --report_to wandb \
    --wandb_project decoder-sequentail-classifier \
    --run_name "$rn" \
    --output_dir "$outpath" \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --train_num_workers 4 \
    --eval_num_workers 4 \
    --learning_rate "$lr" \
    --dataset_class DecoderCombinedSinlgeDataset \
    --n_layer "$layer" \
    --block_size "$bs""

    CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_classifier.py \
    --with_tracking True \
    --report_to wandb \
    --wandb_project decoder-sequentail-classifier \
    --run_name "$rn" \
    --output_dir "$outpath" \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --train_num_workers 4 \
    --eval_num_workers 4 \
    --learning_rate "$lr" \
    --dataset_class DecoderCombinedSinlgeDataset \
    --n_layer "$layer" \
    --block_size "$bs"
  done
done


# sbatch --gpus 1 --cpus-per-gpu=8 slurm-decoder-gpt-classifier-Layers-LR.sh
