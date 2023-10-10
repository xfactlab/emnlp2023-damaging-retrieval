#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

path="/scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1"
layers=(6 12 18 24)
# bs stands for block_size
bs=80
lrs=(6e-5)
dataset=DecoderCombinedPositiveDataset

if [ $bs -eq 20 ]; then
    batchsize=128
    gr=1
elif [ $bs -eq 40 ]; then
    batchsize=64
    gr=2
elif [ $bs -eq 60 ]; then
    batchsize=32
    gr=4
elif [ $bs -eq 80 ]; then
    batchsize=16
    gr=8
elif [ $bs -eq 100 ]; then
    batchsize=8
    gr=16
else
    echo "Invalid value of bs: $bs"
    exit 1
fi

for layer in "${layers[@]}"
do
  for lr in "${lrs[@]}"
  do
#    echo $layer
#    echo $lr
#    sequential-decoder-classifier-batch64X2-lr6e-5-n_layer12-combdata-block_size40
    rn="decoder-seq-classifier-layer$layer-batch${batchsize}Xgr${gr}-lr$lr-combposdata-blocksize$bs"
    outpath="$path/$rn"

#    echo $rn
#    echo $outpath

    echo "CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_classifier.py \
    --with_tracking True \
    --report_to wandb \
    --wandb_project decoder-sequential-classifier-pos \
    --run_name "$rn" \
    --output_dir "$outpath" \
    --per_device_train_batch_size "$batchsize" \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps "$gr" \
    --train_num_workers 4 \
    --eval_num_workers 4 \
    --learning_rate "$lr" \
    --dataset_class "$dataset" \
    --n_layer "$layer" \
    --block_size "$bs""

    CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_classifier.py \
    --with_tracking True \
    --report_to wandb \
    --wandb_project decoder-sequential-classifier-pos \
    --run_name "$rn" \
    --output_dir "$outpath" \
    --per_device_train_batch_size "$batchsize" \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps "$gr" \
    --train_num_workers 4 \
    --eval_num_workers 4 \
    --learning_rate "$lr" \
    --dataset_class "$dataset" \
    --n_layer "$layer" \
    --block_size "$bs"
  done
done


# sbatch -p rtx6000 --gpus 1 --cpus-per-gpu=8 slurm-decoder-gpt-classifier-BS80-pos.sh
