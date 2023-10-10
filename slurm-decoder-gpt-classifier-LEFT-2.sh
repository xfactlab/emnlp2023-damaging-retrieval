#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

path="/scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1"

# bs stands for block_size
bss=(60 80 100)
lr=6e-4
layer=24
dataset=DecoderCombinedSinlgeDataset

#echo "Batch size: $batchsize, Growth rate: $gr"

for bs in "${bss[@]}"
do
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

  rn="decoder-seq-classifier-layer$layer-batch${batchsize}Xgr${gr}-lr$lr-combdata-blocksize$bs"
  outpath="$path/$rn"

#  echo $rn
#  echo $outpath

  echo "CUDA_VISIBLE_DEVICES="$gpu_" python decoder_gpt_classifier.py \
  --with_tracking True \
  --report_to wandb \
  --wandb_project decoder-sequentail-classifier \
  --run_name "$rn" \
  --output_dir "$outpath" \
  --per_device_train_batch_size "$batchsize" \
  --per_device_eval_batch_size "32" \
  --gradient_accumulation_steps "$gr" \
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
  --per_device_train_batch_size "$batchsize" \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps "$gr" \
  --train_num_workers 4 \
  --eval_num_workers 4 \
  --learning_rate "$lr" \
  --dataset_class DecoderCombinedSinlgeDataset \
  --n_layer "$layer" \
  --block_size "$bs"
done


# sbatch --gpus 1 --cpus-per-gpu=8 slurm-decoder-gpt-classifier-LEFT-2.sh

### Doing leftovers
#/scratch/philhoon-relevance/decoder-classification/results/NQ-DEV-DPR/5-fold/1/

#decoder-seq-classifier-layer24-batch128-lr6e-5-combdata-blocksize20

#decoder-seq-classifier-layer24-batch64Xgr2-lr6e-5-combdata-blocksize40

#decoder-seq-classifier-layer24-batch32Xgr4-lr6e-4-combdata-blocksize60
#decoder-seq-classifier-layer24-batch32Xgr4-lr6e-5-combdata-blocksize60

#decoder-seq-classifier-layer24-batch16Xgr8-lr6e-4-combdata-blocksize80
#decoder-seq-classifier-layer24-batch16Xgr8-lr6e-5-combdata-blocksize80

#decoder-seq-classifier-layer24-batch8Xgr16-lr6e-4-combdata-blocksize100
#decoder-seq-classifier-layer24-batch8Xgr16-lr6e-5-combdata-blocksize100

#### By lr 6e-5 (slurm-decoder-gpt-classifier-LEFT-1.sh)
#decoder-seq-classifier-layer24-batch128Xgr1-lr6e-5-combdata-blocksize20
#decoder-seq-classifier-layer24-batch64Xgr2-lr6e-5-combdata-blocksize40
#decoder-seq-classifier-layer24-batch32Xgr4-lr6e-5-combdata-blocksize60
#decoder-seq-classifier-layer24-batch16Xgr8-lr6e-5-combdata-blocksize80
#decoder-seq-classifier-layer24-batch8Xgr16-lr6e-5-combdata-blocksize100


#### By lr 6e-4 (slurm-decoder-gpt-classifier-LEFT-2.sh)
#decoder-seq-classifier-layer24-batch32Xgr4-lr6e-4-combdata-blocksize60
#decoder-seq-classifier-layer24-batch16Xgr8-lr6e-4-combdata-blocksize80
#decoder-seq-classifier-layer24-batch8Xgr16-lr6e-4-combdata-blocksize100

#decoder-seq-classifier-layer24-batch32Xgr4-lr6e-4-combdata-blocksize60
#decoder-seq-classifier-layer24-batch16Xgr8-lr6e-4-combdata-blocksize80
#decoder-seq-classifier-layer24-batch8Xgr16-lr6e-4-combdata-blocksize100





