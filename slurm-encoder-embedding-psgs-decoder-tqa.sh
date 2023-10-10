#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

#CUDA_VISIBLE_DEVICES="$gpu_" python encoder_embedding_psgs_decoder.py \
#--input_file_path /data/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1/ctx100id_split_train_1.json \
#--output_file_path /data/philhoon-relevance/decoder-classification/NQ-DEV-DPR/5-fold/1/ctx100id_embedding_train_1.pickle \
#--batch_size 1 \
#--n_context 100 \
#--max_seq_length 200 \
#--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \

CUDA_VISIBLE_DEVICES="$gpu_" python encoder_embedding_psgs_decoder.py \
--input_file_path /scratch/philhoon-relevance/decoder-classification/TQA-DEV-DPR/5-fold/1/ctx100id_split_train_1.json \
--output_file_path /scratch/philhoon-relevance/decoder-classification/TQA-DEV-DPR/5-fold/1/embedding/ \
--batch_size 1 \
--n_context 100 \
--max_seq_length 200 \
--model_name_or_path /scratch/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \

CUDA_VISIBLE_DEVICES="$gpu_" python encoder_embedding_psgs_decoder.py \
--input_file_path /scratch/philhoon-relevance/decoder-classification/TQA-DEV-DPR/5-fold/1/ctx100id_split_dev_1.json \
--output_file_path /scratch/philhoon-relevance/decoder-classification/TQA-DEV-DPR/5-fold/1/embedding/ \
--batch_size 1 \
--n_context 100 \
--max_seq_length 200 \
--model_name_or_path /scratch/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \

CUDA_VISIBLE_DEVICES="$gpu_" python encoder_embedding_psgs_decoder.py \
--input_file_path /scratch/philhoon-relevance/decoder-classification/TQA-TEST-DPR/ctx100id_test.json \
--output_file_path /scratch/philhoon-relevance/decoder-classification/TQA-TEST-DPR/embedding/ \
--batch_size 1 \
--n_context 100 \
--max_seq_length 200 \
--model_name_or_path /scratch/philhoon-relevance/FiD/pretrained_models/tqa_reader_large \

# sbatch --gpus 1 --cpus-per-gpu=8 slurm-encoder-embedding-psgs-decoder-tqa.sh