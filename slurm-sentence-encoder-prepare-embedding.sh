#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

#CUDA_VISIBLE_DEVICES="$gpu_" python sentence_encoder_prepare_embedding.py \
#--input_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer_exclude_indecisve/sequence_exclude_no_answer_exclude_indecisve_ctx100id_split_train_1.json \
#--output_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer_exclude_indecisve/sequence_exclude_no_answer_exclude_indecisve_ctx100id_split_train_1.pickle \
#--num_labels 2 \
#--max_seq_length 200 \
#--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \

#CUDA_VISIBLE_DEVICES=1 python sentence_encoder_prepare_embedding.py \
#--input_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer_exclude_indecisve/sequence_exclude_no_answer_exclude_indecisve_ctx100id_split_train_1.json \
#--output_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer_exclude_indecisve/sequence_exclude_no_answer_exclude_indecisve_ctx100id_split_train_1.pickle \
#--num_labels 2 \
#--max_seq_length 200 \
#--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large
#
#CUDA_VISIBLE_DEVICES=1 python sentence_encoder_prepare_embedding.py \
#--input_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer_exclude_indecisve/sequence_exclude_no_answer_exclude_indecisve_ctx100id_split_dev_1.json \
#--output_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer_exclude_indecisve/sequence_exclude_no_answer_exclude_indecisve_ctx100id_split_dev_1.pickle \
#--num_labels 2 \
#--max_seq_length 200 \
#--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large

#CUDA_VISIBLE_DEVICES=1 python sentence_encoder_prepare_embedding.py \
#--input_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer/sequence_exclude_no_answer_ctx100id_split_dev_1.json \
#--output_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer/sequence_exclude_no_answer_ctx100id_split_dev_1.pickle \
#--num_labels 2 \
#--max_seq_length 200 \
#--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large

#CUDA_VISIBLE_DEVICES=1 python sentence_encoder_prepare_embedding.py \
#--input_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer/sequence_exclude_no_answer_ctx100id_split_train_1.json \
#--output_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer/sequence_exclude_no_answer_ctx100id_split_train_1.pickle \
#--num_labels 2 \
#--max_seq_length 200 \
#--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large
#
#CUDA_VISIBLE_DEVICES=1 python sentence_encoder_prepare_embedding.py \
#--input_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_include_all/ctx100id_split_dev_1.json \
#--output_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_include_all/ctx100id_split_dev_1.pickle \
#--num_labels 2 \
#--max_seq_length 200 \
#--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large
#
#CUDA_VISIBLE_DEVICES=1 python sentence_encoder_prepare_embedding.py \
#--input_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_include_all/ctx100id_split_train_1.json \
#--output_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_include_all/ctx100id_split_train_1.pickle \
#--num_labels 2 \
#--max_seq_length 200 \
#--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large

#CUDA_VISIBLE_DEVICES=1 python sentence_encoder_prepare_embedding.py \
#--input_file_path /data/philhoon-relevance/binary-classification/NQ-TEST-DPR/ctx100id.json \
#--output_file_path /data/philhoon-relevance/binary-classification/NQ-TEST-DPR/ctx100id.pickle \
#--num_labels 2 \
#--max_seq_length 200 \
#--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large

CUDA_VISIBLE_DEVICES=1 python sentence_encoder_prepare_embedding.py \
--input_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer_partial_decisive/sequence_exclude_no_answer_partial_decisive_ctx100id_split_train_1.json \
--output_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer_partial_decisive/sequence_exclude_no_answer_partial_decisive_ctx100id_split_train_1.pickle \
--num_labels 2 \
--max_seq_length 200 \
--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large

CUDA_VISIBLE_DEVICES=1 python sentence_encoder_prepare_embedding.py \
--input_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer_partial_decisive/sequence_exclude_no_answer_partial_decisive_ctx100id_split_dev_1.json \
--output_file_path /data/philhoon-relevance/binary-classification/NQ-DEV-DPR/5-fold/1/sequence_exclude_no_answer_partial_decisive/sequence_exclude_no_answer_partial_decisive_ctx100id_split_dev_1.pickle \
--num_labels 2 \
--max_seq_length 200 \
--model_name_or_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large

============================================================================================================================================
























