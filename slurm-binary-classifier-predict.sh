#!/bin/bash

#echo $SLURM_ARRAY_TASK_ID
#export i=$SLURM_ARRAY_TASK_ID
#echo $CUDA_VISIBLE_DEVICES
export gpu_=$CUDA_VISIBLE_DEVICES

# decisive_binary_gold_data
#CUDA_VISIBLE_DEVICES=1 python binary_classifier_predict.py \
#--do_train False \
#--do_eval True \
#--do_predict True \
#--per_device_eval_batch_size 256 \
#--prediction_model_name_or_path /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/roberta-decisive_binary_gold_data_trial1 \
#--prediction_model_step 320 \
#--max_seq_length 200 \
#--eval_file /data/philhoon-relevance/binary-classification/NQ-TEST-DPR/binary_decisive_gold_ctx100id_test.json \
#--intact_eval True \
#--dataset_class BinaryCustomDatasetDecisiveBinaryGold \
#--num_labels 2

# decisive_binary_gold_data
#CUDA_VISIBLE_DEVICES="$gpu_" python binary_classifier_predict.py \
#--do_train False \
#--do_eval True \
#--do_predict True \
#--per_device_eval_batch_size 256 \
#--prediction_model_name_or_path /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/roberta-decisive_binary_gold_data_trial1 \
#--prediction_model_step 320 \
#--max_seq_length 200 \
#--eval_file /data/philhoon-relevance/binary-classification/NQ-TEST-DPR/binary_decisive_gold_ctx100id_test.json \
#--intact_eval True \
#--dataset_class BinaryCustomDatasetDecisiveBinaryGold \
#--num_labels 2


CUDA_VISIBLE_DEVICES=1 python binary_classifier_predict.py \
--do_train False \
--do_eval True \
--do_predict True \
--per_device_eval_batch_size 256 \
--prediction_model_name_or_path /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/roberta-decisive_binary_data-weighted-trial3_desktop1 \
--prediction_model_step 240 \
--max_seq_length 200 \
--intact_eval True \
--eval_file /data/philhoon-relevance/binary-classification/NQ-TEST-DPR/binary_in_ctx100id_test.json \
--dataset_class BinarySentenceDataset \
--num_labels 2


# decisive_binary_gold_data_without_answer_input
#CUDA_VISIBLE_DEVICES=1 python binary_classifier_predict.py \
#--do_train False \
#--do_eval True \
#--do_predict True \
#--per_device_eval_batch_size 256 \
#--prediction_model_name_or_path /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/roberta-decisive_binary_gold_data_trial1 \
#--prediction_model_step 320 \
#--max_seq_length 200 \
#--eval_file /data/philhoon-relevance/binary-classification/NQ-TEST-DPR/binary_in_ctx100id_test.json \
#--intact_eval True \
#--dataset_class BinaryCustomDatasetShuffle \
#--num_labels 2



