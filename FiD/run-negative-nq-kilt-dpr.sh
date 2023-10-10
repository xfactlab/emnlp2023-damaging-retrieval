#!/usr/bin/env python3

a=6
for ((i = 1; i < a; i++)); do
  for ((j = i; j < a; j++)); do
    device_per_batch=$((576 / j))
    name="kilt_dpr_nq_dev_pos${i}"
    dataset="${name}.json"
    attempt="${name}_context${j}"

    echo "CUDA_VISIBLE_DEVICES=1 python test_reader.py \
        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
        --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ_KILT_DPR/"$dataset" \
        --write_results \
        --per_gpu_batch_size "$device_per_batch" \
        --n_context "$j" \
        --name "$attempt" \
        --checkpoint_dir /data/philhoon-relevance/FiD/results/KILT_DPR_NQ
    "
    CUDA_VISIBLE_DEVICES=1 python test_reader.py \
        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
        --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ_KILT_DPR/"$dataset" \
        --write_results \
        --per_gpu_batch_size "$device_per_batch" \
        --n_context "$j" \
        --name "$attempt" \
        --checkpoint_dir /data/philhoon-relevance/FiD/results/KILT_DPR_NQ
  done
done

# DEPRECIATED (Incremental Inference by pos and ctx size on nq-kilt-dpr)
#CUDA_VISIBLE_DEVICES=1 python test_reader.py \
#--model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \ fixed
#--eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ_KILT_DPR/kilt_dpr_nq_dev_pos1.json \ train/dev
#--write_results \ fixed
#--per_gpu_batch_size 1 \ 5,4,3,2,1
#--n_context 1 \ 1,2,3,4,5
#--name first_context \ 1_context, 2_context, 3_context, ...
#--checkpoint_dir /data/philhoon-relevance/FiD/results fixed
#kilt_dpr_nq_dev_pos5.json
#kilt_dpr_nq_dev_pos_5.json