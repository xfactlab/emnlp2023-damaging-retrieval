#!/usr/bin/env python3

a=6
for ((i = 1; i < a; i++)); do
  for ((j = i; j < a; j++)); do
    device_per_batch=$((576 / j))

    name="kilt_rand_nq_dev_pos${i}"
    dataset="${name}.json"
    attempt="${name}_context${j}"

    echo "CUDA_VISIBLE_DEVICES=3 python test_reader.py \
        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
        --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ_KILT_RANDOM/"$dataset" \
        --write_results \
        --per_gpu_batch_size "$device_per_batch" \
        --n_context "$j" \
        --name "$attempt" \
        --checkpoint_dir /data/philhoon-relevance/FiD/results/KILT_RANDOM_NQ
    "
    CUDA_VISIBLE_DEVICES=3 python test_reader.py \
        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
        --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ_KILT_RANDOM/"$dataset" \
        --write_results \
        --per_gpu_batch_size "$device_per_batch" \
        --n_context "$j" \
        --name "$attempt" \
        --checkpoint_dir /data/philhoon-relevance/FiD/results/KILT_RANDOM_NQ
  done
done

# Incremental Inference by pos and ctx size on nq-kilt-random
#CUDA_VISIBLE_DEVICES=1,2 python test_reader.py \
#--model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
#--eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ_KILT_RANDOM/kilt_rand_nq_dev_pos1.json \
#--write_results \
#--per_gpu_batch_size 576 \
#--n_context 1 \
#--name kilt_dpr_nq_dev_pos1_context1_gpu12 \
#--checkpoint_dir /data/philhoon-relevance/FiD/results/KILT_RANDOM_NQ_test

#kilt_dpr_nq_dev_pos5.json
#kilt_dpr_nq_dev_pos_5.json