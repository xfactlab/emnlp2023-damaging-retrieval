#!/usr/bin/env python3

a=6
for ((i = 1; i < a; i++)); do
  for ((j = i; j < a; j++)); do
    device_per_batch=$((576 / j))
    name="kilt_bm25_nq_dev_pos${i}"
    dataset="${name}.json"
    attempt="${name}_context${j}"

    echo "CUDA_VISIBLE_DEVICES=1 python test_reader.py \
        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
        --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ_KILT_BM25/"$dataset" \
        --write_results \
        --per_gpu_batch_size "$device_per_batch" \
        --n_context "$j" \
        --name "$attempt" \
        --checkpoint_dir /data/philhoon-relevance/FiD/results/KILT_BM25_NQ
    "
    CUDA_VISIBLE_DEVICES=1 python test_reader.py \
        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
        --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ_KILT_BM25/"$dataset" \
        --write_results \
        --per_gpu_batch_size "$device_per_batch" \
        --n_context "$j" \
        --name "$attempt" \
        --checkpoint_dir /data/philhoon-relevance/FiD/results/KILT_BM25_NQ
  done
done

# Incremental Inference by pos and ctx size on nq-kilt-bm25
#CUDA_VISIBLE_DEVICES=1 python test_reader.py \
#        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
#        --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ_KILT_BM25/"$dataset" \
#        --write_results \
#        --per_gpu_batch_size "$device_per_batch" \
#        --n_context "$j" \
#        --name "$attempt" \
#        --checkpoint_dir /data/philhoon-relevance/FiD/results/KILT_BM25_NQ