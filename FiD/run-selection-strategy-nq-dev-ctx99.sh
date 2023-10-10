#!/usr/bin/env python3

for file in /data/philhoon-relevance/FiD/open_domain_data/NQ_DPR_DEV_SELECTION/ctx_99/*.json
do
    name=${file##*/}
    base=${name%.json}

    echo "CUDA_VISIBLE_DEVICES=2 python test_reader.py \
        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
        --eval_data "$file" \
        --write_results \
        --per_gpu_batch_size 8 \
        --n_context 100 \
        --name "$base" \
        --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR_DEV_SELECTION/ctx_99
    "
    CUDA_VISIBLE_DEVICES=2 python test_reader.py \
        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
        --eval_data "$file" \
        --write_results \
        --per_gpu_batch_size 8 \
        --n_context 100 \
        --name "$base" \
        --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR_DEV_SELECTION/ctx_99
done

# Script for selection strategies
# inputs : all json files on /data/philhoon-relevance/FiD/open_domain_data/NQ_KILT_BM25_SELECTION
# outputs : /data/philhoon-relevance/FiD/results/NQ_KILT_BM25_SELECTION

#CUDA_VISIBLE_DEVICES=1 python test_reader.py \
#        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
#        --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ_KILT_BM25_SELECTION/"$dataset" \
#        --write_results \
#        --per_gpu_batch_size 128 \
#        --n_context 5 \
#        --name "$attempt" \
#        --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_KILT_BM25_SELECTION

