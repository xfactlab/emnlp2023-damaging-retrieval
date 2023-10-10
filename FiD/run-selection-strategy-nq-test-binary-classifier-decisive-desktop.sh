#!/usr/bin/env python3

for file in /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/roberta-decisive_binary_data-weighted-trial3_desktop1/step_240/intact_prediction/selection_methods/*.json
do
    name=${file##*/}
    base=${name%.json}

    echo "CUDA_VISIBLE_DEVICES=1 python test_reader-slurm.py \
        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
        --eval_data "$file" \
        --write_results \
        --per_gpu_batch_size 10 \
        --n_context 100 \
        --name "$base" \
        --checkpoint_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/fid-encoder-linear-decisive-trial6/step_80/intact_prediction/selection_methods
    "
    CUDA_VISIBLE_DEVICES=1 python test_reader-slurm.py \
        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
        --eval_data "$file" \
        --write_results \
        --per_gpu_batch_size 3 \
        --n_context 100 \
        --name "$base" \
        --checkpoint_dir /data/philhoon-relevance/binary-classification/results/NQ-DEV-DPR/5-fold/1/roberta-decisive_binary_data-weighted-trial3_desktop1/step_240/intact_prediction/selection_methods/
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

