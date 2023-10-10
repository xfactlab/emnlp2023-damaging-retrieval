#!/usr/bin/env python3

for i in {6..10}; do

  echo "${i}"
  dataset="/data/philhoon-relevance/FiD/open_domain_data/NQ_DPR_DEV_SELECTION_METHOD3/ctx_100/top_zeros_${i}/method3_topzeros${i}.json"
  name=${dataset##*/}
  base=${name%.json}

  echo "CUDA_VISIBLE_DEVICES=0 python test_reader.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data "$dataset" \
      --write_results \
      --per_gpu_batch_size 8 \
      --n_context 100 \
      --name "$base" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR_DEV_SELECTION_METHOD3/ctx_100
  "
  CUDA_VISIBLE_DEVICES=0 python test_reader.py \
      --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
      --eval_data "$dataset" \
      --write_results \
      --per_gpu_batch_size 8 \
      --n_context 100 \
      --name "$base" \
      --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR_DEV_SELECTION_METHOD3/ctx_100

done
#for file in /data/philhoon-relevance/FiD/open_domain_data/NQ_DPR_DEV_SELECTION_METHOD3/ctx_100/*/*.json
#do
#    name=${file##*/}
#    base=${name%.json}
#
#    echo "CUDA_VISIBLE_DEVICES=0 python test_reader.py \
#        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
#        --eval_data "$file" \
#        --write_results \
#        --per_gpu_batch_size 8 \
#        --n_context 100 \
#        --name "$base" \
#        --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR_DEV_SELECTION_METHOD3/ctx_100
#    "
#    CUDA_VISIBLE_DEVICES=0 python test_reader.py \
#        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
#        --eval_data "$file" \
#        --write_results \
#        --per_gpu_batch_size 8 \
#        --n_context 100 \
#        --name "$base" \
#        --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR_DEV_SELECTION_METHOD3/ctx_100
#done

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

