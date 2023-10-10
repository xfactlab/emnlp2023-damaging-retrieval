#!/usr/bin/env python3

dataArray=("dev" "test")

for data in ${dataArray[@]}; do
  dataset="${data}.json"

  echo "CUDA_VISIBLE_DEVICES=5 python passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages /data/philhoon-relevance/FiD/open_domain_data/wikipedia_psgs/psgs_w100.tsv \
    --passages_embeddings "/data/philhoon-relevance/contriever/wikipedia_embeddings/contriever_msmacro/wikipedia_embeddings/*" \
    --data /data/philhoon-relevance/FiD/open_domain_data/TQA/"$dataset" \
    --output_dir /data/philhoon-relevance/contriever/TQA/contriever-msmarco \
    --per_gpu_batch_size 256
  "
  CUDA_VISIBLE_DEVICES=5 python passage_retrieval.py \
  --model_name_or_path facebook/contriever-msmarco \
  --passages /data/philhoon-relevance/FiD/open_domain_data/wikipedia_psgs/psgs_w100.tsv \
  --passages_embeddings "/data/philhoon-relevance/contriever/wikipedia_embeddings/contriever_msmacro/wikipedia_embeddings/*" \
  --data /data/philhoon-relevance/FiD/open_domain_data/TQA/"$dataset" \
  --output_dir /data/philhoon-relevance/contriever/TQA/contriever-msmarco \
  --per_gpu_batch_size 256
  
done

#python passage_retrieval.py \
#    --model_name_or_path facebook/contriever \
#    --passages psgs_w100.tsv \
#    --passages_embeddings "contriever_embeddings/*" \
#    --data nq_dir/test.json \
#    --output_dir contriever_nq \