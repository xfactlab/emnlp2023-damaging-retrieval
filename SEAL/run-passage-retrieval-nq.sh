#!/usr/bin/env python3

dataArray=("dev" "test")

for data in ${dataArray[@]}; do
  dataset="${data}.json"
  output="${data}.json"

  echo "TOKENIZERS_PARALLELISM=false python -m seal.search \
    --topics_format dpr_out --topics /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
    --output_format dpr --output /data/philhoon-relevance/SEAL/NQ/"$output" \
    --checkpoint /data/philhoon-relevance/SEAL/SEAL-checkpoint+index.NQ/SEAL.NQ.pt \
    --fm_index /data/philhoon-relevance/SEAL/SEAL-checkpoint+index.NQ/NQ.fm_index \
    --jobs 24 --progress --device cuda:7 --batch_size 32 \
    --beam 15
  "
  TOKENIZERS_PARALLELISM=false python -m seal.search \
  --topics_format dpr_out --topics /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
  --output_format dpr --output /data/philhoon-relevance/SEAL/NQ/"$output" \
  --checkpoint /data/philhoon-relevance/SEAL/SEAL-checkpoint+index.NQ/SEAL.NQ.pt \
  --fm_index /data/philhoon-relevance/SEAL/SEAL-checkpoint+index.NQ/NQ.fm_index \
  --jobs 24 --progress --device cuda:7 --batch_size 32 \
  --beam 15
done

