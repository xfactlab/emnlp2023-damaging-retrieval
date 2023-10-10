#!/usr/bin/env python3


for i in {86..100}; do
    val1=$((192 / i))

    attempt="NQ_dpr_noraml_${i}_context"
    echo "CUDA_VISIBLE_DEVICES=0 python test_reader.py \
    --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
    --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/dpr_normal/nq-dev.json \
    --write_results \
    --per_gpu_batch_size 1 \
    --n_context "$i" \
    --name "$attempt" \
    --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/dpr_normal3
    "
    CUDA_VISIBLE_DEVICES=0 python test_reader.py \
    --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
    --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/dpr_normal/nq-dev.json \
    --write_results \
    --per_gpu_batch_size 1 \
    --n_context "$i" \
    --name "$attempt" \
    --checkpoint_dir /data/philhoon-relevance/FiD/results/NQ_DPR/dpr_normal3
done


#dataArray=("dev" "train")
#
#for data in ${dataArray[@]}; do
#    for i in {1..20}; do
#        val1=$((192 / i))
#        dataset="${data}.json"
#        attempt="NQ_${data}_${i}_context"
##        echo "${val1}"
#        echo "CUDA_VISIBLE_DEVICES=1 python test_reader.py \
#        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
#        --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
#        --write_results \
#        --per_gpu_batch_size "$val1" \
#        --n_context "$i" \
#        --name "$attempt" \
#        --checkpoint_dir /data/philhoon-relevance/FiD/results
#        "
#        CUDA_VISIBLE_DEVICES=1 python test_reader.py \
#        --model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \
#        --eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/"$dataset" \
#        --write_results \
#        --per_gpu_batch_size "$val1" \
#        --n_context "$i" \
#        --name "$attempt" \
#        --checkpoint_dir /data/philhoon-relevance/FiD/results
#    done
#done

#CUDA_VISIBLE_DEVICES=1 python test_reader.py \
#--model_path /data/philhoon-relevance/FiD/pretrained_models/nq_reader_large \ fixed
#--eval_data /data/philhoon-relevance/FiD/open_domain_data/NQ/dev.json \ train/dev
#--write_results \ fixed
#--per_gpu_batch_size 1 \ 5,4,3,2,1
#--n_context 1 \ 1,2,3,4,5
#--name first_context \ 1_context, 2_context, 3_context, ...
#--checkpoint_dir /data/philhoon-relevance/FiD/results fixed
