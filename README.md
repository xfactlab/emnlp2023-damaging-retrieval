This repository contains code for damaging retrieval in open-domain question answering

# Data
- Data can be downloaded from the link :  

## Data description
### 1. Simulating Damaging Passages in QA 

#### Random Sampling
- Data : 2539 instances in NQ-dev set
- For random sampling, we uniformly select 4 random passage over the Corpus and add 1 gold context
- Path : `./data/simluation/random-sampling/random_dev_pos{i}.json`
- `i` refers to the position of the gold context

#### Negative Sampling
- Data : 2539 instances in NQ-dev set
- For negative sampling, we use to select 4 passages with high BM25 scores but do not containing answers, and add 1 gold context
- Path : `./data/simluation/negative-sampling/bm25_nq_dev_pos{i}.json`
- `i` refers to the position of the gold context

### 2. Damaging Passages in Retrievers
#### Top-100 passages by retrievers (NQ)
- DPR
  - We use retrieved list on FiD repo
  - Path : `./data/retrieval/NQ/dpr/{dev/test}.json`
- SEAL 
  - Run inference code available on SEAL repo  
  - Path : `./data/retrieval/NQ/seal/{dev/test}.json`
- Contriever 
  - Run inference script available on Contriever repo
  - Path : `./data/retrieval/NQ/contriever/{dev/test}.jsonl`
#### Top-100 passages by retrievers (TriviaQA)
- DPR 
  - We use retrieved list on FiD repo 
  - Path : `./data/retrieval/TQA/dpr/{dev/test}.json`
- SEAL :
  - Run inference code available on SEAL repo
  - Path : `./data/retrieval/TQA/seal/{dev/test}.json`
- Contriever
  - Run inference script available on Contriever repo
  - Path : `./data/retrieval/TQA/contriever/{dev/test}.jsonl`
 
### 3. Selection Inference
#### NQ test
- This Probe set is extracted from DPR-retrieved NQ test set.
- Path : `./data/selection_infer/nq-test-dpr-probe3.json`

#### TQA test 
- This Probe set is extracted from DPR-retrieved TQA test set.
- Path : `./data/selection_infer/tqa-test-dpr-probe3.json`

  
# Experiments

### 1. Simulating Damaging Passages in QA
To reproduce the simulation result in section 3, run following code.
```shell
cd FiD

python test_reader.py \
--model_path {FiD model path} \
--eval_data ./data/simluation/random-sampling/random_dev_pos{i}.json \
--write_results \
--per_gpu_batch_size {batch_size} \
--n_context {number of contexts} \
--name {experiment name} \
--checkpoint_dir {output_dir}
```

### 2. Damaging Passages in Retrievers
Commands used for running the incremental inference for a given retrieved list by executing below command for `n_context` ranges from 1 to 100.
This process is memory-intensive and took about 1 week to finish inference on 6 cases(3 retriever X 2 dasets) on 8 rtx6000 gpus.
```shell
cd FiD

python test_reader-slurm.py \
--model_path {FiD model path} \
--eval_data ./data/retrieval/NQ/dpr/{dev/test}.json \
--write_results \
--per_gpu_batch_size {batch_size} \
--n_context {number of contexts} \
--name {experiment name} \
--checkpoint_dir {output_dir}
```

### 3. Selection Inference
Selection Inference on Probe3 with varying number of the context sizes (5, 10, 20, 40)

```shell
cd FiD

python test_reader-slurm.py \
--model_path {FiD model path} \
--eval_data {./data/selection_infer/nq-test-dpr-probe3.json} \
--write_results \
--per_gpu_batch_size {batch_size} \
--n_context {number of contexts} \
--name {experiment name} \
--checkpoint_dir {output_dir}
```




## References

[//]: # ([//]: # &#40;[1] G. Izacard, E. Grave [*Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering*]&#40;https://arxiv.org/abs/2007.01282&#41;&#41;)
[//]: # ()
[//]: # (```bibtex)

[//]: # (@inproceedings{)

[//]: # (anonymous2023is,)

[//]: # (title={Is too much context detrimental for open-domain question answering?},)

[//]: # (author={Anonymous},)

[//]: # (booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},)

[//]: # (year={2023},)

[//]: # (url={https://openreview.net/forum?id=HickNiCqk9})

[//]: # (})

[//]: # (```)

