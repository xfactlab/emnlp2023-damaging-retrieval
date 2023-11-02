# Detrimental Contexts in Open-Domain Question Answering  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for "[Detrimental Contexts in Open-Domain Question Answering](https://arxiv.org/abs/2310.18077)" by [Philhoon Oh](https://philhoonoh.github.io/) and [James Throne](https://jamesthorne.com/)

# Data
- Data can be downloaded from the link : [Download](https://drive.google.com/file/d/1uGUiUhiRc2bVWyKaRaBvlV-afN49e5iK/view?usp=drive_link) 

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

# Retrievers (DPR, SEAL, Contriever)
Each retriever might be slightly different from the original implementation due to adjustments for local settings. Therefore, I recommend using the uploaded datasets

# Reader (FiD)
## Major modifications
1. The original inference code was based on the slurm system, so `FiD/test_reader.py` has been adjusted for local inference settings
2. Depending on the Probe pattern, the number of contexts can vary by instance. Therefore, in cases where the number of contexts used for inference is fewer than the top-N, we added mask tokens to compensate.
  
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




## Paper Reference
Please cite the following, if you find this work useful (current arxiv version)
```bibtex
@misc{oh2023detrimental,
      title={Detrimental Contexts in Open-Domain Question Answering}, 
      author={Philhoon Oh and James Thorne},
      year={2023},
      eprint={2310.18077},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Main References
Main reference papers for this work

[1] G. Izacard, E. Grave [*Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering*](https://arxiv.org/abs/2007.01282)

```bibtex
@misc{izacard2020leveraging,
      title={Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering},
      author={Gautier Izacard and Edouard Grave},
      url = {https://arxiv.org/abs/2007.0128},
      year={2020},
      publisher = {arXiv},
}
```

[2] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih [*Dense Passage Retrieval for Open-Domain Question Answering*](https://arxiv.org/abs/2004.04906)

```
@inproceedings{karpukhin-etal-2020-dense,
    title = "Dense Passage Retrieval for Open-Domain Question Answering",
    author = "Karpukhin, Vladimir and Oguz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.550",
    doi = "10.18653/v1/2020.emnlp-main.550",
    pages = "6769--6781",
}
```

[3] Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Wen-tau Yih, Sebastian Riedel, Fabio Petroni. [*“Autoregressive Search Engines: Generating Substrings as Document Identifiers.”*](https://arxiv.org/abs/2204.10628)
```bibtex
@inproceedings{bevilacqua2022autoregressive,
 title={Autoregressive Search Engines: Generating Substrings as Document Identifiers}, 
 author={Michele Bevilacqua and Giuseppe Ottaviano and Patrick Lewis and Wen-tau Yih and Sebastian Riedel and Fabio Petroni},
 booktitle={arXiv pre-print 2204.10628},
 url={https://arxiv.org/abs/2204.10628},
 year={2022},
}
```

[4] G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, E. Grave [*Unsupervised Dense Information Retrieval with Contrastive Learning*](https://arxiv.org/abs/2112.09118)
```
@misc{izacard2021contriever,
      title={Unsupervised Dense Information Retrieval with Contrastive Learning}, 
      author={Gautier Izacard and Mathilde Caron and Lucas Hosseini and Sebastian Riedel and Piotr Bojanowski and Armand Joulin and Edouard Grave},
      year={2021},
      url = {https://arxiv.org/abs/2112.09118},
      doi = {10.48550/ARXIV.2112.09118},
}
````


