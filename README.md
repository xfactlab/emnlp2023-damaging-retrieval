This repository contains code for damaging retrieval in open-domain question answering

# Data

## Download the dataset from the link
- Link : 



## Simulating Damaging Passages in QA 
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

## Damaging Passages in Retrievers
### Top-100 passages by retrievers 
#### Natural Questions
- DPR
  - We use retrieved list on FiD repo
  - Path : `./data/retrieval/NQ/dpr/{dev/test}.json`
- SEAL 
  - Run inference code available on SEAL repo  
  - Path : `./data/retrieval/NQ/seal/{dev/test}.json`
- Contriever 
  - Run inference script available on Contriever repo
  - Path : `./data/retrieval/NQ/contriever/{dev/test}.jsonl`
#### TriviaQA
- DPR 
  - We use retrieved list on FiD repo 
  - Path : `./data/retrieval/TQA/dpr/{dev/test}.json`
- SEAL :
  - Run inference code available on SEAL repo
  - Path : `./data/retrieval/TQA/seal/{dev/test}.json`
- Contriever
  - Run inference script available on Contriever repo
  - Path : `./data/retrieval/TQA/contriever/{dev/test}.jsonl`
 
## Selection Inference (Probe3 with varying number of contexts)

### NQ test
- This Probe set is extracted from DPR-retrieved NQ test set.
- Path : `./data/selection_infer/nq-test-dpr-probe3.json`

### TQA test
- This Probe set is extracted from DPR-retrieved TQA test set.
- Path : `./data/selection_infer/tqa-test-dpr-probe3.json`

  
# Reproducing Results

## Simulating Damaging Passages in QA
#### Inference

### Damaging Passages in Retrievers

### Creating Probes

### Selection Inference on NQ test


Pretrained models can be downloaded using [`get-model.sh`](get-model.sh). Currently availble models are [nq_reader_base, nq_reader_large, nq_retriever, tqa_reader_base, tqa_reader_large, tqa_retriever].

```shell
bash get-model.sh -m model_name
```

Performance of the pretrained models:

<table>
  <tr><td>Mode size</td><td colspan="2">NaturalQuestions</td><td colspan="2">TriviaQA</td></tr>
  <tr><td></td><td>dev</td><td>test</td><td>dev</td><td>test</td></tr>
  <tr><td>base</td><td>49.2</td><td>50.1</td><td>68.7</td><td>69.3</td></tr>
  <tr><td>large</td><td>52.7</td><td>54.4</td><td>72.5</td><td>72.5</td></tr>
</table>

The retriever obtained by distilling the reader in the retriever obtains the following results:

<table>
  <tr>
    <td colspan="3">NaturalQuestions</td>
    <td colspan="3">TriviaQA</td>
  </tr>
  <tr>
      <td>R@5</td>
      <td>R@20</td>
      <td>R@100</td>
      <td>R@5</td>
      <td>R@20</td>
      <td>R@100</td>
  </tr>
  <tr>
      <td>73.8</td>
      <td>84.3</td>
      <td>89.3</td>
      <td>77.0</td>
      <td>83.6</td>
      <td>87.7</td>
  </tr>
</table>



# I. Fusion-in-Decoder

Fusion-in-Decoder models can be trained using [`train_reader.py`](train_reader.py) and evaluated with [`test_reader.py`](test_reader.py).

### Train

[`train_reader.py`](train_reader.py) provides the code to train a model. An example usage of the script is given below:

```shell
python train_reader.py \
        --train_data train_data.json \
        --eval_data eval_data.json \
        --model_size base \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name my_experiment \
        --checkpoint_dir checkpoint \
```

Training these models with 100 passages is memory intensive. To alleviate this issue we use checkpointing with the `--use_checkpoint` option. Tensors of variable sizes lead to memory overhead. Encoder input tensors have a fixed size by default, but not the decoder input tensors. The tensor size on the decoder side can be fixed using `--answer_maxlength`. The large readers have been trained on 64 GPUs with the following hyperparameters:

```shell
python train_reader.py \
        --use_checkpoint \
        --lr 0.00005 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --text_maxlength 250 \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --total_step 15000 \
        --warmup_step 1000 \
```

### Test

You can evaluate your model or a pretrained model with [`test_reader.py`](test_reader.py). An example usage of the script is provided below.

```shell
python test_reader.py \
        --model_path checkpoint_dir/my_experiment/my_model_dir/checkpoint/best_dev \
        --eval_data eval_data.json \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name my_test \
        --checkpoint_dir checkpoint \
```



# II. Distilling knowledge from reader to retriever for question answering
This repository also contains code to train a retriever model following the method proposed in our paper: Distilling knowledge from reader to retriever for question answering. This code is heavily inspired by the [DPR codebase](https://github.com/facebookresearch/DPR) and reuses parts of it. The proposed method consists in several steps:

### 1. Obtain reader cross-attention scores
Assuming that we have already retrieved relevant passages for each question, the first step consists in generating cross-attention scores. This can be done using the option `--write_crossattention_scores` in [`test.py`](test.py). It saves the dataset with cross-attention scores in `checkpoint_dir/name/dataset_wscores.json`. To retrieve the initial set of passages for each question, different options can be considered, such as DPR or BM25.

```shell
python test.py \
        --model_path my_model_path \
        --eval_data data.json \
        --per_gpu_batch_size 4 \
        --n_context 100 \
        --name my_test \
        --checkpoint_dir checkpoint \
        --write_crossattention_scores \
```

### 2. Retriever training

[`train_retriever.py`](train_retriever.py) provides the code to train a retriever using the scores previously generated.

```shell
python train_retriever.py \
        --lr 1e-4 \
        --optim adamw \
        --scheduler linear \
        --train_data train_data.json \
        --eval_data eval_data.json \
        --n_context 100 \
        --total_steps 20000 \
        --scheduler_steps 30000 \
```


### 3. Knowldege source indexing

Then the trained retriever is used to index a knowldege source, Wikipedia in our case.

```shell
python3 generate_retriever_embedding.py \
        --model_path <model_dir> \ #directory
        --passages passages.tsv \ #.tsv file
        --output_path wikipedia_embeddings \
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 500 \
```

### 4. Passage retrieval

After indexing, given an input query, passages can be efficiently retrieved:


```shell
python passage_retrieval.py \
    --model_path <model_dir> \
    --passages psgs_w100.tsv \
    --data_path data.json \
    --passages_embeddings "wikipedia_embeddings/wiki_*" \
    --output_path retrieved_data.json \
    --n-docs 100 \
```

We found that iterating the four steps here can improve performances, depending on the initial set of documents.


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

