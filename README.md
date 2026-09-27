# NLP Topic Modeling

Implementation and evaluation of topic modeling methods on Wikipedia text using traditional and BERT-based approaches.

## Methods

- LDA
- NMF
- BERT Document Clustering
- BERT Token Clustering

## Dataset

The experiments use a subset of the WikiText-103 Wikipedia corpus consisting of 1,000 documents.

## Evaluation

The extracted topics are evaluated using:

- C_v topic coherence
- Topic diversity
- Topic overlap
- LLM-based qualitative evaluation

For the LLM evaluation, Qwen2.5-1.5B-Instruct is used locally to assess topic quality based on coherence, interpretability, and specificity.

## Project Structure

```text
NLP_Topic_Modeling/
├── Experiments/
│   ├── 01_prepare_wikitext.py
│   ├── 02_lda.py
│   ├── 03_nmf.py
│   ├── 04_bert_document.py
│   ├── 05_bert_token.py
│   ├── 06_prepare_llm_evaluation.py
│   ├── 07_llm_evaluation.py
│   ├── 08_common_evaluation.py
│   └── 09_results_visualization.py
├── data/
│   └── wikipedia/
├── results/
│   ├── bert_document/
│   ├── bert_token/
│   ├── bert_token_filtered/
│   ├── bert_token_final/
│   ├── common_evaluation/
│   ├── figures/
│   ├── lda/
│   ├── llm_evaluation/
│   └── nmf/
├── README.md
└── .gitignore