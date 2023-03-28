# stsbenchmark-it

This repository contains the simple train/eval code for Italian models on the STSB dataset (available at [link](https://huggingface.co/datasets/stsb_multi_mt)).
The goal of this project is to compare the performance of the various Italian language models on the STSB dataset, which mesures the quality of sentence embeddings.

## Table of Contents

- [Getting Started](#getting-started)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Results

| Model                                                                                             | Pearson | Spearman | MSE  | Avg. |
| ------------------------------------------------------------------------------------------------- | ------- | -------- | ---- | ---- |
| [dbmdz/bert-base-italian-cased](https://huggingface.co/dbmdz/bert-base-italian-cased)             | 0.77    | 0.77     | 0.53 | 0.77 |
| [dbmdz/bert-base-italian-xxl-cased](https://huggingface.co/dbmdz/bert-base-italian-xxl-cased)     | 0.78    | 0.78     | 0.50 | 0.78 |
| [dbmdz/bert-base-italian-xxl-uncased](https://huggingface.co/dbmdz/bert-base-italian-xxl-uncased) | 0.78    | 0.78     | 0.50 | 0.78 |


