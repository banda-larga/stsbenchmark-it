# stsbenchmark-it

This repository contains the simple train/eval code for Italian models on the `STS Benchmark` (available at [link](https://huggingface.co/datasets/stsb_multi_mt)).
The goal of this project is to compare the performance of the various Italian language models on the `STS Benchmark`, which mesures the quality of sentence embeddings.

> STS Benchmark comprises a selection of the English datasets used in the STS tasks organized
> in the context of SemEval between 2012 and 2017. The selection of datasets include text from
> image captions, news headlines and user forums. ([source](https://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark))

These are different multilingual translations (e.g. Italian). Translation has been done with [deepl.com](https://www.deepl.com/).

## Usage

1. Train the models with:

```bash
python train.py --model_name <model_name_or_path> --num_epochs <num_epochs> --output_path <output_path>
```

Or simply run the `run.sh` script.

2. Get the results with:

```bash
python get_results.py --path <path>
```

3. Get avg. inference time with:

```bash
python inference_score.py --path <path>
```

## Results

| Model                                                  | Avg.   | Model Size | Samples/sec |
| ------------------------------------------------------ | ------ | ---------- | ----------- |
| [dbmdz/electra-base-italian-xxl-cased-discriminator]() | 0.6504 | 440MB      | 708         |
| [dbmdz/electra-base-italian-mc4-cased-discriminator]() | 0.6236 | 440MB      | 693         |
| [indigo-ai/BERTino]()                                  | 0.8225 | 273MB      | 1391        |
| [dbmdz/bert-base-italian-xxl-cased]()                  | 0.8258 | 445MB      | 702         |
| [dbmdz/bert-base-italian-xxl-uncased]()                | 0.8373 | 445MB      | 706         |
| [Musixmatch/umberto-commoncrawl-cased-v1]()            | 0.8259 | 445MB      | n.d.        |
| [Musixmatch/umberto-wikipedia-uncased-v1]()            | 0.7915 | 445MB      | n.d.        |