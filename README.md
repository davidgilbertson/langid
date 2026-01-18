This repo generates a model that detects the language of a code snippet.

It's meant as a reference, not something you can clone and run to produce a model.

The goal is to do langugage detection (much) better than Highlight.js, using ML/AI, but in a very small size (< 1 MB shipped to browser).

# Generating a model

`main.py` calls all the major components:

1. Load a dataset
2. Extract features from the dataset
3. Train a model on the features
4. Analyze the model

# Demo

The `web/` directory has a simple demo where you can type code and see the language detected.

Run it with `uvicorn web.serve:app --reload`, once a `model.json` file has been created.

You can set a limit on the number of features to use from the model, to get an idea for final model size.

# Datasets

(These aren't embedded in the repo.)

## Smola dataset

https://github.com/smola/language-dataset.
That repo doesn't work in Windows (invalid dir names) so I've coverted it into a sqlite database and put it in `data/local/`.

~200 languages, 10-30 samples of each

## CSN dataset

https://huggingface.co/datasets/code-search-net/code_search_net. Store the unzipped language folders under `data/local/csn/<language>/final/jsonl/<split>/*.jsonl.gz`.

6 languages, millions of samples

## Rosetta dataset

https://huggingface.co/datasets/pharaouk/rosetta-code

I've filtered the Rosetta code dataset to only the languages detected by Medium.com.

Rosetta doesn't have all of them (GraphQL, markdown, yaml) but there's still 29 languages. The problem is that it's all about code, so lacking in scripts, css, html, etc.

## The Stack V2

https://huggingface.co/datasets/bigcode/the-stack-v2

I've filtered this to 100 samples of each of these languages (aligning with medium.com).

```
[
  'C',
  'C++',
  'C-Sharp',
  'CSS',
  'Dart',
  'Diff',
  'Go',
  'GraphQL',
  'INI',
  'Java',
  'JavaScript',
  'JSON',
  'Kotlin',
  'Less',
  'Lua',
  'Makefile',
  'XML',
  'Markdown',
  'Objective-C',
  'Perl',
  'PHP',
  'Text',
  'Python',
  'R',
  'Ruby',
  'Rust',
  'SCSS',
  'Shell',
  'SQL',
  'Swift',
  'TypeScript',
  'Visual_Basic_.NET',
  'WebAssembly',
  'YAML'
]
```

# Feature extraction

`features.py` processes the data to create features.

# Training

`train_model.py` trains a model to predict the target langugae from the features.

# Experiments

`experiments.md` tracks various iterations of both feature extraction and training.
