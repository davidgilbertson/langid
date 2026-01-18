# Rules

Each experiment branches from the previous experiment, unless stated otherwise.
Once an experiment has been run, the files changed should be copied into `experiments/` with a suffix like `_ex<n>` to signify the experiment (without updating them at all, not even import paths).
These files are for reference only, and won't necessarily run in the future.
Experiments 1-4 used the smola/language-dataset sqlite snapshot.

# Experiment 1

Assets:

- `experiments/features_ex1.py`
- `experiments/train_model_ex1.py`

Metrics:

- Accuracy: 29%
- Speed: ~7,500 items/second
- Train time: ~300ms
- Feature file size: `len(Path('features.py').read_text())` is ~2,000

## Features

`extract_features` is just string matching for some common words.

## Model

Logistic regression

# Experiment 2

Assets:

- `experiments/features_ex2.py`
- `experiments/train_model_ex2.py`

Metrics:

- Accuracy: 60%
- Speed: ~1,374 items/second
- Feature file size: `len(Path('features.py').read_text())` is ~7,500

## Features

Used a broader set of rule-based features without regex by using faster string and line scans:

- Word-boundary checks for keywords (e.g., from/import, SQL, Python, Rust, Lua, etc.).
- Operator and symbol cues (e.g., :=, =>, ->, ::, triple quotes, XML declaration).
- Simple structural hints (YAML key/value lines, closing tags, indent colons, MATLAB % comments).
- Lightweight language-specific hints (shebang, PHP tags, $vars, Rust macros, Go package/func).

## Model

Same LogisticRegression setup.

Top confusions (Actual -> Predicted):

- Nemerle -> C#
- Monkey -> BlitzMax
- Ruby -> Crystal
- Vala -> EQ
- Crystal -> Ruby
- Haskell -> Idris
- Haxe -> ActionScript
- IGOR Pro -> Scilab
- Java -> AspectJ
- LiveScript -> CoffeeScript
- REALbasic -> Xojo
- Shen -> Arc
- TXL -> XML
- Verilog -> SystemVerilog
- XQuery -> Kit
- Xojo -> REALbasic
- ZIL -> Forth

# Experiment 3

Assets:

- `experiments/features_ex3.py`
- `experiments/train_model_ex3.py`

Metrics:

- Accuracy: 61%
- Speed: ~1,293 items/second
- Feature file size: `len(Path('features.py').read_text())` is ~8,700

## Features

Targeted new features aimed at the biggest confusions (non-regex):

- AspectJ vs Java: aspect/pointcut keywords.
- Verilog vs SystemVerilog: always_ff/always_comb/logic/modport and endmodule.
- Ruby vs Crystal: puts/attr_* vs macro/pointerof.
- Haxe vs ActionScript: typedef/inline/macro.
- Idris vs Haskell: total/mutual.
- Nemerle vs C#: macro/syntax.

## Model

Same LogisticRegression setup.

Top confusions (Actual -> Predicted):

- Nemerle -> C#
- Monkey -> BlitzMax
- REALbasic -> Xojo
- Ruby -> Crystal
- Vala -> EQ
- Common Lisp -> Emacs Lisp
- PL/SQL -> Ada
- Crystal -> Ruby
- Haskell -> Idris
- Haxe -> ActionScript
- IGOR Pro -> Scilab
- LiveScript -> CoffeeScript
- Scheme -> Emacs Lisp
- Shen -> Arc
- TXL -> XML
- Xojo -> REALbasic
- ZIL -> Forth

# Experiment 4

Assets:

- `experiments/features_ex3.py`
- `experiments/train_model_ex4.py`

Metrics:

- Accuracy: 61% (LogisticRegression)
- Accuracy: 56% (LinearSVC)
- Accuracy: 56% (SGDClassifier)
- Accuracy: 56% (RandomForestClassifier)
- Accuracy: 53% (BernoulliNB)
- Accuracy: 53% (MultinomialNB)
- Accuracy: 50% (Perceptron)

## Features

Same feature set as Experiment 3.

## Model

Expanded model sweep: LogisticRegression, BernoulliNB, MultinomialNB, LinearSVC, Perceptron, SGDClassifier (log_loss), RandomForestClassifier.

# Experiment 5

Assets:

- `experiments/features_ex5.py`
- `experiments/train_model_ex5.py`

Metrics:

- Accuracy: 99.934%
- Feature extraction speed: ~12,000 items/second
- Train time: ~6s
- Feature function size: ~7,600 chars
- Dataset: CodeSearchNet (6-language HF export)

## Features

Simplified, language-focused features for the six CSN languages, plus shared keyword groups and a few regex hints.

## Model

LogisticRegression

# Experiment 6

Assets:

- `experiments/features_ex6.py`
- `experiments/train_model_ex6.py`

Metrics:

- Accuracy: 86.471%
- Feature extraction speed: ~200 items/second
- Train time: ~0:00:00.120026
- Dataset: The Stack V2 (34-language subset)

## Features

Expanded feature set for The Stack V2 subset, including:

- Language-specific keyword/symbol cues for C/C++/Rust/Go, PHP, Perl, Lua, R, VB.NET.
- Diff/Makefile/Markdown/YAML/INI/XML structural hints.
- CSS/Less/SCSS markers and GraphQL signals.
- Import statement patterns (JS/TS vs Python).

## Model

LogisticRegression

# Experiment 7

Assets:

- `experiments/train_model_ex7.py`

Metrics:

- Accuracy: 79%
- Speed: 3ms
- Dataset: The Stack V2 (3,400 samples)

## Features

Uses the existing boolean feature set (0/1 per feature).

## Model

Centroid classifier: mean feature vector per language, predict by nearest centroid.

# Experiment 8

Assets:

- `experiments/features_ex8.py`

Metrics:

- Dataset: The Stack V2 (34-language subset)
- Data size: ~0.5 GB, ~100 snippets per language (large snippets)

## Features

Tried unsupervised substring mining to surface predictive tokens without hand-written priors:

- Character n-grams and word tokens, scored by best-language vs rest ratio.
- Inequality scoring (Gini) to rank substrings.
- TF-IDF variants (per-snippet, then aggregated by language).

## Outcome

Dominated by rare, corpus-specific tokens (e.g., artifacts from a single file) rather than general language markers, even at ~0.5 GB. Concluded this approach is not reliable at this dataset size; next step is curated language keyword lists.

# Experiment 9

Assets:

- `experiments/features_ex9.py`
- `experiments/train_model_ex9.py`

Metrics:

- Accuracy: 73% (binary + scaling)
- Features: ~600

## Features

Binary token presence with scaling reached 73%. Also tried scaled per-token frequency (count per snippet length), which only reached 65% and requires storing scaler parameters at inference time.

## Model

LogisticRegression with StandardScaler.

# Experiment 10

Assets:

- `experiments/features_ex10.py`
- `experiments/train_model_ex10.py`

Metrics:

- Accuracy: 92.6% (up from 80% with snippet_limit=10)
- Features: 666 😈
- Dataset size: 153,609 (up from 3,400)
- Train time: 29 seconds

Highlight JS gets 70% on the same dataset, taking 8m24s

## Features

Used smaller snippets with `get_stack_data(snippet_limit=10)` (max 10 lines). Harder to predict per snippet, but increased dataset size without increasing total text. Added more keywords.

## Model

LogisticRegression with StandardScaler.

# Experiment 11

Assets:

- `experiments/features_ex11.py`
- `experiments/train_model_ex11.py`

Metrics:

- Accuracy: 94.4% (headline; after digit-heavy filtering + two extra features)
- Top-2 accuracy: 95.7%
- Top-5 accuracy: 98.6%
- Top-10 accuracy: 99.4%
- Dataset: The Stack V2 (34-language subset), 10,200 source rows (300 per language)
- Snippet limit: 10 (448,752 snippets; 435,780 after digit-heavy filtering)
- Features: 666 tokens (same base set as Experiment 10, plus `": "` and `";\n"`)

## Features

Same feature set as Experiment 10, with two additional "code-y" tokens (`": "` and `";\n"`). Filtering out digit-heavy snippets (digit ratio > 0.9) improved accuracy.

Feature-count tradeoff (subset results):

- 25: 74.9%
- 50: 86.0%
- 100: 89.0%
- 250: 91.4%
- 300: 91.7%
- 400: 91.8%
- 500: 91.9%
- 550: 91.9%
- 666: 91.9%

Full dataset accuracy with 400 features: 92.9%.

## Model

LogisticRegression.

Model comparisons on 10% subset:

- LogisticRegression: 92.1%
- LogisticRegression (fit_intercept=False): 88.6%
- LogisticRegression (fit_intercept=False, X = X - 0.5): 91.6% (4x slower)
- RandomForestClassifier: 92.8% (not viable for browser goal)
- HistGradientBoostingClassifier: 87%

# Experiment 12

Spaces after some keywords. E.g. `"fun"` matches `"func"` and `"function"`, so `"fun "` gives a stronger signal.

Before: 94.4% on 681 features and 10% subset
After: 95.3% on 697 features and 10% subset
And 96.4% on the full dataset

Even after sorting features by 'importance' (coef size), picking only 5 of these (2, 7, 15, 127, 181) is 97.9% accuracy, while 681 of them get to 99.1%

Tried using permutation to calculate importance of features. The top 10 features from that process has an accuracy of 90.8%.

Tried greedy (selecting the best next feature from top 200). Top 10 features gets to 94.2%, top 20 gets to 98.7%. So much better than cord 'importance', but less good that the hand-picked set.

Alas all this was measuring accuracy, but the dataset, after breaking up into 10-line snippets, is 83% WebAssembly, so accuracy is not valid. From now on will report F1.

With `get_stack_data(snippet_limit=10, subset=0.01)`

- F1 is 87.3% (my way)
- F1 is 33.9% with Highlight.js

OK, better train and perf tuning setup, works like this:

- Train the model, save the coef sorted by importance (based on mean size)
- In `inference.py` you can
  - view the n/F1 curve, comparing 'top-n' features with the resulting F1 score.
  - find the value of `n` within some `delta` of the best score. E.g. you only need 438 features, not 697, to get within 1% of the best score.

# Experiment 13

Testing on just 6 languages to align with https://huggingface.co/huggingface/CodeBERTa-language-id
`["Go", "Java", "JavaScript", "PHP", "Python", "Ruby"]`
And full snippets.
This gets 99% F1 and only needs 74 features, so is < 2kb

# Experiment 14

Cleaned The Stack dataset (see `create_stack_snippets.py`). Split into n-line snippets, then deleted junk (~10k out of ~430k items).
Makes not much difference:

- With 10-line snippets (420k rows): F1 is 88.6% with 456 features
- With 20-line snippets (210k rows): F1 is 93.9% with 398 features

# Experiment 15 – Bag of words

Assets:

- `experiments/train_model_ex15.py`

Replace the `generate_features` logic with a `CountVectorizer`.

On the 10-line snippets dataset (150k rows):

| Method              | Runtime | F1    | Size (KB)  |
|---------------------|---------|-------|------------|
| `generate_features` | 1 min   | 87.3% | 20 KB      |
| `CountVectorizer`   | 15 min  | 92.4% | 157,000 KB |

This generates 477,842 features, hence the slowness. We can still sort and use just a subset of these. Taking just the first 5,000 features gets an F1 of 87.5%. 10,000 features gets 89.7% (diminishing returns). In other words, this method can get better results, but needs 10x more features to match the results of `generate_features`, and much more to get further ahead.  
