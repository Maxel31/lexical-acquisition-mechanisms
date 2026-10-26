# Mechanisms of lexical acquisition and sense addition in language models

[日本語](README.md)

Code for experiments that manipulate the vocabulary of OLMo 2 (1B / 7B / 13B) through continued pretraining, to study the order in which a new word acquires knowledge and the mechanism that selects a word sense from context.

## Overview

| Experiment | Manipulation | Question |
|---|---|---|
| Exp A | Replace a known noun (e.g. pasta) in the training data with a string the model has never seen (e.g. resonunt) | In which order does a new word acquire form familiarity, meaning, and lexical integration (defined in [Exp A](#exp-a))? |
| Exp B | Replace a known word (e.g. grape) in the training data with another known word (e.g. car), adding a sense to car | When a sense is selected from context, which heads judge which attribute (noun category, part of speech)? |
| Connection | No extra training. Disable the heads found in exp B on the exp A weights | Do unknown words that became integrated as words depend on the same category mechanism as known words? |

- 60 continued-pretraining runs (exp A: 9 runs × 3 models, exp B: 11 runs × 3 models).
- The main experiment uses 7B; 1B and 13B follow the same procedure for the model-size comparison ([Execution order](#execution-order)).

### Terms

| Term | Meaning |
|---|---|
| Run / control run | One continued-pretraining run on one training dataset / a run that replaces nothing (in exp A it inserts nothing) |
| Unknown word | A word the model has never seen, used in exp A. 2- and 3-token ones are strings; 1-token ones are new token IDs (managed names `<new_01>`–`<new_24>`) |
| Near / far | The spelling type of 2- and 3-token unknown words: near is within edit distance 2 of a real word, far is a pronounceable nonword |
| Source / donor | The known word replaced in a passage: a noun in exp A (source) and the word whose sense is added in exp B (donor) |
| Target word | The known word that receives senses in exp B |
| Managed words | The words the experiments manipulate (72 unknown words, 72 nouns, 72 adjectives, 18 target words) |
| Natural words | Real words outside the managed words, used to train probes, for task examples and for checks |
| Role | The use assigned to each word in the lexicon (noun, target word, adjective, unknown word, kind of natural word, reserve candidate) |
| Canonical form | The only form used in passages: leading space, lowercase, singular (e.g. " pasta") |
| Base data / validation data | The first ~300M tokens of the OLMo 2 pretraining data (olmo-mix-1124) in training order / the next ~10M tokens |
| Passage | A 256-token text cut from the Pile that contains the canonical form of the word |
| Exposures | The number of passages of a word inserted into one run's training data |
| Conditions (ii) / (i-a) / (i-b) | The kind of sense added in exp B: a noun of the same category / a noun of another category / an adjective |
| N / K | The number of senses added to one target word in exp B (1, 2, 4) / the number of passages inserted per donor (provisional: 3,000) |
| Neutral context | A sentence opening that does not reveal the word's meaning (e.g. "Do you know about the ___") |
| Minimal pair | Two sentences that differ only in a cue word and select different senses of the target word |
| Task 1 / Task 2 | Few-shot questions answered by comparing the log-probabilities of the option words: sense identification / attribute judgement (category, part of speech) |
| Representation | The residual stream of each layer (the layer's output vector) at a position |
| Probe | A logistic regression that predicts an attribute from representations |
| m / m̂ | The value of an exp A indicator / its normalised value m̂ = (m − m0) / (m1 − m0) (0 is the unknown state, 1 corresponds to a known word) |
| Component | Each attention head of each layer (its per-head output before the output projection) and each MLP |
| Disabling | Replacing the component of a head's output along given directions with its mean value |

## Requirements

- Linux (x86_64), Python 3.12, [uv](https://docs.astral.sh/uv/) 0.8 or later
- NVIDIA GPUs (a driver that supports CUDA 13.0). Rough needs are below. Commands that use the LLM and commands that run OLMo 2 on a GPU are separate and never need both at once.

| Work | GPUs |
|---|---|
| Continued pretraining 1B / 7B / 13B | 96GB × 1 / × 2 / × 4 |
| LLM generation and judging (vLLM) | 96GB × 2 |
| Evaluation and analysis with OLMo 2 | 96GB × 1 |

- Disk
  - Models: the three OLMo 2 models (~86GB) and the LLM (~77GB).
  - Data (base data, passages, training data): ~50GB. The Pile is streamed, not stored.
  - Weights (bf16): 1B 3GB / 7B 15GB / 13B 27GB each. Saving 4 points for all 60 runs takes ~3.6TB (1B 0.24TB, 7B 1.2TB, 13B 2.2TB); final weights only take ~0.9TB.
  - Resume state during training: 1B 18GB / 7B 88GB / 13B 165GB (deleted when the run ends).
- Internet access (for the external data and services below)

## External data and services

| Name | Use |
|---|---|
| OLMo 2 weights (Hugging Face `allenai/OLMo-2-0425-1B`, `allenai/OLMo-2-1124-7B`, `allenai/OLMo-2-1124-13B`; revision `main`, the published final version) | Initial weights |
| Training configs of the first part of OLMo 2 pretraining (stage 1) (`configs/official-0425/OLMo2-1B-stage1.yaml`, `configs/official-1124/OLMo2-{7B,13B}-stage1.yaml` in [allenai/OLMo](https://github.com/allenai/OLMo)) and training data (token arrays on olmo-data.org) | Rebuilding the base data and the validation data |
| The Pile (Hugging Face `EleutherAI/the_pile_deduplicated`, the deduplicated version) | Cutting passages |
| [infini-gram API](https://infini-gram.readthedocs.io/) indexes `v4_olmo-mix-1124_llama`, `v4_olmo-2-1124-13b-instruct_llama` (the full OLMo 2 13B training data, including the mid-training data dolmino-mix-1124), `v4_piletrain_llama` (the Pile before deduplication) | Frequencies, absence checks, sampling occurrences |
| [ARC Nonword Database](http://www.cogsci.mq.edu.au/research/resources/nwdb/nwdb.html) (queried through its web search form) | Candidates for far unknown words |
| WordNet (nltk) | Categories (hypernyms), senses (synsets and glosses), synonyms |
| `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` (served with vLLM) | Generating word candidates, judging passages, rating similarity, generating minimal pairs and paraphrases |
| Weights & Biases (optional) | Training logs |

## Setup

```bash
uv sync
uv run python -m nltk.downloader wordnet
```

Before the commands that use the LLM (marked `# LLM` in the steps), start a vLLM server in a separate environment (put the server URL under `llm` in `configs/lam.yaml`):

```bash
uvx --from vllm==0.32.0 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 --tensor-parallel-size 2 --port 8000
```

To send training logs to W&B, run `uv run wandb login` first. To skip it, run with the environment variable `WANDB_MODE=disabled`.

## Configuration and conventions

- All settings live in `configs/lam.yaml` (models, categories, word counts, thresholds, seeds, LLM, training and evaluation settings; provisional values are changed here too). LLM prompts live in `configs/prompts/`, one file per version. The config is checked for required fields and value ranges when it is loaded.
- Commands are `uv run lam <stage> <action>`; `--config` selects another config file.
- Commands that use a model take `--model {1b,7b,13b}` (checks during data creation use all three models; `train run` takes the model from the run config). Choose GPUs with `CUDA_VISIBLE_DEVICES`. `# GPU` in the steps marks commands that run OLMo 2 on a GPU.
- Every model input outside training starts with `<|endoftext|>` (OLMo 2 was trained on documents separated by this token). The word is given in its leading-space form even without context.
- Each output directory gets a `manifest.json` with the hashes of the config entries the command uses and of its input files. If neither changed, the work is skipped (`--force` redoes it).
- Run names: exp A uses `run-1`–`run-8` and `control`; exp B uses `{ii,ia,ib}-n{1,2,4}` (condition and N) and `control-noun`, `control-adj`.

## Steps

### Building the datasets

#### D1. Base data

```bash
uv run lam corpus base
```

- Rebuilds the training order of the 4,096-token sequences from the OLMo 2 stage-1 training config (data list and seed). The three models' configs have the same data list and seed, so the order is shared.
- Fetches from olmo-data.org only the needed ranges: the first ~300M tokens in training order as the base data and the next ~10M tokens as the validation data.
- Outputs: `data/base/{train,valid}.npy` (tokens), `data/base/sequences.parquet` (source file and position of each sequence)

#### D2. Word candidates

```bash
uv run lam lexicon nouns        # LLM
uv run lam lexicon adjectives   # LLM
uv run lam lexicon unknown
```

- `nouns`: the LLM proposes countable concrete nouns and their category for each of the six categories (provisional: food, animal, vehicle, clothing, furniture/household items, building). Only words that have a WordNet sense (synset) under the category's hypernym and do not belong to more than one category are kept; that sense becomes the "intended sense". The candidates serve the 72 nouns, the target words and the natural nouns.
- `adjectives`: proposes adjectives, each with the strength of its association with each category (LLM rating). The candidates serve the 72 adjectives and the part-of-speech natural adjectives.
- Noun and adjective candidates get the token count of the canonical form, frequencies (occurrence counts in olmo-mix-1124 and the Pile from infini-gram) and the number of WordNet senses. In addition, 50 occurrences sampled at random from the Pile through infini-gram are judged by the LLM, giving a pass rate (for nouns, the share used in the intended sense; for adjectives, the share used as adjectives).
- `unknown`: near candidates are made from real words outside the six categories (adjectives, abstract nouns and so on) by 1–2 character substitutions, insertions or deletions. Far candidates come from the search form of the ARC nonword database (search conditions in `configs/lam.yaml`). Kept are strings that are 2 or 3 tokens with a leading space and never appear in olmo-mix-1124, dolmino-mix-1124 or the Pile (counting word-boundary matches, including capitalised forms).
- Outputs: `data/lexicon/candidates/{nouns,adjectives,unknown}.parquet`

#### D3. Word selection

```bash
uv run lam lexicon select
```

- Selects the following words from the candidates and assigns roles (thresholds in `configs/lam.yaml`). For every role, synonyms of each other and the words in `data/lexicon/excluded.csv` (words removed in D5 and D6) are not used. The number of passing passages is estimated as "frequency in the Pile (scaled to the deduplicated version) × pass rate × share left after the window exclusions".
  - 72 nouns: 12 per category; canonical form of 1 token, pass rate at or above the threshold (near-monosemous), and an estimated 10,500 or more passing passages. Frequency bands are matched across categories.
  - 18 target words: one word per category whose canonical form is 1, 2 and 3 tokens. Same conditions as nouns, but 500 passing passages are enough. They do not overlap the 72 nouns. Frequency bands are matched across the three token counts (towards the rarer multi-token words), and their counts in the base data are recorded.
  - 72 adjectives: canonical form of 1 token, pass rate at or above the threshold, and no strong association with any category. Frequency bands are matched to the 72 nouns, frequent enough for K = 10,000.
  - 72 unknown words: 24 each of 1, 2 and 3 tokens. The 2- and 3-token ones are 12 near and 12 far each, drawn at random from more candidates than needed. Their first token must not match that of any other unknown word or managed word. The 24 one-token words get the first 24 of the unused token IDs (100,278–100,351).
  - Natural words: category nouns (30 per category), part-of-speech nouns and adjectives (60 each), and known nouns for the normalisation references (30 each of 1, 2 and 3 tokens). They do not overlap the managed words.
- Reserve candidates are also selected per role for nouns, adjectives and target words (to replace words that lack passages). A rerun does not change approved words.
- Stops if a category lacks 12 one-token nouns or one target word each of 1, 2 and 3 tokens (replace the category in the config, or add candidates and redo from D2).
- Outputs: `data/lexicon/draft.parquet` (draft lexicon: word, tokens, token count, frequency, category, intended sense (synset and gloss), role; reserve candidates included)

#### D4. Neutral-context check

```bash
uv run lam evaldata contexts    # GPU
```

- For no context and each neutral-context candidate (`configs/lam.yaml`; 20 at the start), averages per category, on each of the three pre-trained models, the log-probability of the category natural nouns right after the context.
- The six category means minus their overall mean form the "difference pattern". Contexts whose pattern lies farther than a threshold from the mean pattern over all contexts are dropped (dropped if this happens on any one model). From the rest, the 20 closest are chosen and shared by the three models. If fewer than 20 remain, add candidates of the same form to the config and rerun.
- Outputs: `data/eval/contexts.parquet` (21 conditions including no context)

#### D5. Sense similarity check and manual review

```bash
uv run lam lexicon rate          # LLM
uv run lam lexicon similarity    # GPU
uv run lam lexicon finalize --review data/lexicon/review.csv
```

- `rate`: for every pair of two words among the nouns and target words (reserve candidates included), the LLM rates similarity on a 7-point scale.
- `similarity`: for the same pairs, computes the cosine similarity of the representations of the three pre-trained models (mean over the 21 neutral-context conditions; middle layer = layer 8 of 1B, 16 of 7B, 20 of 13B) and records it with the ratings as covariates. Checks that same-category pairs fall in the upper half of all pairs and cross-category pairs in the lower half, and writes the review sheet `review.csv`.
- `review.csv` has one row per managed word (word, role, category, intended sense, frequency, similarity values). Write 1 in the `ok` column if the word fits its role, otherwise 0.
- `finalize`: reads `review.csv` and fixes the lexicon. Words marked 0 go to `excluded.csv`; rerun from `select`.
- Outputs: `data/lexicon/{ratings,similarity}.parquet`, `data/lexicon/review.csv`, `data/lexicon/lexicon.parquet` (final lexicon)

#### D6. Passages

```bash
uv run lam usages collect
uv run lam usages dedup
uv run lam usages judge      # LLM
uv run lam usages review
uv run lam usages finalize --review data/usages/review.csv
```

- Required counts (passages that passed judging): nouns 10,000 train + 500 eval; adjectives K train + 500 eval (provisional K = 3,000 until K is fixed); target words 500 eval; natural words 20. The eval passages are built for evaluating the original senses but are not used by the analyses below.
- `collect`: reads the Pile from the start and cuts 256-token windows (passage candidates) around the canonical form of the nouns, adjectives, target words and natural words, with the occurrence near the centre.
  - Windows that contain another form ("Pasta", " pastas", a form without the leading space), another managed word, or an occurrence inside a URL or code are not used. At most 3 windows per document.
  - Collection for a word stops at three times its required count. A rerun continues where the previous one stopped.
- `dedup`: removes near-duplicate candidates (≥80% identical) and splits train/eval by source document (later additions keep the existing split). Eval candidates that overlap the base data are removed.
- `judge`: the LLM judges every occurrence (intended sense, not part of an idiom or proper noun, adjectives used as adjectives), answering pass/fail plus a reason. Judging uses temperature 0 and versioned prompts; stored judgements are reused. Only candidates whose occurrences all pass become passages; judging stops for a word once its required count is reached.
- `review`: for nouns, adjectives and target words, samples 20 passing passages per word into `review.csv` (one row per passage: word, passage, LLM reason). Write 1 in the `ok` column if the judgement is right, otherwise 0.
- `finalize`: reads `review.csv` and records per-word precision (the share of 1 among the 20). It stops if a word is below 95% (19 of 20); revise the prompt and rerun `judge` for that word. The final train passages are shuffled with a fixed seed, and that order is the "fixed order" used from then on.
- Words that do not reach their required count go to `excluded.csv`; the count is never filled with other forms. `select` replaces them with reserve candidates, and D5 and D6 are redone for the new words before D7 (also D4 when a category natural noun is replaced).
- Outputs: `data/usages/{candidates,judgments}/`, `data/usages/review.csv`, `data/usages/precision.parquet` (per-word precision), `data/usages/{train,eval}.parquet` (final passages with occurrence counts)

#### D7. Assignments

```bash
uv run lam assign build --exp a
uv run lam assign build --exp b
```

- Exp A: builds 9 groups of 8 unknown words and 8 nouns (3 groups per token count; each 2- and 3-token group has 4 near and 4 far; each token count has 4 nouns per category). A pair of orthogonal Latin squares of order 8 built over GF(8) fixes "unknown word → noun and exposures" for the 8 runs.
  - Exposure levels are {0, 1, 10, 100, 1,000, 3,000, 5,000, 10,000}. Each unknown word gets each level and each noun of its group once. No noun × level combination repeats, and no noun is used twice in a run.
  - For 2 and 3 tokens, the three 0-exposure words of each run (one per group) include both near and far words.
- Exp B: splits each category's 12 nouns into frames of 4 per target token count, numbered 1–4. Adjectives are assigned 4 per target word at random and numbered 1–4. With category index c = 0–5 and token count j, target word (c, j) gets the donors below.
  - (ii): items 1..N of frame (c, j)
  - (i-a): item n of frame (c+n, j) (n = 1..N; c+n modulo 6)
  - (i-b): items 1..N of its adjectives
  - If an (i-a) pair is not in the lower half in the check of D5 (not in the lower 50% of all target × noun pairs by either the mean cosine of the three models or the LLM rating), the command reorders the frame. It stops if reordering cannot fix it.
- Random choices use the seed in the config.
- Outputs: `data/assignments/{a,b}.parquet`, `data/assignments/a_table.csv` (rows = unknown words, columns = runs, cells = "noun / exposures"), `data/assignments/b_table.csv` (rows = target words, columns = runs, cells = donor lists)

#### D8. Training data

```bash
uv run lam traindata build --exp a
uv run lam traindata build --exp b
```

- Replacement: replaces the source word in a passage (the noun in exp A, the donor in exp B) with the new word (the unknown word or the target word), only where it matches on word boundaries.
  - A preceding a / an is adjusted to the new word's initial sound, and possessive 's is kept. The 1-token unknown words of exp A are inserted by token ID without going through a string, keeping a / an unchanged.
  - The text is re-tokenised and re-cut to 256 tokens around the word. If a passage still contains the source word or had nothing replaced, the command stops with an error. Every replacement is logged.
- Exp A: in each run, for each unknown word, takes the first n train passages (n = its exposures) of its assigned noun in the fixed order, replaces the noun and inserts them (171,999 passages per run). The control run inserts nothing (its tokens equal the base data).
- Exp B: builds a noun pool (72 nouns × K passages) and an adjective pool (72 adjectives × K) from the first K passages in the fixed order. The 6 runs of (ii) and (i-a) and `control-noun` insert the noun pool; the 3 runs of (i-b) and `control-adj` insert the adjective pool. Only the passages of assigned donors are replaced with the target word; all other passages are inserted unchanged.
- Insertion: each passage becomes one document of "256 tokens + `<|endoftext|>`", inserted at a document boundary of the base data without crossing a 4,096-token sequence boundary (the base data after it shifts by the inserted length).
  - Insertion positions are spread evenly over training and shared by all runs of an experiment. For each word, training is split into as many intervals as its exposures, with one passage at a random position in each interval.
  - In exp A, every run has 9 words (one per group) at each exposure level. The positions are fixed per level for 9 words and shared by all runs; each run puts there the passages of the words at that level. In exp B, K positions are fixed per donor index (shared by the noun and adjective pools).
- Checks (automatic): exposures per word (passages and occurrences), no remaining words that should have been replaced, equal total tokens across all runs of an experiment (except the exp A control run), decoded random samples.
- Outputs: `data/train/{a,b}/<run>/` (`tokens.npy`: tokens, `inserted.npy`: inserted positions, `replacements.parquet`: replacement log, `report.json`: check results, `samples.txt`: decoded passages)

#### D9. Evaluation data

```bash
uv run lam evaldata natural
uv run lam evaldata refs
uv run lam evaldata pairs     # LLM
uv run lam evaldata tasks     # LLM
uv run lam evaldata check     # GPU
uv run lam evaldata finalize --review data/eval/pairs_review.csv
```

- `natural`: takes one sentence containing the word from each passage of the natural words (category: 30 words × 20 sentences per category; part of speech: 60 nouns and 60 adjectives × 20 sentences).
- `refs`: builds the references of exp A.
  - For the real-word probe, builds for 2 and 3 tokens real words (WordNet nouns that do not overlap the lexicon) and nonwords (counts in the config). Nonwords are made by swapping the 2nd and later tokens between real words; only those that keep the same token sequence when tokenised normally and do not exist (not in WordNet, infini-gram count at or below a threshold) are used. The 1-token negatives are made by `exp-a probe`.
  - Builds artificial 2- and 3-token splits of the 72 nouns (e.g. `pas|ta`, `p|as|ta`).
- `pairs`: builds minimal-pair candidates for exp B from templates and the LLM (per target word and per pair of the original sense and an added sense, more than needed so that 20 remain after the check).
  - Only the cue word differs, the cue comes before the target word, and both sentences have the same number of tokens.
  - The cue shows the category for (i-a) (e.g. "She peeled the ___." / "She parked the ___."), a distinction within the category for (ii), and the part of speech for (i-b) (e.g. "It was so ___." / "It was the ___.").
- `tasks`: builds the task 1 and 2 items (format below), using the 21 neutral-context conditions for exp A, the minimal-pair sentences for exp B, and natural-word sentences for `check`.
  - The task examples (3) are natural-word sentences, fixed per task.
  - Paraphrases (for the original senses of the 18 target words and for the natural words used in examples and checks) are chosen from WordNet synonyms and LLM candidates: canonical form of 1 token, not a managed word.
  - The two items used for patching (the two sentences of a minimal pair; an exp A unknown word and a 0-exposure string) are identical except for the word, and are checked to have the same token count and word position.
- `check`: on the three pre-trained models, checks the following and records the usable minimal pairs and tasks.
  - Minimal pairs: in the sentence cueing sense X, the word that originally has X (the donor or the target word) put in the blank is more likely than the other word, and the reverse holds for the sentence cueing Y. Pairs passing on all three models are kept, up to 20 per pair of senses.
  - Tasks: whether tasks 1 and 2 are solved significantly above chance for natural words (binomial test). For a model size where a task is not, the analyses that use the task are not run, and this is recorded.
  - Paraphrases: whether each is chosen correctly for the original word.
  - Writes `pairs_review.csv` (one row per pair) with a sample of the kept minimal pairs for a manual spot check. Write 1 in the `ok` column if the cue fixes the sense as intended, otherwise 0.
- `finalize`: reads `pairs_review.csv`, drops the pairs marked 0, and fixes the evaluation data.
- Outputs: `data/eval/{natural,refs,pairs,tasks}.parquet`, `data/eval/check.json`, `data/eval/pairs_review.csv`

Task format (task 1; three examples are shown):

```
Sentence: I put the knife in the drawer.
Question: In this sentence, the word knife means the same as: spoon / blade / bottle / lamp
Answer: blade

Sentence: {sentence to evaluate}
Question: In this sentence, the word {word} means the same as: {option 1} / {option 2} / ...
Answer:
```

- Task 1 (sense identification) options
  - Exp A: the source noun and 3 of the 72 nouns from other categories.
  - Exp B: a paraphrase of the original sense (e.g. car → automobile; not one of the 72 nouns or 72 adjectives) and the N added donors.
  - Natural words (examples and checks): a paraphrase and 3 natural nouns from other categories.
- Task 2 (attribute judgement) question line
  - Category: `Question: In this sentence, the word {word} is a kind of: food / animal / vehicle / clothing / furniture / building` (options follow the category names in the config)
  - Part of speech: `Question: In this sentence, the part of speech of the word {word} is: noun / adjective`
- Options are 1-token words with a leading space, in random order per item (the same order for the two items used for patching).
- The word is not put in quotes (right after a quote the leading space disappears, giving a different token sequence from the trained form).

### Continued pretraining

```bash
uv run lam train plan --exp a --model 7b
uv run torchrun --nproc-per-node 2 -m lam train run --run outputs/runs/a/7b/run-1.yaml   # GPU
```

- `plan`: writes the config files of all runs of an experiment and model (model, training data, seed, learning rate, batch size, warmup, save schedule) to `outputs/runs/<exp>/<model>/`. All runs are trained by passing these files to `train run` one by one.
- `run`: trains one run (`--nproc-per-node` is the number of GPUs). If interrupted, the same command resumes it. Settings shared by all runs:
  - The training data is used in the order of `tokens.npy`, without shuffling (step s uses the 256 sequences starting at sequence s × 256).
  - Learning rate: the final stage-1 value (provisional: 1B 7.4e-5, 7B 6.2e-5, 13B 3.0e-5), raised linearly from 0 over the first 1% of steps and constant afterwards. Sequence length 4,096, batch 256 sequences (~1M tokens per step).
  - Optimizer: AdamW (β = (0.9, 0.95), ε = 1e-8, weight decay 0.1 except embeddings, gradient clipping 1.0) and z-loss 1e-5 (an auxiliary loss that keeps the output logits small), both as in OLMo 2 stage 1. Compute in bf16, weights and optimizer states in fp32. With several GPUs the weights are sharded (FSDP) and gradient checkpointing is used.
  - The seed (for the noise of the new-token initialisation and so on) is the same for all runs, so runs differ only in their training data.
  - In exp A, the input and output embeddings of the 1-token unknown words are initialised to the mean of the existing embeddings plus small noise.
  - The loss is logged overall and for inserted passages only. Validation ppl (perplexity) is measured before training and at each save point; a warning is raised if it changes by more than 1% from before training (training continues; report and investigate). Logs also go to W&B when it is used.
  - Weights at 25 / 50 / 75 / 100% of training are saved in bf16 Hugging Face format (the save points can be changed in the config; the analyses use the final weights).
- Outputs: `outputs/runs/<exp>/<model>/<run>.yaml`, `outputs/checkpoints/<exp>/<model>/<run>/` (`step-*/`: weights, `metrics.jsonl`: loss and ppl, `resume/`: resume state, deleted when the run ends)

### Exp A

```bash
uv run lam exp-a probe --model 7b      # GPU
uv run lam exp-a measure --model 7b    # GPU
uv run lam exp-a analyze --model 7b
uv run lam exp-a report
```

- `probe`: trains the real-word probes on the pre-trained model, per layer and per token count. The inputs are the 21 neutral-context conditions followed by the word, using the representation at the word's last token. Positives are real words and negatives the nonwords of D9 (the 1-token negatives are the 50 unused token IDs not given to unknown words, initialised the same way as in training). Accuracy on words not used for training is reported too (used in the pilot to check that the probes work).
- `measure`: on the final weights of each run, measures the three indicators for the 21 neutral-context conditions. `control` is measured the same way and recorded as the values after training without insertions. The normalisation references (0-exposure unknown words, known nouns, artificially split sources) are measured on the same weights, and task 1 and 2 results are recorded too. The extracted representations are saved in `outputs/activations/a/<model>/`.
  - Form familiarity: surprisal of the k-th token S_k = −log p(t_k | context, t_1, …, t_{k−1}) (k ≥ 2). It comes from the output probabilities, so it has no layer. For 3 tokens, tokens 2 and 3 and their mean are reported, and the mean is used for normalisation and curves. 1-token strings are excluded.
  - Lexical integration: the probability that the probe, applied to the representation of each layer at the word's last token, classifies it as a real word (words used to train the probe are not evaluated).
  - Meaning: cosine similarity between the representation of each layer at the word's last token and the source's representation (same context), averaged over the 21 conditions.
- `analyze`: normalises with m̂ = (m − m0) / (m1 − m0). When m1 − m0 is below a threshold, it warns, marks the value and continues.
  - m0: mean of the 0-exposure words of the same run with the same token count and near / far (no distinction for 1 token). For meaning, m0 is the mean cosine between the 0-exposure words and the source of the word being normalised.
  - m1: for form and integration, the mean of known nouns with the same token count; for meaning, the cosine similarity between the source and its artificial split into the same number of tokens (D9 `refs`; 1-token strings use the 2-token value).
  - Fits, per indicator, the curve of m̂ against exposures with a mixed-effects model (accounting for word, source and run), and reports the exposures at which m̂ reaches 0.5 and a test of the order of the three indicators (bootstrap resampling unknown words), overall and by token count and by near / far. Layers are handled both as "the layer with the largest value" and as "the mean over layers"; the results table states which one is used.
- `report`: draws the trajectories of the three indicators and per-layer heatmaps, and builds comparison tables by factor (token count, near / far, model size). Model sizes are compared among the models whose results are complete.
- Fixing K: from the 7B curve of the meaning m̂, a person picks the exposures at which it saturates (more exposures barely raise it) and writes it with the reason to `configs/lam.yaml`.
- Outputs: `outputs/results/a/<model>/` (`probes/`: probes, `measures.parquet`: before normalisation, `normalized.parquet`: after normalisation, including the m̂ used by the connection, `curves.parquet`, `order_test.json`), `outputs/results/a/compare.parquet`, `outputs/figures/a/`

### Exp B

```bash
uv run lam exp-b accuracy --model 7b   # GPU
uv run lam exp-b map --model 7b        # GPU
uv run lam exp-b stats --model 7b
uv run lam exp-b encode --model 7b     # GPU
uv run lam exp-b causal --model 7b     # GPU
uv run lam exp-b report
```

- The final weights are used, with minimal-pair sentences placed in the tasks. The target-word position is the last token of the word in the Sentence line.
- In a minimal pair (s_X, s_Y), s_X cues sense X and s_Y cues sense Y. Δ(s) is log p(option of sense X) − log p(option of sense Y) in task 1.
- B1 and B2 use task 1. B3 uses, for category, the (i-a) minimal pairs with the category task 2, and for part of speech, the (i-b) minimal pairs with the part-of-speech task 2.

- `accuracy` (B1: checking sense use): task 1 accuracy, measured on the sentences cueing either sense, as the difference from the control run (`control-noun` for (ii) and (i-a), `control-adj` for (i-b)). Target words and senses not significantly above chance (binomial test) are excluded from B2 and B3, and the exclusions are recorded.
- `map` (B2: head map): with Δ_c the Δ obtained when, while processing s_X, the output of component c at the target position is replaced with its output on s_Y, reports the effect e_c = Δ(s_X) − Δ_c and the normalised e_c / (Δ(s_X) − Δ(s_Y)) (1 means a full switch to sense Y).
  - All components are first measured with a gradient approximation (attribution patching); the top 50 and 50 random components are confirmed with actual patching (the agreement between the two is recorded).
  - The control run's effect is subtracted, and both directions (X→Y, Y→X) and all pairs are averaged. As a supplement, a map patching all positions after the target word is made too.
- `stats` (B2): selects significant components with bootstrap confidence intervals over target words and minimal pairs and FDR 5% (multiple-comparison correction over components).
  - Classifies them by the conditions where they work (all: sense selection itself; (i-a) and (i-b): category judgement; (i-b) only: part-of-speech judgement; (ii) only: distinctions within a category; otherwise: described individually).
  - Adopts those whose effect has a positive slope over N and that replicate in both halves of the target words (9 words each with 3 per token count, categories balanced as far as possible; the split is fixed by a seed and checked in both directions).
  - The sense-similarity values and base-data frequencies are attached to the results table as covariates.
- `encode` (B3: encoding): for each adopted head, trains probes on the pre-trained model that predict the category (6 classes) and part of speech (2 classes) from the head output, with natural-word sentences placed in the same task 2 format. Checks that both (a) accuracy on held-out natural words (split by word) and (b) accuracy on the minimal pairs in the sense-added model (predicting the attribute of the cued sense) are significantly above chance. The head outputs are saved in `outputs/activations/b/<model>/`.
- `causal` (B3: causality and specificity): replaces only the components in the subspace spanned by the probe weights (the attribute directions; up to 5 dimensions for category, 1 for part of speech) with their values on s_Y while processing task 2 on s_X.
  - Measures the share of answers that change to the attribute of s_Y, the change in its log-probability, and the change in Δ of task 1, compared against a random head of the same layer and random directions of the same dimension.
  - Also checks specificity (category heads work in (i-a) and (i-b) but not in (ii); part-of-speech heads work in (i-b)). Heads that pass encoding, causality and specificity become "heads that judge the attribute" and are saved with their subspaces.
- `report`: draws layer × head maps and builds comparison tables and figures by N, token count and model size.
- Outputs: `outputs/results/b/<model>/` (`accuracy.parquet`, `maps.parquet`, `components.parquet`, `encoding.parquet`, `heads.parquet`: heads that judge an attribute, `subspaces/`: attribute directions), `outputs/results/b/compare.parquet`, `outputs/figures/b/`

### Connection

```bash
uv run lam connect dependence --model 7b   # GPU
uv run lam connect analyze --model 7b
uv run lam connect map --model 7b          # GPU
uv run lam connect report
```

- `dependence`: on the final weights of each exp A run, replaces the category-direction component of the outputs of the "heads that judge the category" of the same model size (exp B `heads.parquet` and `subspaces/`) with its mean over natural words (disabling; the position is the last token of the word in the Sentence line).
  - The mean is taken on the same weights as the disabling, with natural-word sentences placed in the category task 2 format. With several heads, disabling all of them at once is the main analysis, and per-head values are recorded too. For a model size with no such head, the analysis is not run, and this is recorded.
  - In task 2 over the 21 neutral-context conditions, measures the drop in the log-probability of the correct category (the source's category) for the unknown word (d_unknown) and for the source itself (d_known) (mean over the 21 conditions). Dependence = d_unknown / d_known. Source × run pairs with d_known below a threshold (the heads do not affect even the known word) are excluded.
  - The same operation is applied to a random head of the same layer and random directions of the same dimension. Task 2 accuracy before disabling is recorded too.
- `analyze`: fits, per model size, dependence ~ meaning m̂ + integration m̂ + token count + near / far + (1 | unknown word) + (1 | source) + (1 | run) ((1 | x) is a random intercept per x).
  - 0-exposure unknown words are included. The m̂ values are reduced to one value with the same layer choice as the exp A order test. Near / far gets a "none" level for 1 token.
  - Also compares states defined by whether the meaning and integration m̂ are above or below thresholds (0.3 / 0.5 / 0.7) (e.g. "meaning acquired but not yet integrated") and checks that the conclusion holds.
- `map`: for each unknown word, compares the exp A map (in task 1, the drop in the correct option's log-probability when each head's output at the unknown word is replaced with its output on a 0-exposure string with the same token count; 0-exposure words excluded) with the exp B maps (of the senses whose donor is the unknown word's source; each (ii) and (i-a) run): rank correlation of the per-head effects and overlap of the top heads.
- `report`: draws the relation between the meaning / integration m̂ and dependence (per model size) and builds the table of results under the different thresholds.
- Outputs: `outputs/results/connect/<model>/` (`dependence.parquet`, `regression.json`, `thresholds.parquet`, `maps.parquet`), `outputs/figures/connect/`

### Aggregation

```bash
uv run lam results collect
```

- Collects all results into one long table (experiment, model, run, word, condition, N, metric, layer, head, value) and draws the standard figures (trajectories, per-layer heatmaps, layer × head maps).
- Outputs: `outputs/results/all.parquet`, `outputs/figures/`

## Execution order

1. Build the datasets (D1–D9). Only the exp B training data (`traindata build --exp b`) waits until K is fixed.
2. Pilot: train `run-1` and `control` of exp A on 7B. Fix the learning rate, batch size, warmup and new-token initialisation, write them to `configs/lam.yaml`, and check the ppl change and that the probes work (accuracy on words not used for training). If a value changes, retrain the two runs with the new settings.
3. Exp A: train and evaluate all 7B runs, then fix K.
4. Exp B: if the adjective passages fall short of K + 500, collect and judge more (`usages collect` → `dedup` → `judge` → `review` → `finalize`). Build the training data with `traindata build --exp b`, then train and evaluate all 7B runs.
5. Connection on 7B.
6. Train and evaluate exp A, exp B and the connection on 1B and 13B. The learning rate follows the rule fixed in the pilot for each model (provisionally, each model's final stage-1 value). K and the other settings are those fixed on 7B and are not redetermined.
7. Aggregation: `results collect`.

## Output layout

```
data/
  base/          base data and validation data
  lexicon/       word candidates, lexicon, sense similarity, review sheets
  usages/        passage candidates, judgements, final passages (train / eval)
  assignments/   assignment tables
  train/         training data and check records per run
  eval/          neutral contexts, natural words, references, minimal pairs, tasks 1 and 2
  cache/         LLM, infini-gram and ARC responses
outputs/
  runs/          run configs
  checkpoints/   weights and training logs per run
  activations/   extracted representations and head outputs
  results/       result tables
  figures/       figures
```

## Tests

```bash
uv run pytest
```

Runs every command end to end without a GPU, using a tiny random OLMo 2 model and stand-ins that return fixed responses in place of the LLM and HTTP services.
