# Mechanisms of lexical acquisition and sense addition in language models

[日本語](README.md)

Code for experiments that manipulate the vocabulary of OLMo 2 (1B / 7B / 13B) through continued pretraining, to study how language models acquire and process new words and new senses. The main experiment runs on 7B; 1B and 13B are used to compare model sizes.

There are three experiments.

- **Unknown-word acquisition**: in passages that contain a known noun (e.g. pasta), replace the noun with a string the model has never seen (an unknown word, e.g. resonunt), insert the passages into the training data, and train. Measure, each with its own indicator, the order in which the unknown words acquire these three properties:
  - Form familiarity: the unknown word's second and later tokens become easier to predict
  - Meaning: the unknown word's hidden states move closer to those of the noun it replaced
  - Integration as a word: the unknown word's hidden states come to be judged as those of a real word (judged by a classifier, a probe, that tells from hidden states whether a word is real)
- **Sense addition**: in passages that contain a known word (e.g. grape), replace it with another known word (the target word, e.g. car) and train, so that the target word gains a sense. The added senses are of three kinds: a noun of the same category as the target word (a category is a meaning class such as food or animals), a noun of another category, and an adjective. Depending on the run (one continued-pretraining job), the target word gains 1, 2 or 4 senses. Identify the attention heads that select a sense from context, and the attribute each of them judges (the noun's category or the part of speech).
- **Dependence of unknown words on category-judging heads**: no further training. In the unknown-word acquisition models, disable the heads, identified in sense addition, that judge a noun's category (replace the component of the head's output along the category directions with its mean), and compare how much the judgment of the unknown word's category (the category of the noun it replaced) drops with the drop for the noun itself. This tests whether, at the same level of acquired meaning, unknown words that are more integrated as words rely more on the same heads as known words to judge their category.

The evaluation uses tasks that ask the model to choose an answer from options. There are two kinds: sense identification (choose the word with the same meaning as the word in the sentence) and attribute judgment (choose the word's category or part of speech).

## Requirements

- Python 3.12, [uv](https://docs.astral.sh/uv/), NVIDIA GPUs (driver supporting CUDA 13.0), internet access
- LLM (large language model): Qwen3-Next-80B (a model separate from OLMo 2), served with vLLM (server software that runs LLMs), generates word candidates and judges passages (256-token excerpts of text that contain a word). Below, "the LLM" means this model
- GPUs (96 GB): 2 to train 7B (1 for 1B, 4 for 13B), 2 for the LLM server, 1 for the other jobs that run OLMo 2
- Disk
  - Models: about 160 GB (the three OLMo 2 models and the LLM)
  - Data: about 50 GB
  - Trained weights: about 3.6 TB (by default each run saves four checkpoints, at 25 / 50 / 75 / 100% of training; the checkpoints can be changed under `training` in `settings.yaml`, and final weights only take about 0.9 TB)
  - State for resuming training (such as the optimizer state): up to about 165 GB (one 13B run; removed when the run ends)

## Setup

```bash
uv sync
uv run python -m nltk.downloader wordnet
```

- To send training logs to W&B (Weights & Biases, a web service that records experiments), run `uv run wandb login` beforehand. Otherwise, run `export WANDB_MODE=disabled`.
- Before the commands marked `# LLM`, start the LLM server in a separate terminal and leave it running. It takes a few minutes to start, so wait until `curl http://localhost:8000/v1/models` responds. Put the server's URL (for example `http://localhost:8000/v1`) in `llm.server_url` in `settings.yaml`.

```bash
CUDA_VISIBLE_DEVICES=0,1 uvx --from vllm==0.32.0 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --tensor-parallel-size 2 --port 8000
```

## Conventions

- Commands
  - Run them in bash from the top directory of the repository. They have the form `uv run lexical-acquisition-mechanisms <stage> <action>`; add `--help` at the end to see what a command does.
  - All settings are in `settings.yaml`.
  - Commands skip work whose inputs and settings have not changed since they last ran (add `--recompute` to redo it). If a command stops, fix what its message says and rerun from the command it names.
  - Steps 2–5 start by putting the model and other values in shell variables. In a new terminal, rerun `export WANDB_MODE=disabled` (if you do not send logs to W&B) and the variable lines at the start of the step before continuing.
- GPUs and the LLM server
  - Commands marked `# GPU` run OLMo 2 on GPUs; commands marked `# LLM` use the LLM server. Choose GPUs with `CUDA_VISIBLE_DEVICES`.
  - No command uses both. To use the GPUs the LLM server is on, stop the server first.
  - "The pretrained models" means the published OLMo 2 weights (before any continued pretraining).
- CSV files for the manual checks
  - Before a command with `--review-csv`, fill in by hand the `approved` column of the CSV it takes (1 if it fits, 0 if not; the criteria are given in each step). The command stops if a row is empty.
  - Commands that write these CSV files keep the `approved` values already filled in and add new rows empty. Only the empty rows need filling.
- Continued pretraining
  - `training write-run-settings` writes one settings file per run to `outputs/runs/<experiment>/<model>/`, and `training run` trains one of them.
  - To use several GPUs, it is launched with torchrun (PyTorch's command that starts one process per GPU), which runs this repository's Python package, `lexical_acquisition_mechanisms`. `--nproc-per-node` is the number of GPUs to use.
  - A run takes a long time, so run it inside tmux (software that keeps commands running after the terminal closes) or similar. A run that stopped continues from where it stopped when the same command is run again.
  - `training run` warns when the perplexity (how hard the next token is to predict) on the validation data (the ~10M tokens right after the base data, not used for training) changes by more than 1% from before training (training continues). Investigate and report the cause for any run that warns.
- Datasets are written to `data/`; trained weights, result tables and figures to `outputs/`.

## Procedure

### 1. Prepare the datasets

#### 1-1. Base data

```bash
# From the settings of the first stage of OLMo 2 pretraining (stage 1, most of the pretraining),
# reproduce the order of the OLMo 2 pretraining data, and download the first ~300M tokens
# (base data) and the ~10M tokens right after them (validation data)
uv run lexical-acquisition-mechanisms base-data download
```

#### 1-2. Words used in the experiments

```bash
# Noun and target-word candidates: the LLM lists countable concrete nouns for each of six
# categories (provisionally: food; animals; vehicles; clothing; furniture and household items;
# buildings). Keep the nouns that WordNet (a lexical database of English) puts in that category
# only. Give each its token count, its frequency (number of occurrences in the OLMo 2 pretraining
# data and in the Pile, a public dataset of English text), and the share of its occurrences in
# the Pile that use the sense of that category (judged by the LLM)
uv run lexical-acquisition-mechanisms words generate-noun-candidates                # LLM

# Adjective candidates: the LLM lists adjectives and rates how strongly each is associated with
# each category. Give each its token count, its frequency, and the share of its occurrences in
# the Pile that use it as an adjective (judged by the LLM)
uv run lexical-acquisition-mechanisms words generate-adjective-candidates           # LLM

# Unknown-word candidates: collect strings made by changing 1 or 2 letters of a real word that
# belongs to none of the six categories (spelling close to a real word), and pronounceable
# nonwords (strings that are not English words) from the ARC Nonword Database (spelling far from
# real words). Keep those that are 2 or 3 tokens long and never appear in the OLMo 2 pretraining
# data or the Pile
uv run lexical-acquisition-mechanisms words generate-unknown-word-candidates

# Choose the words used in the experiments from the candidates (with spare candidates for
# replacements)
#   72 nouns: one-token words often used in the sense of their category, 12 per category
#     (with similar frequencies across categories, and only words expected to yield enough passages)
#   18 target words: from the noun candidates other than the 72 nouns, one each of 1, 2 and 3
#     tokens per category
#   72 adjectives: one-token words mostly used as adjectives and only weakly associated with
#     every category
#   72 unknown words: 24 each of 1, 2 and 3 tokens. The 2- and 3-token ones are drawn at random,
#     12 close and 12 far in spelling, from candidates whose first token overlaps no other word.
#     The 1-token ones get unused token IDs
#   natural words: real words the experiments do not manipulate (nouns and adjectives used as task
#     examples, to train probes and as references for the indicators)
uv run lexical-acquisition-mechanisms words select

# For every pair of a target word and a noun (18 × 72 pairs, plus the spare candidates),
# the LLM rates how close their meanings are on a 7-point scale
uv run lexical-acquisition-mechanisms words rate-similarity-with-llm               # LLM

# Choose the neutral contexts (sentence openings that say nothing about a word's meaning, e.g.
# "Do you know about the"). On the three pretrained models, compare the probabilities of each
# category's natural nouns right after each candidate, drop the candidates that make one category
# easier to predict, and keep 20
uv run lexical-acquisition-mechanisms evaluation-data select-neutral-contexts       # GPU

# For the same pairs, compute on the three pretrained models the cosine similarity of the hidden
# states of the words placed after the 20 neutral contexts and with no context. Check that
# same-category pairs rank high and different-category pairs rank low, and write
# data/words/review.csv (one word per row) for the manual check
uv run lexical-acquisition-mechanisms words measure-hidden-state-similarity        # GPU
```

**By hand**: fill in the `approved` column of `data/words/review.csv` (1 if the word fits the role, category and sense written in its row, 0 if not).

```bash
# Read approved. If any word is 0, write it to data/words/excluded.csv and stop. Otherwise,
# write the word list that the later commands use (word, role, category, sense, tokens)
uv run lexical-acquisition-mechanisms words finalize --review-csv data/words/review.csv
```

- `words select` stops if a category does not have enough words that meet the conditions. Then change the categories under `words` in `settings.yaml` and rerun from `words generate-noun-candidates`.
- `evaluation-data select-neutral-contexts` stops if fewer than 20 contexts remain. Then add sentence openings of the same form to the candidates under `evaluation` in `settings.yaml`, and rerun this command.
- If `words finalize` stops, rerun the commands in order from `words select` (`words select` replaces the dropped words with spare candidates and keeps the other words).

#### 1-3. Passages

```bash
# Read the Pile from the start and cut out 256-token passages that contain a chosen word
# (passage candidates), up to three times the number needed for each word (e.g. for a noun,
# 10,000 for training and 500 for evaluation). Only one form of each word is used (with a leading
# space, lowercase, singular), and passages that contain another word the experiments manipulate
# are not used
uv run lexical-acquisition-mechanisms passages collect

# Remove near-duplicate candidates, and split them into training and evaluation by source document
uv run lexical-acquisition-mechanisms passages remove-duplicates-and-split

# For each occurrence of the word in a candidate, the LLM judges whether it is used in its
# intended sense (for a noun, the sense of the category it was chosen for): not part of an idiom
# or a proper name, and, for an adjective, used as an adjective
uv run lexical-acquisition-mechanisms passages judge-word-sense                     # LLM

# For each noun, adjective and target word, sample 20 candidates that passed and write them to
# data/passages/review.csv (one passage per row) for the manual check
uv run lexical-acquisition-mechanisms passages sample-for-review
```

**By hand**: fill in the `approved` column of `data/passages/review.csv` (1 if the word is used in the sense written in its row, 0 if not).

```bash
# For each word, compute how precise the LLM's judgments are (the share marked 1 by hand). If the
# precision and the number of passages are enough, write the candidates that passed as the
# training and evaluation passages, as many as needed
uv run lexical-acquisition-mechanisms passages finalize --review-csv data/passages/review.csv
```

- `passages finalize` stops if the precision is too low for some word. Then fix the instructions to the LLM (`llm_prompts/`) and rerun from `passages judge-word-sense`.
- `passages finalize` stops if some word has too few passages, after writing it to `data/words/excluded.csv`. Then rerun the commands in order from `words select`.

#### 1-4. Assignments and the unknown-word acquisition training data

```bash
# Unknown-word acquisition: make 9 groups of 8 unknown words and 8 nouns, and for the 8 runs decide
# which noun each unknown word replaces and how many times it is inserted: 0, 1, 10, 100, 1,000,
# 3,000, 5,000 or 10,000 times. Across the 8 runs, each unknown word gets each count once and each
# of the 8 nouns in its group once
uv run lexical-acquisition-mechanisms assignments build --experiment unknown-word-acquisition

# Sense addition: for each of the 9 runs (3 kinds of added sense × 1, 2 or 4 senses), decide
# which nouns' or adjectives' senses each target word gains
uv run lexical-acquisition-mechanisms assignments build --experiment sense-addition

# Following the assignments, replace the noun in its passages with the unknown word, insert them
# into the base data, and build the training data (token sequences) of the 9 runs (the 8 runs and
# a control run in which nothing is inserted). Also write the check results (such as the number
# of inserted passages) and some of the passages, decoded
uv run lexical-acquisition-mechanisms training-data build --experiment unknown-word-acquisition
```

- After `training-data build`, look at the check results and the decoded passages in `data/training/<experiment>/<run>/` (also in step 4).

#### 1-5. Evaluation data

```bash
# From the passages of the natural words, extract one sentence containing the word (used as task
# examples, to train probes and for checks)
uv run lexical-acquisition-mechanisms evaluation-data extract-natural-word-sentences

# Build the words for the unknown-word indicators: real words and nonwords (strings made by
# swapping the second and later tokens between real words) to train the probes, and, as the
# reference for the meaning indicator, the nouns split artificially into 2 or 3 tokens
# (e.g. pas|ta, where | separates tokens)
uv run lexical-acquisition-mechanisms evaluation-data build-probe-and-normalization-words

# Generate candidate minimal pairs (two sentences that differ only in one cue word, so that the
# target word takes a different sense, e.g. "She peeled the car." / "She parked the car.") for
# each pair of the original sense and one added sense, from templates and the LLM
uv run lexical-acquisition-mechanisms evaluation-data generate-minimal-pairs         # LLM

# Build the task questions (sense identification and attribute judgment, with examples shown
# before each question). The LLM proposes paraphrase words for the target words' original senses
# (e.g. car → automobile)
uv run lexical-acquisition-mechanisms evaluation-data build-task-questions           # LLM

# On the three pretrained models, check the following. In each sentence of a minimal pair, is the
# word that already has the sense the cue points to more likely in the blank? Can the tasks be
# solved for natural words? Is each paraphrase word chosen for its original word? Sample some of
# the minimal pairs that pass and write them to data/evaluation/minimal_pairs_review.csv for the
# manual check
uv run lexical-acquisition-mechanisms evaluation-data check-with-pretrained-models   # GPU
```

**By hand**:

1. Decide the number of examples shown before each question (provisionally 3) from the accuracy on natural words reported by `evaluation-data check-with-pretrained-models`, and write it under `evaluation` in `settings.yaml`. If it changes, rerun from `evaluation-data build-task-questions`.
2. Fill in the `approved` column of `data/evaluation/minimal_pairs_review.csv` (one minimal pair per row; 1 if, in each sentence, the target word's meaning is fixed to the sense written in the row, 0 if not).

```bash
# Drop the minimal pairs with approved 0, and write the minimal pairs and task questions used in
# the evaluation
uv run lexical-acquisition-mechanisms evaluation-data finalize \
  --review-csv data/evaluation/minimal_pairs_review.csv
```

- For a model that cannot solve a task for natural words, the commands that use that task only record that and finish.

### 2. Pilot (7B)

Trial training (two runs) to settle the training settings. `run-1` is the first unknown-word acquisition run; `control` is the control run, in which nothing is inserted (it trains on the base data only).

```bash
MODEL=OLMo-2-1124-7B
export CUDA_VISIBLE_DEVICES=0,1   # GPUs to use (ones the LLM server is not using)
NUMBER_OF_GPUS=2                  # number of GPUs to train on (the number chosen above)

# Write one training settings file per run (model, training data, learning rate and so on)
uv run lexical-acquisition-mechanisms training write-run-settings \
  --experiment unknown-word-acquisition --model "$MODEL"

# Continue pretraining for the two runs
uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
  training run --run-settings "outputs/runs/unknown-word-acquisition/$MODEL/run-1.yaml"     # GPU
uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
  training run --run-settings "outputs/runs/unknown-word-acquisition/$MODEL/control.yaml"   # GPU

# On the pretrained model, train probes that tell from hidden states whether a word is real, one
# per layer and token count, and report their accuracy on words not used to train them
uv run lexical-acquisition-mechanisms unknown-word-acquisition train-real-word-probes \
  --model "$MODEL"   # GPU
```

- Check the following.
  - In the training log `outputs/checkpoints/unknown-word-acquisition/OLMo-2-1124-7B/<run>/metrics.jsonl`: the perplexity changes by at most 1%, the loss on the inserted passages goes down, and the loss neither jumps suddenly nor becomes not-a-number (NaN)
  - The probes also work on words they were not trained on
- From the results, settle the learning rate, the batch size, the warmup and the initial embeddings of the one-token unknown words, and write them under `training` in `settings.yaml`. Settle the learning rate as a rule that also applies to 1B and 13B (provisionally each model's last learning rate in stage 1 of pretraining).
- If a value changes, redo the two runs from `training write-run-settings`.

### 3. Unknown-word acquisition (7B)

Train the 9 runs (the 8 runs and the control run) and measure the three indicators.

```bash
MODEL=OLMo-2-1124-7B
export CUDA_VISIBLE_DEVICES=0,1   # GPUs to use (ones the LLM server is not using)
NUMBER_OF_GPUS=2                  # number of GPUs to train on (the number chosen above)

# Continue pretraining for the 9 runs (finished runs are skipped)
uv run lexical-acquisition-mechanisms training write-run-settings \
  --experiment unknown-word-acquisition --model "$MODEL"
for run_settings in outputs/runs/unknown-word-acquisition/"$MODEL"/*.yaml; do
  uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
    training run --run-settings "$run_settings" || break   # GPU
done
```

After all runs have finished training, run the following.

```bash
# Train the probes that tell whether a word is real (skipped if trained in the pilot)
uv run lexical-acquisition-mechanisms unknown-word-acquisition train-real-word-probes \
  --model "$MODEL"   # GPU

# On each run's trained model, place the unknown words after the 20 neutral contexts and with no
# context, and measure the three indicators at each layer: form familiarity (how predictable the
# second and later tokens are), meaning (cosine similarity with the hidden states of the noun it
# replaced) and integration as a word (the probability that the probes judge it a real word).
# Also record the answers to the tasks
uv run lexical-acquisition-mechanisms unknown-word-acquisition measure-indicators \
  --model "$MODEL"   # GPU

# Normalize the three indicators so that unknown words inserted 0 times are at 0 and known words
# at 1, fit curves against the number of insertions, and test the order in which the three are
# acquired
uv run lexical-acquisition-mechanisms unknown-word-acquisition analyze-acquisition-order \
  --model "$MODEL"

# Make figures of how the indicators change, and tables comparing them by token count, spelling
# closeness and model size
uv run lexical-acquisition-mechanisms unknown-word-acquisition make-report
```

- Decide K, the number of passages inserted per added sense in sense addition: in the meaning curves in `outputs/figures/unknown-word-acquisition/`, pick the count beyond which more insertions barely raise the curve, and write it under `sense_addition` in `settings.yaml` with the reason.

### 4. Sense addition (7B)

Train the 11 runs (3 kinds of added sense × 1, 2 or 4 senses, plus a control run for nouns and one for adjectives) and identify the heads that select a sense.

```bash
MODEL=OLMo-2-1124-7B
export CUDA_VISIBLE_DEVICES=0,1   # GPUs to use (ones the LLM server is not using)
NUMBER_OF_GPUS=2                  # number of GPUs to train on (the number chosen above)

# Add adjective passages to match K (skipped if there are already enough)
uv run lexical-acquisition-mechanisms passages collect
uv run lexical-acquisition-mechanisms passages remove-duplicates-and-split
uv run lexical-acquisition-mechanisms passages judge-word-sense                     # LLM
uv run lexical-acquisition-mechanisms passages sample-for-review
```

**By hand**: fill in the `approved` column of the rows added to `data/passages/review.csv` (same criteria as in 1-3).

```bash
# As in 1-3, check each word's judgment precision and number of passages, and write the added
# passages
uv run lexical-acquisition-mechanisms passages finalize --review-csv data/passages/review.csv
```

- If words are dropped here, rerun the commands in order from `words select`, then return to the start of step 4 (`words select` replaces only the dropped words with spare candidates). If this rebuilds the unknown-word acquisition training data (`training-data build --experiment unknown-word-acquisition` is not skipped), redo step 3 for 7B as well.

```bash
# Build the training data: insert into the base data K passages of each of the 72 nouns for the
# noun runs and the noun control run, and K passages of each of the 72 adjectives for the
# adjective runs and the adjective control run. Replace the word with the target word only in the
# passages of the assigned words (nothing is replaced in the control runs)
uv run lexical-acquisition-mechanisms training-data build --experiment sense-addition

# Continue pretraining for the 11 runs
uv run lexical-acquisition-mechanisms training write-run-settings \
  --experiment sense-addition --model "$MODEL"
for run_settings in outputs/runs/sense-addition/"$MODEL"/*.yaml; do
  uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
    training run --run-settings "$run_settings" || break   # GPU
done
```

After all runs have finished training, run the following.

```bash
# Measure the sense identification accuracy on each sentence of the minimal pairs, and compare it
# with the control run in which the same passages were inserted without replacement (senses that
# are not told apart are left out of the later analyses)
uv run lexical-acquisition-mechanisms sense-addition measure-sense-accuracy \
  --model "$MODEL"   # GPU

# While processing one sentence of a minimal pair, replace the output of each attention head and
# MLP (the feed-forward block of each layer) at the target word with its output on the other
# sentence, measure how far the sense identification answer switches, and subtract the value on
# the control run (as a supplement, also measure it at every position after the target word)
uv run lexical-acquisition-mechanisms sense-addition measure-head-and-mlp-effects \
  --model "$MODEL"   # GPU

# Select the heads and MLPs with significant effects, using confidence intervals and a correction
# for multiple comparisons, and classify them by the kinds of added sense for which they matter
# (all three, only nouns of another category and adjectives, and so on)
uv run lexical-acquisition-mechanisms sense-addition select-significant-heads-and-mlps \
  --model "$MODEL"

# For each selected head, train probes on the pretrained model that tell the noun's category and
# the part of speech from the head's output on natural-word sentences. Check that they work on
# natural words not used to train them and on the minimal pairs in the sense-added model
uv run lexical-acquisition-mechanisms sense-addition train-attribute-probes \
  --model "$MODEL"   # GPU

# While processing an attribute judgment question, replace only the component of the head's
# output along the attribute directions given by the probe weights with its value on the other
# sentence, and compare how often the answer changes with random heads and random directions.
# Save the heads that the probes can read, whose replacement changes the answer, and that matter
# only for the kinds of added sense that differ in that attribute, as the heads that judge the
# attribute, together with the attribute directions
uv run lexical-acquisition-mechanisms sense-addition test-attribute-directions \
  --model "$MODEL"   # GPU

# Make the layer × head maps of the effects, and tables and figures comparing them by the number
# of added senses, token count and model size
uv run lexical-acquisition-mechanisms sense-addition make-report
```

- Decide the number of minimal pairs (provisionally 20 per pair of the original sense and an added sense) from the width of the confidence intervals reported by `sense-addition select-significant-heads-and-mlps`: if they are too wide, increase it and write it under `evaluation` in `settings.yaml`. If it increases, rerun `evaluation-data generate-minimal-pairs` through `evaluation-data finalize`, then `sense-addition measure-sense-accuracy` onwards.

### 5. Dependence of unknown words on category-judging heads (7B)

No further training: this step uses the models trained in step 3 and the heads saved in step 4.

```bash
MODEL=OLMo-2-1124-7B
export CUDA_VISIBLE_DEVICES=0   # GPU to use (one the LLM server is not using)

# On each run's model trained in unknown-word acquisition, solve attribute judgment (category)
# questions on the neutral-context sentences. Disable all the heads that judge a noun's category
# (saved in step 4) at once, and measure how much the probability of the correct category drops
# for the unknown word and for the noun it replaced (dependence = drop for the unknown word /
# drop for the noun). Do the same with random heads and random directions for comparison
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads measure-dependence \
  --model "$MODEL"   # GPU

# Fit a regression that explains the dependence by the normalized meaning and integration values,
# the token count and the spelling closeness (a mixed-effects model that accounts for differences
# between unknown words, nouns and runs). Also compare states defined by thresholds on the
# meaning and integration values
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads fit-regression \
  --model "$MODEL"

# Compare whether the same heads matter when a noun's meaning is given to an unknown word and when
# it is added to a known word. On sense identification questions, replace the output of each head
# at the unknown word with its output for an unknown word inserted 0 times, measure the drop in
# the probability of the correct answer, and compare with the head maps of step 4 by rank
# correlation and the overlap of the top heads
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads compare-head-maps \
  --model "$MODEL"   # GPU

# Make figures and tables of how meaning and integration relate to the dependence
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads make-report
```

- For a model in which step 4 found no heads that judge the category, `dependence-on-category-judging-heads measure-dependence` and `dependence-on-category-judging-heads fit-regression` only record that and finish.

### 6. 1B and 13B

- Work out the 1B and 13B learning rates with the rule settled in the pilot, and write them under `training` in `settings.yaml`.
- Repeat steps 3–5, changing the variables at the start of each step as follows (step 5 has only `MODEL` and `CUDA_VISIBLE_DEVICES`).
  - 1B: `MODEL=OLMo-2-0425-1B`, 1 GPU (one GPU in `CUDA_VISIBLE_DEVICES`, `NUMBER_OF_GPUS=1`)
  - 13B: `MODEL=OLMo-2-1124-13B`, 4 GPUs (four GPUs in `CUDA_VISIBLE_DEVICES`, `NUMBER_OF_GPUS=4`)
- Skip the pilot and the decisions on K and on the number of minimal pairs; use the values settled on 7B. Adding passages in step 4 is skipped, so the LLM server is not needed.

### 7. Collect the results

```bash
# Combine all results into one table, outputs/results/all_results.parquet (a table in the
# Parquet format), and draw the figures in outputs/figures/
uv run lexical-acquisition-mechanisms results collect
```

## Tests

```bash
uv run pytest
```

The tests run every command end to end without a GPU, using a small randomly initialized model and stand-ins that return fixed responses in place of the LLM and the external services (such as the search for word occurrences and the nonword database).
