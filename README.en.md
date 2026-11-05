# Mechanisms of lexical acquisition and sense addition in language models

[日本語](README.md)

Code for three experiments that manipulate the vocabulary of OLMo 2 (1B / 7B / 13B) through continued pretraining. 7B is the main model; 1B and 13B are used to compare model sizes. Training uses 256-token excerpts of text that contain a word (passages), cut from the Pile (a public English text dataset); a word in them is replaced, and the passages are inserted into the training data. One continued-pretraining job is called a run.

- **Unknown-word acquisition**: train on passages in which a noun (e.g. pasta) is replaced with a string the model has never seen (an unknown word, e.g. resonunt). Measure, each with its own indicator, the order in which the unknown words acquire these three:
  - Form familiarity: the unknown word's second and later tokens become easier to predict
  - Meaning: the unknown word's hidden states move closer to those of the original noun
  - Integration as a word: the unknown word's hidden states come to be judged as those of a real word
- **Sense addition**: train on passages in which a known word (e.g. grape) is replaced with another known word (the target word, e.g. car), so that the target word gains a sense. The added senses are of three kinds: a noun of the target word's category (a meaning class such as food or animals), a noun of another category, and an adjective; depending on the run, the target word gains 1, 2 or 4 senses. Identify the attention heads that select a sense from context, and the attribute each of them judges (the noun's category or the part of speech).
- **Dependence of unknown words on category-judging heads**: no training. In the unknown-word acquisition models, disable the heads, identified in sense addition, that judge a noun's category (replace the component of their output along the category directions with its mean), and compare how much the judgment of the unknown word's category drops with the drop for the original noun.

Besides the unknown words' indicators, the evaluation uses tasks that ask the model to choose an answer from options (sense identification and attribute judgment). A few examples are shown before each question.

## Requirements

- Python 3.12, [uv](https://docs.astral.sh/uv/), internet access
- NVIDIA GPUs (96 GB): for training, 1 for 1B, 2 for 7B and 4 for 13B; 2 for the LLM server
- Disk: about 4 TB (of which the trained weights take about 3.6 TB)

## Setup

```bash
uv sync
uv run wandb login   # only to send training logs to W&B (a web service that keeps experiment logs)
```

Commands marked `# LLM` use an LLM (Qwen3-Next-80B) to generate word candidates or judge passages. While you run them, keep the LLM server running in another terminal (its URL is `llm.server_url` in `settings.yaml`).

```bash
CUDA_VISIBLE_DEVICES=0,1 uvx --from vllm==0.32.0 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --tensor-parallel-size 2 --port 8000
```

## Usage

- Run commands from the top directory of the repository, in the form `uv run lexical-acquisition-mechanisms <stage> <action>` (`--help` prints a description). All settings are in `settings.yaml`.
- Commands marked `# GPU` use GPUs. Choose the GPUs with `CUDA_VISIBLE_DEVICES`. To use the GPUs of the LLM server, stop the server first.
- An action whose inputs and settings have not changed since it last ran is skipped (to redo it, add `--recompute`). If a command stops, fix what it reports, and rerun from the command it names.
- Training runs on several GPUs, so it is started with torchrun (the PyTorch command that starts one process per GPU). It takes a long time, so run it inside tmux (software that keeps jobs running after the terminal closes) or similar. A run that stopped continues from where it stopped when you run the same command again.
- Datasets are written to `data/`; trained weights, results and figures to `outputs/`.

## Steps

### 1. Build the datasets

#### 1-1. Base data and words

```bash
# Download the base data (the first ~300M tokens, in training order, of OLMo 2's pretraining data)
# and the ~10M tokens right after it (the validation data)
uv run lexical-acquisition-mechanisms base-data download

# Make word candidates: the LLM lists nouns and adjectives, and judges how they are used in the Pile
# (the share of uses in the sense or part of speech); unknown words are made from strings that never
# appear in OLMo 2's pretraining data or the Pile
uv run lexical-acquisition-mechanisms words generate-noun-candidates                # LLM
uv run lexical-acquisition-mechanisms words generate-adjective-candidates           # LLM
uv run lexical-acquisition-mechanisms words generate-unknown-word-candidates

# From the candidates, select 72 nouns, 18 target words, 72 adjectives, 72 unknown words,
# and natural words (real words the experiments do not manipulate, used as task examples
# and to train probes)
uv run lexical-acquisition-mechanisms words select

# For each pair of a target word and a noun, the LLM rates how close their meanings are
uv run lexical-acquisition-mechanisms words rate-similarity-with-llm               # LLM

# Select 20 neutral contexts (sentence openings that do not hint at a meaning,
# e.g. "Do you know about the")
uv run lexical-acquisition-mechanisms evaluation-data select-neutral-contexts       # GPU

# For each pair of a target word and a noun, measure their hidden-state similarity in the three
# published OLMo 2 models, and write the review file data/words/review.csv
uv run lexical-acquisition-mechanisms words measure-hidden-state-similarity        # GPU
```

**Manual step**: in the `approved` column of `data/words/review.csv`, enter 1 if the word fits the category and sense in its row, otherwise 0.

```bash
# Write the list of words that the later steps use, from the words with approved = 1
uv run lexical-acquisition-mechanisms words finalize --review-csv data/words/review.csv
```

**Manual step** (if it stopped because some words have 0): rerun from `words select` (other words are chosen in place of the rejected ones).

#### 1-2. Passages

```bash
# Cut passage candidates that contain the selected words from the Pile
uv run lexical-acquisition-mechanisms passages collect

# Remove near-identical candidates, and split the rest into training and evaluation
# by source document
uv run lexical-acquisition-mechanisms passages remove-duplicates-and-split

# The LLM judges, for each occurrence in the candidates, whether the word is used
# in the intended sense
uv run lexical-acquisition-mechanisms passages judge-word-sense                     # LLM

# Sample 20 passed candidates for each noun, adjective and target word,
# and write the review file data/passages/review.csv
uv run lexical-acquisition-mechanisms passages sample-for-review
```

**Manual step**: in the `approved` column of `data/passages/review.csv`, enter 1 if the word is used in the sense in its row, otherwise 0.

```bash
# Compute the precision of the LLM's judgments per word (the share of passed candidates that
# a person also marked 1), and write the passed candidates as the training and evaluation passages
uv run lexical-acquisition-mechanisms passages finalize --review-csv data/passages/review.csv
```

**Manual step** (if `passages judge-word-sense` or `passages finalize` stopped because some words lack passages): rerun as shown. If words were excluded, rerun from `words select` (other words are chosen in their place); otherwise, increase `passages.collection_factor` in `settings.yaml` and rerun from `passages collect`.

**Manual step** (if it stopped because the precision is too low for some words): revise the LLM's instructions (`llm_prompts/`), and rerun from `passages judge-word-sense`.

#### 1-3. Assignments and training data

```bash
# For each run, decide whose passages to insert, which word replaces the original word,
# and (in unknown-word acquisition) how many
uv run lexical-acquisition-mechanisms assignments build --experiment unknown-word-acquisition
uv run lexical-acquisition-mechanisms assignments build --experiment sense-addition

# Build the training data of the 9 unknown-word acquisition runs (8 runs and a control that
# inserts nothing): replace the noun in noun passages with an unknown word, and insert the
# passages into the base data
uv run lexical-acquisition-mechanisms training-data build --experiment unknown-word-acquisition
```

**Manual step**: in `data/training/unknown-word-acquisition/checks.csv`, look at the check results of each run (the number of inserted passages per word, leftover words that should have been replaced, the total number of tokens). Also read `decoded_passages.txt` (randomly sampled inserted passages, with the tokens turned back into text).

#### 1-4. Evaluation data

```bash
# From the natural words' passages, extract the sentences used as task examples and to train probes
# (classifiers that tell a property of a word from its hidden states)
uv run lexical-acquisition-mechanisms evaluation-data extract-natural-word-sentences

# Make the real words and nonwords (strings that are not words) used to train probes, and the known
# words used as the baseline when normalizing the unknown words' indicators (known words become 1)
uv run lexical-acquisition-mechanisms evaluation-data build-probe-and-normalization-words

# Make minimal pairs (two sentences that differ in one cue word, giving the target word
# different senses)
uv run lexical-acquisition-mechanisms evaluation-data generate-minimal-pairs         # LLM

# Make the task questions
uv run lexical-acquisition-mechanisms evaluation-data build-task-questions           # LLM

# In the three published OLMo 2 models, check that the cues of the minimal pairs fix the sense,
# and compute the task accuracy on natural words. Sample some of the pairs that passed,
# and write them to the review file data/evaluation/minimal_pairs_review.csv
uv run lexical-acquisition-mechanisms evaluation-data check-with-pretrained-models   # GPU
```

**Manual step** (if it stopped because some pairs of senses have no minimal pairs): increase `evaluation.minimal_pair_candidate_factor` (the factor for the number of candidates that the LLM makes) in `settings.yaml`, and rerun from `evaluation-data generate-minimal-pairs`.

**Manual steps**:

1. From the task accuracy on natural words that the previous command reports, decide the number of examples shown before each question (provisional: 3), and write it to `evaluation.number_of_task_examples` in `settings.yaml` (if you change it, rerun from `evaluation-data build-task-questions`). A task whose accuracy does not exceed chance is not used for that model.
2. In the `approved` column of `data/evaluation/minimal_pairs_review.csv`, enter 1 if in each sentence the target word's sense is fixed to the sense in its row, otherwise 0.

```bash
# Remove the minimal pairs with approved = 0, and write the minimal pairs and task questions
# used for evaluation
uv run lexical-acquisition-mechanisms evaluation-data finalize \
  --review-csv data/evaluation/minimal_pairs_review.csv
```

### 2. Pilot (7B)

To decide the training settings, train only two runs. In steps 2–5, first run the following each time you open a terminal.

```bash
MODEL=OLMo-2-1124-7B
export CUDA_VISIBLE_DEVICES=0,1   # GPUs to use (if the LLM server uses them, stop it first)
NUMBER_OF_GPUS=2                  # the number of GPUs chosen above
export WANDB_MODE=disabled        # only if you do not send training logs to W&B
```

```bash
# Write a training settings file for each run
uv run lexical-acquisition-mechanisms training write-run-settings \
  --experiment unknown-word-acquisition --model "$MODEL"

# Train run-1 (the first unknown-word acquisition run) and control (the control run
# that inserts nothing)
uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
  training run --run-settings "outputs/runs/unknown-word-acquisition/$MODEL/run-1.yaml"     # GPU
uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
  training run --run-settings "outputs/runs/unknown-word-acquisition/$MODEL/control.yaml"   # GPU

# In the published model, train a probe that tells from hidden states whether a word is real,
# and report its accuracy
uv run lexical-acquisition-mechanisms unknown-word-acquisition train-real-word-probes \
  --model "$MODEL"   # GPU
```

**Manual steps**:

1. In the training logs (`outputs/checkpoints/unknown-word-acquisition/$MODEL/<run>/metrics.jsonl`), check the following.
   - The perplexity (how hard the next token is to predict) on the validation data changes by 1% or less from before training
   - The loss on the inserted passages goes down
   - The loss does not jump suddenly or become NaN (not a number)
2. Check that the probe also works on words it was not trained on.
3. Decide the learning rate, batch size, warmup and the initial embeddings of one-token unknown words (unknown words given new token IDs), and write them under `training` in `settings.yaml` (`learning_rate_by_model`, `batch_size_in_tokens`, `warmup_fraction`, `new_token_embedding_initialization`). Decide the learning rate as a method that gives the values for 1B and 13B in the same way (provisional: the final learning rate of each model's pretraining stage 1). If you change a value, redo step 2.

### 3. Unknown-word acquisition (7B)

```bash
# Train the 9 runs
uv run lexical-acquisition-mechanisms training write-run-settings \
  --experiment unknown-word-acquisition --model "$MODEL"
for run_settings in outputs/runs/unknown-word-acquisition/"$MODEL"/*.yaml; do
  uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
    training run --run-settings "$run_settings" || break   # GPU
done
```

**Manual step**: after all runs have finished training, check in each run's training log (`metrics.jsonl`) that the perplexity changes by 1% or less from before training. For a run that exceeds it, find the cause and report it.

```bash
# Train the probe that tells whether a word is real (skipped if trained in the pilot)
uv run lexical-acquisition-mechanisms unknown-word-acquisition train-real-word-probes \
  --model "$MODEL"   # GPU

# In each run's model, measure the unknown words' three indicators (form familiarity, meaning,
# integration as a word) in each layer
uv run lexical-acquisition-mechanisms unknown-word-acquisition measure-indicators \
  --model "$MODEL"   # GPU

# Normalize the indicators so that unknown words inserted 0 times are 0 and known words are 1,
# and test the order in which the three indicators are acquired
uv run lexical-acquisition-mechanisms unknown-word-acquisition analyze-acquisition-order \
  --model "$MODEL"

# Make figures and tables in outputs/figures/unknown-word-acquisition/
uv run lexical-acquisition-mechanisms unknown-word-acquisition make-report
```

**Manual step**: from the figure of the meaning indicator in `outputs/figures/unknown-word-acquisition/`, take as K (the number of passages inserted per added sense in sense addition) the number beyond which more passages barely raise it, and write K to `sense_addition.passages_per_added_sense` in `settings.yaml`.

### 4. Sense addition (7B)

```bash
# Add adjective passages to match K (skipped if there are enough)
uv run lexical-acquisition-mechanisms passages collect
uv run lexical-acquisition-mechanisms passages remove-duplicates-and-split
uv run lexical-acquisition-mechanisms passages judge-word-sense                     # LLM
uv run lexical-acquisition-mechanisms passages finalize --review-csv data/passages/review.csv

# Build the training data of the 11 sense-addition runs (3 kinds of added sense × 1, 2 or 4 senses,
# and a noun control and an adjective control). The control runs insert the noun or adjective
# passages without replacing any word
uv run lexical-acquisition-mechanisms training-data build --experiment sense-addition
```

**Manual step**: look at `checks.csv` and `decoded_passages.txt` in `data/training/sense-addition/`, as in 1-3.

```bash
# Train the 11 runs
uv run lexical-acquisition-mechanisms training write-run-settings \
  --experiment sense-addition --model "$MODEL"
for run_settings in outputs/runs/sense-addition/"$MODEL"/*.yaml; do
  uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
    training run --run-settings "$run_settings" || break   # GPU
done
```

**Manual step**: after all runs have finished training, check the perplexity of each run as in step 3.

```bash
# In the minimal-pair sentences, measure whether the model tells the target word's senses apart
uv run lexical-acquisition-mechanisms sense-addition measure-sense-accuracy \
  --model "$MODEL"   # GPU

# Swap the output of each head and MLP (the fully connected block of each layer) with its output
# on the other sentence of the minimal pair, and measure the effect on sense selection
uv run lexical-acquisition-mechanisms sense-addition measure-head-and-mlp-effects \
  --model "$MODEL"   # GPU

# Compute confidence intervals of the effects, select the heads and MLPs that meet the selection
# criteria, and classify them by the kinds of added sense for which they have an effect
uv run lexical-acquisition-mechanisms sense-addition select-significant-heads-and-mlps \
  --model "$MODEL"
```

**Manual step**: if the confidence intervals that the previous command reports are too wide, increase `evaluation.minimal_pairs_per_sense_pair` (the number of minimal pairs per pair of senses; provisional: 20) in `settings.yaml`, and rerun from `evaluation-data generate-minimal-pairs` to `evaluation-data finalize` in 1-4 and from `sense-addition measure-sense-accuracy` in this step (training is not redone).

```bash
# In the published model, train probes that tell the noun's category and the part of speech
# from the selected heads' outputs
uv run lexical-acquisition-mechanisms sense-addition train-attribute-probes \
  --model "$MODEL"   # GPU

# Swap the component of a head's output along the attribute directions, identify the heads
# that judge an attribute, and save them
uv run lexical-acquisition-mechanisms sense-addition test-attribute-directions \
  --model "$MODEL"   # GPU

# Make figures and tables in outputs/figures/sense-addition/
uv run lexical-acquisition-mechanisms sense-addition make-report
```

### 5. Dependence of unknown words on category-judging heads (7B)

```bash
# In each run's model trained in step 3, disable the category-judging heads saved in step 4,
# and measure how much the category judgment drops for the unknown word and for the original noun
# (dependence = the drop for the unknown word ÷ the drop for the original noun)
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads measure-dependence \
  --model "$MODEL"   # GPU

# Fit a regression that explains the dependence by the meaning and integration indicators
# and other variables, and also compare states defined by thresholds on the indicators
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads fit-regression \
  --model "$MODEL"

# Compare the effect of each head (the head map) when a noun's meaning is given to an unknown word
# and when it is added to a known word
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads compare-head-maps \
  --model "$MODEL"   # GPU

# Make figures and tables in outputs/figures/dependence-on-category-judging-heads/
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads make-report
```

### 6. 1B and 13B

**Manual step**: with the method decided in step 2, compute the learning rates of 1B and 13B, and write them to `training.learning_rate_by_model` in `settings.yaml`.

Then replace the first three lines of step 2 with one of the following and repeat steps 3–5. Do not run the pilot; use the K and the number of minimal pairs decided on 7B. The extra collection of adjective passages in step 4 is skipped, so the LLM server is not needed.

```bash
MODEL=OLMo-2-0425-1B
export CUDA_VISIBLE_DEVICES=0
NUMBER_OF_GPUS=1
```

```bash
MODEL=OLMo-2-1124-13B
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUMBER_OF_GPUS=4
```

### 7. Collect the results

```bash
# Collect all results into one table, outputs/results/all_results.parquet,
# and make figures in outputs/figures/
uv run lexical-acquisition-mechanisms results collect
```

## Tests

```bash
uv run pytest
```

The tests check that every command runs to the end without GPUs or the LLM server.
