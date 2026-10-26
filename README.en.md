# Mechanisms of lexical acquisition and sense addition in language models

[日本語](README.md)

Code for experiments that manipulate the vocabulary of OLMo 2 (1B / 7B / 13B) through continued pretraining. The main experiment runs on 7B; 1B and 13B are used to compare model sizes.

- **Experiment A**: train with known nouns replaced by strings the model has never seen (unknown words). Measure the order in which an unknown word acquires these three:
  - Form familiarity: the word's second and later tokens become easier to predict
  - Meaning: the word's hidden states move closer to those of the noun it replaced
  - Integration as a word: the word's hidden states come to be judged as those of a real word
- **Experiment B**: train so that a known word gains the sense of another word, and identify the attention heads that select a sense from context.
- **Connection**: with no further training, disable the experiment B heads in the experiment A models, and measure whether the unknown words depend on those heads.

## Requirements

- Python 3.12, [uv](https://docs.astral.sh/uv/), NVIDIA GPUs (driver supporting CUDA 13.0), internet access
- LLM: Qwen3-Next-80B (a model separate from OLMo 2), served with vLLM, generates word candidates and judges usages
- GPUs (96 GB): 2 to train 7B (1 for 1B, 4 for 13B), 2 for the LLM server, 1 for the other jobs that run OLMo 2
- Disk
  - Models: about 160 GB (the three OLMo 2 models and the LLM)
  - Data: about 50 GB
  - Trained weights: about 3.6 TB (by default each run saves four checkpoints, at 25 / 50 / 75 / 100% of training; about 0.9 TB with final weights only)
  - Resume state during training: up to about 165 GB (one 13B run; deleted when the run ends)

## Setup

```bash
uv sync
uv run python -m nltk.downloader wordnet
```

- To send training logs to W&B (Weights & Biases, a web service that records experiments), run `uv run wandb login` beforehand. Otherwise, run `export WANDB_MODE=disabled` first.
- Before the commands marked `# LLM`, start the LLM server in a separate terminal and leave it running. Put the server's URL (for example `http://localhost:8000/v1`) under `llm` in `configs/lam.yaml`.

```bash
CUDA_VISIBLE_DEVICES=0,1 uvx --from vllm==0.32.0 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 --tensor-parallel-size 2 --port 8000
```

## Conventions

- All settings are in `configs/lam.yaml`. `uv run lam <stage> <action> --help` describes each command.
- Commands marked `# GPU` run OLMo 2 on GPUs (choose them with `CUDA_VISIBLE_DEVICES`). No command uses both the LLM server and OLMo 2, so if there are not enough GPUs, stop the LLM server first.
- Before running a command with `--review`, fill in by hand the `ok` column of the CSV it takes (criteria in step 1).
- Commands skip work whose inputs and settings have not changed since they last ran (add `--force` to redo it). If a command stops, fix what its message says and rerun from the command it names.
- A run is one continued-pretraining job on one training dataset. `train plan` writes one settings file per run to `outputs/runs/<exp>/<model>/`, and `train run` trains them one at a time.
- `train run` warns when ppl (perplexity on the validation data) changes by more than 1% from before training (training continues). Investigate the cause for any run that warns.
- Datasets are written to `data/`; trained weights, result tables and figures to `outputs/`.

## Procedure

### 1. Prepare the datasets

```bash
# Base data (the first ~300M tokens of the OLMo 2 pretraining data, in training order) and the validation data right after it
uv run lam corpus base

# Words: choose the words used in the experiments (nouns, adjectives, unknown words and others) and finalize the word list
uv run lam lexicon nouns          # LLM
uv run lam lexicon adjectives     # LLM
uv run lam lexicon unknown
uv run lam lexicon select
uv run lam evaldata contexts      # GPU
uv run lam lexicon rate           # LLM
uv run lam lexicon similarity     # GPU
uv run lam lexicon finalize --review data/lexicon/review.csv

# Usages: 256-token passages that contain a word, cut from the Pile (a public dataset of English text)
uv run lam usages collect
uv run lam usages dedup
uv run lam usages judge           # LLM
uv run lam usages review
uv run lam usages finalize --review data/usages/review.csv

# Assignments (for each run, which word replaces which, and how many times it is shown) and the experiment A training data
uv run lam assign build --exp a
uv run lam assign build --exp b
uv run lam traindata build --exp a

# Evaluation data
uv run lam evaldata natural
uv run lam evaldata refs
uv run lam evaldata pairs         # LLM
uv run lam evaldata tasks         # LLM
uv run lam evaldata check         # GPU
uv run lam evaldata finalize --review data/eval/pairs_review.csv
```

- Criteria for the `ok` column (1 if it fits, 0 if not)
  - `data/lexicon/review.csv` (one word per row): the word fits the role, category and sense written in its row
  - `data/usages/review.csv` (one usage per row): the LLM's judgment (pass or fail) is correct
  - `data/eval/pairs_review.csv` (two sentences that differ in one word per row): in each sentence, the word's meaning is fixed to the sense written in the row
- When there are dropped words (marked 0, or with too few usages), `lexicon finalize` and `usages finalize` write them to `data/lexicon/excluded.csv` and stop. Then rerun the commands in order from `lexicon select`.
- If `usages finalize` stops because the LLM's judgments are not precise enough, fix the prompts (`configs/prompts/`) and rerun from `usages judge`.
- After `traindata build`, look at the check results and the decoded usages in `data/train/<exp>/<run>/` (also in step 4).
- Decide the number of examples shown before each evaluation question (provisionally 3) from the accuracy on words the experiments do not manipulate, reported by `evaldata check`, and write it to `configs/lam.yaml`. If it changes, rerun from `evaldata tasks`.

### 2. Pilot (7B)

Trial training (two runs) to settle the training settings. `run-1` is the first experiment A run; `control` is the control run, in which nothing is inserted (it trains on the base data only).

```bash
uv run lam train plan --exp a --model 7b
uv run torchrun --nproc-per-node 2 -m lam train run --run outputs/runs/a/7b/run-1.yaml     # GPU
uv run torchrun --nproc-per-node 2 -m lam train run --run outputs/runs/a/7b/control.yaml   # GPU
uv run lam exp-a probe --model 7b   # GPU
```

- `--nproc-per-node` is the number of GPUs to use.
- Check the following, then settle the learning rate, the batch size, the warmup and the initial embeddings of the one-token unknown words, and write them to `configs/lam.yaml`.
  - In the training log `outputs/checkpoints/a/7b/<run>/metrics.jsonl`: ppl changes by at most 1%, the loss on the inserted usages goes down, and training is stable
  - The probe from `exp-a probe` (a classifier that tells from the model's hidden states whether a word is real) also works on words it was not trained on
- If a value changes, redo the two runs from `train plan`.

### 3. Experiment A (7B)

```bash
# Training (finished runs are skipped)
uv run lam train plan --exp a --model 7b
for f in outputs/runs/a/7b/*.yaml; do
  uv run torchrun --nproc-per-node 2 -m lam train run --run "$f" || break   # GPU
done

# Evaluation
uv run lam exp-a probe --model 7b     # GPU
uv run lam exp-a measure --model 7b   # GPU
uv run lam exp-a analyze --model 7b
uv run lam exp-a report
```

- Decide K, the number of usages shown per sense in experiment B: in the meaning curves in `outputs/figures/a/`, pick the count beyond which more usages barely raise the curve, and write it to `configs/lam.yaml` with the reason.

### 4. Experiment B (7B)

```bash
# Add adjective usages to match K (skipped if there are already enough)
uv run lam usages collect
uv run lam usages dedup
uv run lam usages judge           # LLM
uv run lam usages review
uv run lam usages finalize --review data/usages/review.csv

# Training data and training
uv run lam traindata build --exp b
uv run lam train plan --exp b --model 7b
for f in outputs/runs/b/7b/*.yaml; do
  uv run torchrun --nproc-per-node 2 -m lam train run --run "$f" || break   # GPU
done

# Evaluation
uv run lam exp-b accuracy --model 7b  # GPU
uv run lam exp-b map --model 7b       # GPU
uv run lam exp-b stats --model 7b
uv run lam exp-b encode --model 7b    # GPU
uv run lam exp-b causal --model 7b    # GPU
uv run lam exp-b report
```

- If `usages finalize` drops words here, rerun the commands in order from `lexicon select`, then return to step 4.
- Decide the number of minimal pairs (two sentences that differ in one word; provisionally 20 per pair of senses) from the width of the confidence intervals reported by `exp-b stats`, and increase it if they are too wide. If it increases, rerun `evaldata pairs` through `evaldata finalize`, then `exp-b accuracy` onwards.

### 5. Connection (7B)

```bash
uv run lam connect dependence --model 7b   # GPU
uv run lam connect analyze --model 7b
uv run lam connect map --model 7b          # GPU
uv run lam connect report
```

### 6. 1B and 13B

- Work out the 1B and 13B learning rates with the rule settled in the pilot (provisionally each model's final learning rate in the first stage of pretraining, stage 1), and write them to `configs/lam.yaml`.
- Repeat steps 3–5, replacing `7b` (in `--model` and the paths) with `1b` or `13b`. Set `--nproc-per-node` to 1 for 1B and 4 for 13B.
- Skip the pilot and the decisions on K and on the number of minimal pairs; use the values settled on 7B.

### 7. Collect the results

```bash
uv run lam results collect
```

This writes a table of all results, `outputs/results/all.parquet`, and the figures in `outputs/figures/`.

## Tests

```bash
uv run pytest
```

The tests run every command end to end without a GPU, using a small randomly initialised model and stand-ins that return fixed responses in place of the LLM and the external APIs.
