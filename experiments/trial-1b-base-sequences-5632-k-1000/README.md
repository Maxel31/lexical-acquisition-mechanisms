# 試行: 1B、基データ 5,632 系列、K = 1000

語彙獲得の順番の実験（unknown-word-acquisition）と語義の追加の実験（sense-addition）を、小さくしたデータで先に一通り流す試行です。
コマンドは全て、このブランチ（`experiment/trial-1b-base-sequences-5632-k-1000`）を checkout したリポジトリの最上位のディレクトリで実行します。

## 設定（settings.yaml の、本番との違い）

| 設定 | 本番 | 試行 | 理由 |
|---|---|---|---|
| `models` | 1B・7B・13B | 1B だけ | まず 1B で手順を確かめる |
| `training.number_of_base_sequences_used` | null（全て、約3億トークン） | 5632（約 2,300 万トークン） | 結果を早く出すため。1/16（4,608 系列）では、未知語の獲得の用例を置ける文書の境目が足りない |
| `sense_addition.passages_per_added_sense`（K） | 3000（暫定） | 1000 | 同上 |
| `unknown_word_acquisition.exposure_levels` の最大 | 10000 | 9900 | butter の置き換えられる用例が 9,984 件しかない |
| `passages.minimum_judgment_precision` | 0.95 | 0.85 | curtain・arcane・bunk・cot・trailer・viper を含めて流す（本番の前に入れ替える） |
| `evaluation.allow_targets_without_paraphrase` | false | true | coupe・sailboat・recliner に言い換えの語がない（本番の前に入れ替える） |
| `evaluation.allow_sense_pairs_without_minimal_pairs` | false | true | 1B での確認で、最小対が1つもない語義の組（7組）を外して続ける |
| `training.weights_and_biases_project` | lexical-acquisition-mechanisms | lexical-acquisition-mechanisms-trial | W&B で本番と分ける |

## データと出力の置き場所

- `data/` の words・passages・evaluation・base は本番と共通です。
- `data/assignments/unknown-word-acquisition/` は試行の割り当てです（露出回数の最大 9,900）。
  本番の割り当て（最大 10,000）は `data/assignments/unknown-word-acquisition.exposure-top-10000/` に退避してあります。
- `data/training/` は試行の学習データです（実験ごと・run ごとの `tokens.npy`・`passage_starts.npy`・`inserted.parquet`）。
- `outputs/checkpoints`・`runs`・`probes`・`results`・`partial-results` は、`outputs/trial-1b-base-sequences-5632-k-1000/` の中へのシンボリックリンクです。
  - 学習の成果物: `outputs/checkpoints/<実験>/OLMo-2-0425-1B/<run>/final`（モデルの重み）、`metrics.jsonl`（損失・検証データの perplexity）、`weights_and_biases_id.txt`（W&B の run。プロジェクトは lexical-acquisition-mechanisms-trial）
- GPU の作業者のログと仕事の列: `outputs/trial-1b-base-sequences-5632-k-1000/logs/`

## 実行したコマンド（順番）

データセット作成（words・passages・evaluation-data の build-task-questions まで）は、本番の設定（3つのモデル）で実行しました。
ただし、`passages.minimum_judgment_precision` は 0.85、`evaluation.allow_targets_without_paraphrase` は true にしています。
公開モデルでの確認から後は、このブランチの設定（1B だけ）で実行しました。

```bash
# 割り当てと学習データ（試行の設定）
uv run lexical-acquisition-mechanisms assignments build --experiment unknown-word-acquisition
uv run lexical-acquisition-mechanisms training-data build --experiment unknown-word-acquisition
uv run lexical-acquisition-mechanisms training-data build --experiment sense-addition
# 公開モデルでの確認（1B だけ）と、最小対の見本の確認（data/evaluation/minimal_pairs_review.csv の approved を Claude が記入）
CUDA_VISIBLE_DEVICES=0 uv run lexical-acquisition-mechanisms evaluation-data check-with-pretrained-models
uv run lexical-acquisition-mechanisms evaluation-data finalize --review-csv data/evaluation/minimal_pairs_review.csv
# run の設定と、実在語らしさのプローブ
uv run lexical-acquisition-mechanisms training write-run-settings --experiment unknown-word-acquisition --model OLMo-2-0425-1B
uv run lexical-acquisition-mechanisms training write-run-settings --experiment sense-addition --model OLMo-2-0425-1B
CUDA_VISIBLE_DEVICES=0 uv run lexical-acquisition-mechanisms unknown-word-acquisition train-real-word-probes --model OLMo-2-0425-1B
# 学習（1 run を GPU 1枚で。run ごとに）
CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc-per-node 1 --master-port 29500 -m lexical_acquisition_mechanisms \
  training run --run-settings outputs/runs/sense-addition/OLMo-2-0425-1B/same-category-1.yaml
# 語義の追加の最初の結果（足す語義が1つの3つの run と、対照の2つの run だけで）
CUDA_VISIBLE_DEVICES=0 uv run python experiments/trial-1b-base-sequences-5632-k-1000/measure_partial_sense_accuracy.py OLMo-2-0425-1B
# 解析（README の手順3・4と同じ。--model OLMo-2-0425-1B）
```

## 学習と解析を、複数の GPU で並べて流す

`run_gpu_worker.sh` を、使う GPU ごとに1つずつ起動します。作業者は、共有の列から仕事を取ります。

```bash
LOGS=outputs/trial-1b-base-sequences-5632-k-1000/logs
# 学習の列（1行に1つ）と、解析の列（実験の名前を1行に1つ）
printf 'train %s\n' outputs/runs/unknown-word-acquisition/OLMo-2-0425-1B/*.yaml > $LOGS/training_queue.txt
printf '%s\n' sense-addition unknown-word-acquisition > $LOGS/analysis_queue.txt
for gpu in 0 1 2 3 4; do
  setsid nohup experiments/trial-1b-base-sequences-5632-k-1000/run_gpu_worker.sh $gpu $LOGS >> $LOGS/gpu_workers.log 2>&1 < /dev/null &
done
```

- 解析は、その実験の run が全て学習を終え（`completed.json`）、評価データが確定してから、学習より先に行います。

## 課題1の確かめ（2026-11-08。Notion の 33|）

```bash
# 学習の途中の重み（GPU 0〜3）と、学習前の 7B・1B（GPU 4）で、課題1の選択肢の確率を保存し、表にまとめる
setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_sense_choice_probabilities.sh &
uv run python experiments/trial-1b-base-sequences-5632-k-1000/summarize_sense_choice_probabilities.py
```

結果: `outputs/partial-results/sense-choice-probabilities/summary.txt`

## 2つ目の語義の追加（2026-11-09。Notion の 00| の新しい設計、35|〜40|）

未知語の獲得で統合まで済んだ未知語に、別の名詞の語義を足します。割り当て・学習データ・学習の設定は、本来のコマンド（PR #240〜#242）で作りました。評価は、試行用のスクリプトで行っています（本来のコマンドは、結果を見てから作る）。

```bash
# 割り当て（35|）・学習データ（36|）・学習の設定（37|）
uv run lexical-acquisition-mechanisms assignments build --experiment second-sense-addition
uv run lexical-acquisition-mechanisms training-data build --experiment second-sense-addition
uv run lexical-acquisition-mechanisms training write-run-settings --experiment second-sense-addition --model OLMo-2-0425-1B
# 学習（GPU 1〜4。GPU 0 は別の利用者が使っていたため）
LOGS=outputs/trial-1b-base-sequences-5632-k-1000/logs/second-sense-addition
printf '%s\n' outputs/runs/second-sense-addition/OLMo-2-0425-1B/*.yaml > $LOGS/training_queue.txt
: > $LOGS/analysis_queue.txt
for gpu in 1 2 3 4; do
  setsid nohup experiments/trial-1b-base-sequences-5632-k-1000/run_gpu_worker.sh $gpu $LOGS >> $LOGS/gpu_workers.log 2>&1 < /dev/null &
done
# 評価（学習が終わるのを待ってから順に流れる）
# 最小対（34|）・課題1の問題・文脈の効果（27|）
setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_second_sense_evaluation.sh &
# 未知語の表現が X と Y のどちらに近いか（38|）
setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_second_sense_meaning.sh &
# 1つ目と2つ目の語義の用例での未知語の損失（25|）
setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_second_sense_losses.sh &
# 変わった重みを部品ごとに移す（40|）
setsid nohup bash experiments/trial-1b-base-sequences-5632-k-1000/run_second_sense_weight_grafting.sh &
```

結果: `outputs/partial-results/second-sense-evaluation/` の `summary.txt`（文脈の効果）・`meaning_summary.txt`・`losses_summary.txt`・`grafting_summary.txt`

- LLM のサーバは、`VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0` を付けて立てます（この GPU の世代向けの DeepGEMM は CUDA 13 が要り、ここは 12.9 のため）。
