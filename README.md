# 言語モデルにおける語彙獲得と語義追加の機構解析

[English](README.en.md)

OLMo 2（1B / 7B / 13B）を継続事前学習して語彙を操作する実験のコードです。主実験は 7B で行い、1B・13B はモデルサイズの比較に使います。

- **実験A**: 既知の名詞を、モデルが見たことのない文字列（未知語）に置き換えて学習する。未知語が次の3つを獲得する順番を測る。
  - 形への慣れ: 語の2トークン目以降が予測しやすくなる
  - 意味: 内部表現が、置き換えた元の名詞の表現に近づく
  - 語としての統合: 内部表現が、実在語と判定されるようになる
- **実験B**: 既知語に別の語の語義を足して学習し、文脈から語義を選ぶ attention head を特定する。
- **接続**: 追加の学習はせず、実験Aのモデルで実験Bの head を無効化して、未知語がその head に依存しているかを測る。

## 必要なもの

- Python 3.12、[uv](https://docs.astral.sh/uv/)、NVIDIA GPU（CUDA 13.0 に対応したドライバ）、インターネット接続
- LLM: 語の候補の生成や用例の判定に、Qwen3-Next-80B（OLMo 2 とは別のモデル）を vLLM で動かす
- GPU（96GB）: 7B の学習に2枚（1B は1枚、13B は4枚）、LLM のサーバに2枚、そのほかの OLMo 2 を動かす処理に1枚
- ディスク
  - モデル: 約160GB（OLMo 2 の3モデルと LLM）
  - データ: 約50GB
  - 学習した重み: 約3.6TB（既定では、各 run で学習の25 / 50 / 75 / 100%の4時点を保存する。最終の重みだけなら約0.9TB）
  - 学習中の再開用の状態: 最大約165GB（13B の1 run 分。run が終われば消す）

## セットアップ

```bash
uv sync
uv run python -m nltk.downloader wordnet
```

- 学習の記録を W&B（Weights & Biases。実験の記録を残す Web サービス）に送る場合は、`uv run wandb login` を実行しておく。送らない場合は、最初に `export WANDB_MODE=disabled` を実行する。
- `# LLM` を付けたコマンドの前に、別の端末で LLM のサーバを起動したままにしておく。サーバの URL（例: `http://localhost:8000/v1`）は `configs/lam.yaml` の `llm` に書く。

```bash
CUDA_VISIBLE_DEVICES=0,1 uvx --from vllm==0.32.0 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 --tensor-parallel-size 2 --port 8000
```

## 実行の決まり

- 設定は `configs/lam.yaml` にまとめてある。各コマンドの説明は `uv run lam <段階> <処理> --help` で見られる。
- `# GPU` を付けたコマンドは、GPU で OLMo 2 を動かす（GPU は `CUDA_VISIBLE_DEVICES` で選ぶ）。LLM のサーバと OLMo 2 の両方を使うコマンドはないので、GPU が足りなければ LLM のサーバを止めてから実行する。
- `--review` を付けたコマンドは、渡す CSV の `ok` 列を手で埋めてから実行する（基準は手順1）。
- 入力と設定が前回と同じ処理は省かれる（やり直すときは `--force`）。コマンドが止まったら、表示される指示に従って直し、表示されたコマンドから実行し直す。
- run は、1つの学習データでの継続事前学習1回。`train plan` が run ごとの設定ファイルを `outputs/runs/<実験>/<モデル>/` に書き出し、`train run` がそれを1つずつ学習する。
- `train run` は、ppl（検証データでのパープレキシティ）の学習前からの変化が1%を超えると警告を出す（学習は続ける）。警告が出た run は、原因を調べる。
- データセットは `data/` に、学習した重み・結果の表・図は `outputs/` に出力される。

## 実行手順

### 1. データセットの準備

```bash
# 基データ（OLMo 2 の事前学習データを学習の順に並べた、先頭の約3億トークン）と、その直後の検証データ
uv run lam corpus base

# 語: 実験で使う語（名詞・形容詞・未知語など）を選び、語の一覧を確定する
uv run lam lexicon nouns          # LLM
uv run lam lexicon adjectives     # LLM
uv run lam lexicon unknown
uv run lam lexicon select
uv run lam evaldata contexts      # GPU
uv run lam lexicon rate           # LLM
uv run lam lexicon similarity     # GPU
uv run lam lexicon finalize --review data/lexicon/review.csv

# 用例: Pile（公開の英語テキストのデータセット）から切り出した、語を含む256トークンの文章
uv run lam usages collect
uv run lam usages dedup
uv run lam usages judge           # LLM
uv run lam usages review
uv run lam usages finalize --review data/usages/review.csv

# 割り当て（run ごとに、どの語をどの語に置き換えて何回見せるか）と、実験Aの学習データ
uv run lam assign build --exp a
uv run lam assign build --exp b
uv run lam traindata build --exp a

# 評価データ
uv run lam evaldata natural
uv run lam evaldata refs
uv run lam evaldata pairs         # LLM
uv run lam evaldata tasks         # LLM
uv run lam evaldata check         # GPU
uv run lam evaldata finalize --review data/eval/pairs_review.csv
```

- `ok` 列の基準（合えば 1、合わなければ 0）
  - `data/lexicon/review.csv`（1行1語）: 語が、その行に書かれた用途・カテゴリ・語義に合うか
  - `data/usages/review.csv`（1行1用例）: LLM の判定（合否）が正しいか
  - `data/eval/pairs_review.csv`（1行に、1語だけ違う2文）: それぞれの文で、語の意味がその行に書かれた語義に決まるか
- `lexicon finalize`・`usages finalize` は、外れた語（`ok` が 0 の語や、用例が足りない語）があると、`data/lexicon/excluded.csv` に書いて止まる。そのときは `lexicon select` から順に実行し直す。
- `usages finalize` が、LLM の判定の精度が足りずに止まったときは、指示文（`configs/prompts/`）を直して `usages judge` から実行し直す。
- `traindata build` の後に、`data/train/<実験>/<run>/` にある確認の結果と、デコードした用例を見る（手順4でも同じ）。
- 評価の問題の前に見せる例の数（暫定3）は、`evaldata check` が出す、実験で操作しない語での正解率を見て決め、`configs/lam.yaml` に書く。変えたら `evaldata tasks` から実行し直す。

### 2. パイロット（7B）

学習の設定を確定するための、試しの学習（2つの run）です。`run-1` は実験Aの run の1つ目、`control` は何も差し込まない（基データだけで学習する）対照の run です。

```bash
uv run lam train plan --exp a --model 7b
uv run torchrun --nproc-per-node 2 -m lam train run --run outputs/runs/a/7b/run-1.yaml     # GPU
uv run torchrun --nproc-per-node 2 -m lam train run --run outputs/runs/a/7b/control.yaml   # GPU
uv run lam exp-a probe --model 7b   # GPU
```

- `--nproc-per-node` は、使う GPU の数。
- 次のことを確かめてから、学習率・バッチサイズ・ウォームアップ・1トークンの未知語の埋め込みの初期値を確定して、`configs/lam.yaml` に書く。
  - 学習の記録 `outputs/checkpoints/a/7b/<run>/metrics.jsonl` で、ppl の変化が1%以内で、差し込んだ用例の損失が下がり、学習が不安定になっていない
  - `exp-a probe` が出すプローブ（モデルの内部表現から、実在語かどうかを当てる分類器）が、学習に使わなかった語でも当てられる
- 値を変えたら、`train plan` から2つの run をやり直す。

### 3. 実験A（7B）

```bash
# 学習（学習済みの run は省かれる）
uv run lam train plan --exp a --model 7b
for f in outputs/runs/a/7b/*.yaml; do
  uv run torchrun --nproc-per-node 2 -m lam train run --run "$f" || break   # GPU
done

# 評価
uv run lam exp-a probe --model 7b     # GPU
uv run lam exp-a measure --model 7b   # GPU
uv run lam exp-a analyze --model 7b
uv run lam exp-a report
```

- 実験Bで語義1つあたりに見せる用例の数 K を決める。`outputs/figures/a/` の意味の曲線を見て、それ以上見せてもほとんど上がらない回数を選び、理由とともに `configs/lam.yaml` に書く。

### 4. 実験B（7B）

```bash
# 形容詞の用例を K に合わせて足す（足りていれば省かれる）
uv run lam usages collect
uv run lam usages dedup
uv run lam usages judge           # LLM
uv run lam usages review
uv run lam usages finalize --review data/usages/review.csv

# 学習データと学習
uv run lam traindata build --exp b
uv run lam train plan --exp b --model 7b
for f in outputs/runs/b/7b/*.yaml; do
  uv run torchrun --nproc-per-node 2 -m lam train run --run "$f" || break   # GPU
done

# 評価
uv run lam exp-b accuracy --model 7b  # GPU
uv run lam exp-b map --model 7b       # GPU
uv run lam exp-b stats --model 7b
uv run lam exp-b encode --model 7b    # GPU
uv run lam exp-b causal --model 7b    # GPU
uv run lam exp-b report
```

- `usages finalize` で語が外れたときは、`lexicon select` から順に実行し直してから、手順4に戻る。
- 最小対（1語だけ違う2文）の数（暫定: 語義の組ごとに20組）は、`exp-b stats` が出す信頼区間の幅を見て、広すぎれば増やす。増やしたら、`evaldata pairs` から `evaldata finalize` までと、`exp-b accuracy` 以降を実行し直す。

### 5. 接続（7B）

```bash
uv run lam connect dependence --model 7b   # GPU
uv run lam connect analyze --model 7b
uv run lam connect map --model 7b          # GPU
uv run lam connect report
```

### 6. 1B・13B

- 学習率は、パイロットで決めた決め方で 1B・13B の値を求め（暫定: 各モデルの事前学習の前半（ステージ1）の最終値）、`configs/lam.yaml` に書く。
- 手順 3〜5 を、`7b`（`--model` とパス）を `1b` か `13b` に替えて繰り返す。`--nproc-per-node` は、1B で 1、13B で 4 にする。
- パイロットと、K・最小対の数の決定は行わず、7B で決めた値を使う。

### 7. 集計

```bash
uv run lam results collect
```

全結果をまとめた表 `outputs/results/all.parquet` と、図 `outputs/figures/` ができます。

## テスト

```bash
uv run pytest
```

乱数で初期化した小さなモデルと、LLM・外部 API の代わりに決まった応答を返す部品を使い、GPU なしで全コマンドを通して実行します。
