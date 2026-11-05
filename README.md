# 言語モデルにおける語彙獲得と語義追加の機構解析

[English](README.en.md)

OLMo 2（1B / 7B / 13B）を継続事前学習して語彙を操作し、次の3つの実験を行うコードです。主に 7B で実験し、1B・13B はモデルサイズの比較に使います。学習では、語を含む256トークンの文章（用例）を Pile（公開の英語テキストのデータセット）から切り出し、その中の語を置き換えて学習データに差し込みます。1回の継続事前学習を run と呼びます。

- **未知語の獲得**: 名詞（例: pasta）を、モデルが見たことのない文字列（未知語。例: resonunt）に置き換えた用例で学習する。未知語が次の3つを獲得する順番を、それぞれの指標で測る。
  - 形への慣れ: 未知語の2トークン目以降が予測しやすくなる
  - 意味: 未知語の内部表現が、元の名詞の内部表現に近づく
  - 語としての統合: 未知語の内部表現が、実在語のものと判定されるようになる
- **語義の追加**: 既知語（例: grape）を別の既知語（ターゲット語。例: car）に置き換えた用例で学習し、ターゲット語に語義を足す。足す語義は、ターゲット語と同じカテゴリ（食べ物・動物などの意味の分類）の名詞・別のカテゴリの名詞・形容詞の3種類で、run によって1・2・4個を足す。文脈から語義を選ぶ attention head と、その head が判定する属性（名詞のカテゴリ・品詞）を特定する。
- **カテゴリ判定 head への未知語の依存**: 学習はしない。未知語の獲得で学習したモデルで、語義の追加で特定した「名詞のカテゴリを判定する head」を無効化し（出力のうちカテゴリを表す方向の成分を平均値に置き換え）、未知語のカテゴリの判定がどれだけ落ちるかを、元の名詞と比べる。

評価には、未知語の指標のほか、選択肢から答えを選ばせる課題（語義の同定と、属性の判定）を使います。問題の前には、例をいくつか見せます。

## 必要なもの

- Python 3.12、[uv](https://docs.astral.sh/uv/)、インターネット接続
- NVIDIA GPU（96GB）: 学習に 1B は1枚・7B は2枚・13B は4枚。LLM のサーバに2枚
- ディスク: 約4TB（うち学習した重みが約3.6TB）。学習中の run は、続きから学習するための状態に、さらに 7B で約180GB・13B で約330GB を使う（学習を終えると消す）

## セットアップ

```bash
uv sync
uv run wandb login   # 学習の記録を W&B（実験の記録を残す Web サービス）に送る場合だけ
```

`# LLM` を付けたコマンドは、LLM（Qwen3-Next-80B）で語の候補を作ったり用例を判定したりします。これらを実行する間は、別の端末で LLM のサーバを起動しておきます（URL は `settings.yaml` の `llm.server_url`）。

```bash
# DeepGEMM（FP8 の行列演算の部品）は、RTX PRO 6000（Blackwell）では CUDA 13 以上の nvcc が要るので使わない
VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 CUDA_VISIBLE_DEVICES=0,1 \
  uvx --from vllm==0.32.0 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --tensor-parallel-size 2 --port 8000
```

## 使い方

- リポジトリの最上位のディレクトリで、`uv run lexical-acquisition-mechanisms <段階> <処理>` の形で実行します（`--help` で説明が出ます）。設定はすべて `settings.yaml` にあります。
- `# GPU` を付けたコマンドは GPU を使います。使う GPU は `CUDA_VISIBLE_DEVICES` で選びます。LLM のサーバと同じ GPU を使うときは、先にサーバを止めます。
- 入力と設定が前回と同じ処理は省かれます（やり直すときは `--recompute` を付けます）。コマンドが止まったら、表示に従って直してから、表示されたコマンドから実行し直します。
- 学習は、複数の GPU で動かすため、torchrun（GPU ごとにプロセスを起動する PyTorch のコマンド）から実行します。長くかかるので、tmux（端末を閉じても実行を続けられるソフト）などの中で実行します。止まった run は、同じコマンドで続きから学習します。
- データセットは `data/` に、学習した重み・結果・図は `outputs/` に書き出されます。

## 手順

### 1. データセットの作成

#### 1-1. 基データと語

```bash
# 基データ（OLMo 2 の事前学習データの、学習順で先頭の約3億トークン）と、
# その直後の約1,000万トークン（検証データ）をダウンロードする
uv run lexical-acquisition-mechanisms base-data download

# 語の候補を作る: 名詞と形容詞は LLM が挙げ、Pile での使われ方（その語義・品詞で使われる割合）も
# LLM が判定する。未知語は、OLMo 2 の事前学習データと Pile に一度も現れない文字列から作る
uv run lexical-acquisition-mechanisms words generate-noun-candidates                # LLM
uv run lexical-acquisition-mechanisms words generate-adjective-candidates           # LLM
uv run lexical-acquisition-mechanisms words generate-unknown-word-candidates

# 候補から、名詞72語・ターゲット語18語・形容詞72語・未知語72個と、
# 自然な語（実験で操作しない実在語。課題の例やプローブの学習に使う）を選ぶ
uv run lexical-acquisition-mechanisms words select

# ターゲット語と名詞の組ごとに、意味の近さを LLM が評定する
uv run lexical-acquisition-mechanisms words rate-similarity-with-llm               # LLM

# 中立文脈（語の意味を示さない文の書き出し。例: "Do you know about the"）を20種選ぶ
uv run lexical-acquisition-mechanisms evaluation-data select-neutral-contexts       # GPU

# ターゲット語と名詞の組ごとに、公開されている OLMo 2 の3モデルで内部表現の類似度を測り、
# 確認用の data/words/review.csv を書き出す
uv run lexical-acquisition-mechanisms words measure-hidden-state-similarity        # GPU
```

**手作業**: `data/words/review.csv` の `approved` 列に、語がその行のカテゴリ・語義に合えば 1、合わなければ 0 を記入する。

```bash
# approved が 1 の語で、以降の手順で使う語の一覧を書き出す
uv run lexical-acquisition-mechanisms words finalize --review-csv data/words/review.csv
```

**手作業**（0 を付けた語があって止まったとき）: `words select` から順に実行し直す（外した語の代わりの語が選ばれる）。

#### 1-2. 用例

```bash
# Pile から、選んだ語を含む用例の候補を切り出す
uv run lexical-acquisition-mechanisms passages collect

# ほぼ同じ候補を除き、元の文書の単位で学習用と評価用に分ける
uv run lexical-acquisition-mechanisms passages remove-duplicates-and-split

# 候補の中で、語が意図した語義で使われているかを、LLM が出現ごとに判定する
uv run lexical-acquisition-mechanisms passages judge-word-sense                     # LLM

# 名詞・形容詞・ターゲット語ごとに、合格した候補を20件抜き出し、
# 確認用の data/passages/review.csv を書き出す
uv run lexical-acquisition-mechanisms passages sample-for-review
```

**手作業**: `data/passages/review.csv` の `approved` 列に、語がその行の語義で使われていれば 1、そうでなければ 0 を記入する。

```bash
# 語ごとに LLM の判定の精度（合格とした候補のうち、人も 1 とした割合）を出し、
# 合格した候補を、学習用・評価用の用例として書き出す
uv run lexical-acquisition-mechanisms passages finalize --review-csv data/passages/review.csv
```

**手作業**（`passages judge-word-sense` か `passages finalize` が、用例が足りない語があって止まったとき）: 表示に従って実行し直す。外した語があれば `words select` から（代わりの語が選ばれる）、なければ `settings.yaml` の `passages.collection_factor` を大きくして `passages collect` から。

**手作業**（精度が足りない語があって止まったとき）: LLM への指示文（`llm_prompts/`）を直し、`passages judge-word-sense` から実行し直す。

#### 1-3. 割り当てと学習データ

```bash
# run ごとに、どの語の用例を、どの語に置き換えて差し込むか（未知語の獲得では件数も）を決める
uv run lexical-acquisition-mechanisms assignments build --experiment unknown-word-acquisition
uv run lexical-acquisition-mechanisms assignments build --experiment sense-addition

# 未知語の獲得の9つの run（8つと、何も差し込まない対照）の学習データを作る:
# 名詞の用例の名詞を未知語に置き換え、基データに差し込む
uv run lexical-acquisition-mechanisms training-data build --experiment unknown-word-acquisition
```

**手作業**: `data/training/unknown-word-acquisition/checks.csv` で、run ごとの確認の結果（語ごとの差し込んだ件数、置き換え漏れ、総トークン数）を見る。`decoded_passages.txt`（差し込んだ用例を無作為に抜き出し、トークン列を文字に戻したもの）も目で確かめる。

#### 1-4. 評価データ

```bash
# 自然な語の用例から、課題の例と、プローブ（内部表現から語の性質を当てる分類器）の
# 学習に使う文を取り出す
uv run lexical-acquisition-mechanisms evaluation-data extract-natural-word-sentences

# プローブの学習に使う実在語と非語（実在しない文字列）と、未知語の指標を正規化する
# （既知の語の値を 1 にそろえる）ときの基準にする既知の語を作る
uv run lexical-acquisition-mechanisms evaluation-data build-probe-and-normalization-words

# 最小対（手がかりの1語だけが違い、ターゲット語の語義が変わる2文）を作る
uv run lexical-acquisition-mechanisms evaluation-data generate-minimal-pairs         # LLM

# 課題の問題を作る
uv run lexical-acquisition-mechanisms evaluation-data build-task-questions           # LLM

# 公開されている OLMo 2 の3モデルで、最小対の手がかりが語義を決めているかを確かめ、自然な語での
# 課題の正解率を出す。確認に通った最小対から一部を抜き出し、
# 確認用の data/evaluation/minimal_pairs_review.csv に書き出す
uv run lexical-acquisition-mechanisms evaluation-data check-with-pretrained-models   # GPU
```

**手作業**（最小対が1つもない語義の組があって止まったとき）: `settings.yaml` の `evaluation.minimal_pair_candidate_factor`（LLM に作らせる候補の数の倍率）を大きくし、`evaluation-data generate-minimal-pairs` から実行し直す。

**手作業**:

1. 前のコマンドが出す、自然な語での課題の正解率を見て、問題の前に見せる例の数（暫定: 3）を決め、`settings.yaml` の `evaluation.number_of_task_examples` に書く（変えたら、`evaluation-data build-task-questions` から実行し直す）。正解率が偶然の水準を超えない課題は、そのモデルでは使われない。
   - `data/evaluation/target_paraphrase_checks.csv` で、元の語義を示す文で言い換えの語が選ばれているか（`above_chance`）も見る。選ばれない語は、`llm_prompts/list-paraphrases.md` を直して `evaluation-data build-task-questions` から実行し直す（言い換えの語が替わる）。
2. `data/evaluation/minimal_pairs_review.csv` の `approved` 列に、それぞれの文でターゲット語の語義がその行の語義に決まれば 1、そうでなければ 0 を記入する。

```bash
# approved が 0 の最小対を除き、評価に使う最小対と課題の問題を書き出す
uv run lexical-acquisition-mechanisms evaluation-data finalize \
  --review-csv data/evaluation/minimal_pairs_review.csv
```

### 2. パイロット（7B）

学習の設定を決めるために、2つの run だけを学習します。手順2〜5では、端末を開いたら最初に次を実行します。

```bash
MODEL=OLMo-2-1124-7B
export CUDA_VISIBLE_DEVICES=0,1   # 使う GPU（LLM のサーバと同じなら、先にサーバを止める）
NUMBER_OF_GPUS=2                  # 上で選んだ GPU の数
export WANDB_MODE=disabled        # 学習の記録を W&B に送らない場合だけ
```

```bash
# run ごとの学習の設定ファイルを書き出す
uv run lexical-acquisition-mechanisms training write-run-settings \
  --experiment unknown-word-acquisition --model "$MODEL"

# run-1（未知語の獲得の1つ目の run）と control（何も差し込まない対照の run）を学習する
uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
  training run --run-settings "outputs/runs/unknown-word-acquisition/$MODEL/run-1.yaml"     # GPU
uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
  training run --run-settings "outputs/runs/unknown-word-acquisition/$MODEL/control.yaml"   # GPU

# 公開されているモデルで、内部表現から実在語かどうかを当てるプローブを学習し、正解率を出す
uv run lexical-acquisition-mechanisms unknown-word-acquisition train-real-word-probes \
  --model "$MODEL"   # GPU
```

**手作業**:

1. 学習の記録（`outputs/checkpoints/unknown-word-acquisition/$MODEL/<run>/metrics.jsonl`）で、次を確かめる。
   - 検証データでのパープレキシティ（次のトークンの予測のしにくさ）の、学習前からの変化が1%以内
   - 差し込んだ用例の損失が下がっている
   - 損失が急に跳ね上がったり、NaN（数値でない値）になったりしていない
2. プローブが、学習に使わなかった語でも当てられるかを確かめる。
3. 学習率・バッチサイズ・ウォームアップ・1トークンの未知語（新しいトークン番号を割り当てた未知語）の埋め込みの初期値を決め、`settings.yaml` の `training`（`learning_rate_by_model`・`batch_size_in_tokens`・`warmup_fraction`・`new_token_embedding_initialization`）に書く。学習率は、1B・13B の値も同じ方法で求められるように、求め方として決める（暫定: 各モデルの事前学習のステージ1の最後の学習率）。値を変えたら、手順2をやり直す。

### 3. 未知語の獲得（7B）

```bash
# 9つの run を学習する
uv run lexical-acquisition-mechanisms training write-run-settings \
  --experiment unknown-word-acquisition --model "$MODEL"
for run_settings in outputs/runs/unknown-word-acquisition/"$MODEL"/*.yaml; do
  uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
    training run --run-settings "$run_settings" || break   # GPU
done
```

**手作業**: すべての run の学習が終わったら、各 run の学習の記録（`metrics.jsonl`）で、パープレキシティの学習前からの変化が1%以内かを確かめる。超えた run は、原因を調べて報告する。

```bash
# 実在語かどうかを当てるプローブを学習する（パイロットで学習済みなら省かれる）
uv run lexical-acquisition-mechanisms unknown-word-acquisition train-real-word-probes \
  --model "$MODEL"   # GPU

# 各 run のモデルで、未知語の3つの指標（形への慣れ・意味・語としての統合）を層ごとに測る
uv run lexical-acquisition-mechanisms unknown-word-acquisition measure-indicators \
  --model "$MODEL"   # GPU

# 指標を、差し込み0回の未知語が 0、既知の語が 1 になるように正規化し、
# 3つの指標が獲得される順番を検定する
uv run lexical-acquisition-mechanisms unknown-word-acquisition analyze-acquisition-order \
  --model "$MODEL"

# 図と表を outputs/figures/unknown-word-acquisition/ に作る
uv run lexical-acquisition-mechanisms unknown-word-acquisition make-report
```

**手作業**: `outputs/figures/unknown-word-acquisition/` の意味の指標の図を見て、それ以上差し込んでもほとんど上がらない件数を K（語義の追加で、足す語義1つあたりに差し込む用例の数）とし、`settings.yaml` の `sense_addition.passages_per_added_sense` に書く。

### 4. 語義の追加（7B）

```bash
# K に合わせて、形容詞の用例を足す（足りていれば省かれる）
uv run lexical-acquisition-mechanisms passages collect
uv run lexical-acquisition-mechanisms passages remove-duplicates-and-split
uv run lexical-acquisition-mechanisms passages judge-word-sense                     # LLM
uv run lexical-acquisition-mechanisms passages finalize --review-csv data/passages/review.csv

# 語義の追加の11の run（足す語義の種類3つ × 数1・2・4の9つと、名詞・形容詞の対照2つ）の
# 学習データを作る。対照の run には、名詞・形容詞の用例を置き換えずに差し込む
uv run lexical-acquisition-mechanisms training-data build --experiment sense-addition
```

**手作業**: `data/training/sense-addition/` の `checks.csv` と `decoded_passages.txt` を、1-3 と同じように見る。

```bash
# 11の run を学習する
uv run lexical-acquisition-mechanisms training write-run-settings \
  --experiment sense-addition --model "$MODEL"
for run_settings in outputs/runs/sense-addition/"$MODEL"/*.yaml; do
  uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
    training run --run-settings "$run_settings" || break   # GPU
done
```

**手作業**: すべての run の学習が終わったら、手順3と同じように、各 run のパープレキシティの変化を確かめる。

```bash
# 最小対の文で、ターゲット語の語義を使い分けられているかを測る
uv run lexical-acquisition-mechanisms sense-addition measure-sense-accuracy \
  --model "$MODEL"   # GPU

# 各 head と MLP（各層の全結合のブロック）の出力を、最小対のもう一方の文での出力に入れ替え、
# 語義の選択への効果を測る
uv run lexical-acquisition-mechanisms sense-addition measure-head-and-mlp-effects \
  --model "$MODEL"   # GPU

# 効果の信頼区間を出し、採用の基準を満たす head と MLP を選んで、
# どの種類の語義を足したときに効くかで分類する
uv run lexical-acquisition-mechanisms sense-addition select-significant-heads-and-mlps \
  --model "$MODEL"
```

**手作業**: 前のコマンドが出す信頼区間が広すぎれば、`settings.yaml` の `evaluation.minimal_pairs_per_sense_pair`（語義の組ごとの最小対の数。暫定: 20）を増やし、1-4 の `evaluation-data generate-minimal-pairs` から `evaluation-data finalize` までと、この手順の `sense-addition measure-sense-accuracy` から実行し直す（学習はやり直さない）。

```bash
# 公開されているモデルで、選んだ head の出力から名詞のカテゴリと品詞を当てるプローブを学習する
uv run lexical-acquisition-mechanisms sense-addition train-attribute-probes \
  --model "$MODEL"   # GPU

# head の出力のうち属性を表す方向の成分を入れ替えて、属性を判定する head を特定し、保存する
uv run lexical-acquisition-mechanisms sense-addition test-attribute-directions \
  --model "$MODEL"   # GPU

# 図と表を outputs/figures/sense-addition/ に作る
uv run lexical-acquisition-mechanisms sense-addition make-report
```

### 5. カテゴリ判定 head への未知語の依存（7B）

```bash
# 手順3で学習した各 run のモデルで、手順4で保存した、カテゴリを判定する head を無効化し、
# 未知語と元の名詞のそれぞれで、カテゴリの判定がどれだけ落ちるかを測る
# （依存度 = 未知語での低下 ÷ 元の名詞での低下）
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads measure-dependence \
  --model "$MODEL"   # GPU

# 依存度を、意味と語としての統合の指標などで説明する回帰を当てはめる。
# 指標を閾値で区切った状態ごとの比較もする
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads fit-regression \
  --model "$MODEL"

# 同じ名詞の意味を、未知語に付けた場合と既知語に足した場合とで、
# head ごとの効果の並び（地図）を比べる
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads compare-head-maps \
  --model "$MODEL"   # GPU

# 図と表を outputs/figures/dependence-on-category-judging-heads/ に作る
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads make-report
```

### 6. 1B・13B

**手作業**: 手順2で決めた求め方で 1B・13B の学習率を求め、`settings.yaml` の `training.learning_rate_by_model` に書く。

そのあと、手順2の最初の3行を次のどちらかに替えて、手順3〜5を繰り返します。パイロットは行わず、K と最小対の数は 7B で決めた値を使います。手順4の形容詞の用例の追加は省かれるので、LLM のサーバは要りません。

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

### 7. 集計

```bash
# 全結果を1つの表 outputs/results/all_results.parquet にまとめ、図を outputs/figures/ に作る
uv run lexical-acquisition-mechanisms results collect
```

## テスト

```bash
uv run pytest
```

GPU と LLM のサーバがなくても、全コマンドが最後まで通るかを確かめます。
