# 言語モデルにおける語彙獲得と語義追加の機構解析

[English](README.en.md)

OLMo 2（1B / 7B / 13B）を継続事前学習して語彙を操作し、言語モデルが新しい語や語義をどう獲得し、どう処理しているかを調べる実験のコードです。主実験は 7B で行い、1B・13B はモデルサイズの比較に使います。

実験は次の3つです。

- **未知語の獲得**: 既知の名詞（例: pasta）を含む文章で、その名詞を、モデルが見たことのない文字列（未知語。例: resonunt）に置き換えて学習データに差し込み、学習する。未知語が次の3つを獲得する順番を、それぞれの指標で測る。
  - 形への慣れ: 未知語の2トークン目以降が予測しやすくなる
  - 意味: 未知語の内部表現が、元の名詞の内部表現に近づく
  - 語としての統合: 未知語の内部表現が、実在語のものと判定されるようになる（内部表現から実在語かどうかを当てる分類器（プローブ）で判定する）
- **語義の追加**: 既知語（例: grape）を含む文章で、その語を別の既知語（ターゲット語。例: car）に置き換えて学習し、ターゲット語に語義を足す。足す語義は、ターゲット語と同じカテゴリ（食べ物・動物などの意味の分類）の名詞・別のカテゴリの名詞・形容詞の3種類で、run（1回の継続事前学習）によって1個・2個・4個のどれかを足す。文脈から語義を選ぶときに働く attention head と、その head が判定している属性（名詞のカテゴリ・品詞）を特定する。
- **カテゴリ判定 head への未知語の依存**: 追加の学習はしない。未知語の獲得で学習したモデルで、語義の追加で特定した「名詞のカテゴリを判定する head」を無効化し（head の出力のうち、カテゴリを表す方向の成分を平均値に置き換え）、未知語のカテゴリ（元の名詞のカテゴリ）の判定がどれだけ落ちるかを、元の名詞と比べる。意味の獲得の程度をそろえたうえで、語として統合された未知語ほど、既知語と同じ head に頼ってカテゴリを判定しているかを調べる。

評価には、選択肢から答えを選ばせる課題を使います。課題は、語義の同定（文中の語と同じ意味の語を選ぶ）と、属性の判定（語のカテゴリか品詞を選ぶ）の2種類です。

## 必要なもの

- Python 3.12、[uv](https://docs.astral.sh/uv/)、NVIDIA GPU（CUDA 13.0 に対応したドライバ）、インターネット接続
- LLM（大規模言語モデル）: 語の候補の生成や、用例（語を含む256トークンの文章）の判定に、Qwen3-Next-80B（OLMo 2 とは別のモデル）を vLLM（LLM を動かすサーバのソフトウェア）で動かす。以下、LLM はこのモデルを指す
- GPU（96GB）: 7B の学習に2枚（1B は1枚、13B は4枚）、LLM のサーバに2枚、OLMo 2 を動かすそのほかの処理に1枚
- ディスク
  - モデル: 約160GB（OLMo 2 の3モデルと LLM）
  - データ: 約50GB
  - 学習した重み: 約3.6TB（既定では、各 run で学習の25 / 50 / 75 / 100%の4時点を保存する。保存する時点は `settings.yaml` の `training` で変えられ、最終の重みだけなら約0.9TB）
  - 学習を途中から再開するための状態（最適化器の状態など）: 最大約165GB（13B の1 run 分。run が終わると消える）

## セットアップ

```bash
uv sync
uv run python -m nltk.downloader wordnet
```

- 学習の記録を W&B（Weights & Biases。実験の記録を残す Web サービス）に送る場合は、`uv run wandb login` を実行しておく。送らない場合は、`export WANDB_MODE=disabled` を実行しておく。
- `# LLM` を付けたコマンドの前に、別の端末で LLM のサーバを起動したままにしておく。起動には数分かかるので、`curl http://localhost:8000/v1/models` が応答してから使う。サーバの URL（例: `http://localhost:8000/v1`）は `settings.yaml` の `llm.server_url` に書く。

```bash
CUDA_VISIBLE_DEVICES=0,1 uvx --from vllm==0.32.0 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --tensor-parallel-size 2 --port 8000
```

## 実行の決まり

- コマンド
  - bash で、リポジトリの最上位のディレクトリで実行する。形は `uv run lexical-acquisition-mechanisms <段階> <処理>` で、末尾に `--help` を付けると説明が出る。
  - 設定は、すべて `settings.yaml` にある。
  - 入力と設定が前回と同じ処理は省かれる（やり直すときは `--recompute`）。コマンドが止まったら、表示される指示に従って直し、表示されたコマンドから実行し直す。
  - 手順2〜5では、最初に使うモデルなどをシェルの変数に入れる。端末を開き直したら、`export WANDB_MODE=disabled`（W&B に送らない場合）と、その手順の最初の変数の行を実行し直してから続ける。
- GPU と LLM のサーバ
  - `# GPU` を付けたコマンドは GPU で OLMo 2 を、`# LLM` を付けたコマンドは LLM のサーバを使う。GPU は `CUDA_VISIBLE_DEVICES` で選ぶ。
  - 両方を使うコマンドはない。LLM のサーバと同じ GPU を使うときは、サーバを止めてから実行する。
  - 「学習前のモデル」は、公開されている OLMo 2 の重み（継続事前学習をする前）を指す。
- 人手確認の CSV
  - `--review-csv` を付けたコマンドの前に、渡す CSV の `approved` 列を手で埋める（合えば 1、合わなければ 0。基準は各手順に書いてある）。空の行があると、コマンドは止まる。
  - CSV を書き出すコマンドは、記入済みの `approved` を残し、新しい行を空のまま足す。埋めるのは空の行だけでよい。
- 継続事前学習
  - `training write-run-settings` が run ごとの設定ファイルを `outputs/runs/<実験>/<モデル>/` に書き出し、`training run` がそのうち1つを学習する。
  - 複数の GPU を使うため、torchrun（GPU ごとにプロセスを起動する PyTorch のコマンド）で、このリポジトリの Python パッケージ `lexical_acquisition_mechanisms` を起動する。`--nproc-per-node` は使う GPU の数。
  - 1 run に長い時間がかかるので、端末を閉じても止まらないように、tmux（端末を閉じても実行を続けられるソフト）などの中で実行する。止まった run は、同じコマンドで続きから学習する。
  - `training run` は、検証データ（学習に使わない、基データの直後の約1,000万トークン）でのパープレキシティ（次のトークンの予測のしにくさ）が、学習前から1%を超えて変わると警告を出す（学習は続ける）。警告が出た run は、原因を調べて報告する。
- データセットは `data/` に、学習した重み・結果の表・図は `outputs/` に出力される。

## 実行手順

### 1. データセットの準備

#### 1-1. 基データ

```bash
# OLMo 2 の事前学習の最初の段階（ステージ1。事前学習の大部分を占める）の設定から、
# OLMo 2 の事前学習データの順番を再現し、先頭の約3億トークン（基データ）と、
# その直後の約1,000万トークン（検証データ）をダウンロードする
uv run lexical-acquisition-mechanisms base-data download
```

#### 1-2. 実験で使う語

```bash
# 名詞とターゲット語の候補: LLM が、6つのカテゴリ（暫定: 食べ物、動物、乗り物、衣類、
# 家具・家庭用品、建物）ごとに数えられる具体名詞を挙げ、WordNet（英語の語彙のデータベース）で
# そのカテゴリだけに属する語を残す。各語に、トークン数と、頻度（OLMo 2 の事前学習データと
# Pile（公開の英語テキストのデータセット）での出現回数）と、Pile での出現例のうち
# そのカテゴリの語義で使われる割合（LLM が判定）を付ける
uv run lexical-acquisition-mechanisms words generate-noun-candidates                # LLM

# 形容詞の候補: LLM が形容詞を挙げ、各カテゴリとの結びつきの強さを評定する。各語に、
# トークン数・頻度と、Pile での出現例のうち形容詞として使われる割合（LLM が判定）を付ける
uv run lexical-acquisition-mechanisms words generate-adjective-candidates           # LLM

# 未知語の候補: 6つのカテゴリに属さない実在語の1〜2文字を変えた文字列（綴りが実在語に近い）と、
# 非語（英語の単語ではない文字列）のデータベース ARC Nonword Database の、発音できる非語
# （綴りが遠い）を集める。2・3トークンになり、OLMo 2 の事前学習データと Pile に
# 一度も現れないものを残す
uv run lexical-acquisition-mechanisms words generate-unknown-word-candidates

# 候補から、実験で使う語を選ぶ（差し替え用の予備の候補も選ぶ）
#   名詞72語: そのカテゴリの語義で使われる割合が高い1トークンの語を、各カテゴリ12語
#     （カテゴリの間で頻度をそろえ、用例が十分に取れる見込みの語に限る）
#   ターゲット語18語: 名詞72語以外の名詞の候補から、各カテゴリで1・2・3トークンの語を1語ずつ
#   形容詞72語: 形容詞として使われる割合が高く、どのカテゴリとも結びつきが弱い1トークンの語
#   未知語72個: 1・2・3トークン各24個。2・3トークンは、最初のトークンがほかの語と重ならない
#     候補から、綴りの近いもの・遠いもの各12個を無作為に選ぶ。1トークンのものには、
#     使われていないトークン番号を割り当てる
#   自然な語: 実験で操作しない実在語（課題の例、プローブの学習、指標の基準に使う名詞・形容詞）
uv run lexical-acquisition-mechanisms words select

# ターゲット語と名詞のすべての組（18 × 72組と、予備の候補の分）について、
# LLM が意味の近さを7段階で評定する
uv run lexical-acquisition-mechanisms words rate-similarity-with-llm               # LLM

# 中立文脈（語の意味を示さない文の書き出し。例: "Do you know about the"）を選ぶ。
# 学習前の3モデルで、各候補の直後でのカテゴリごとの自然な名詞の確率を比べ、
# 特定のカテゴリの名詞を予測しやすくする候補を外して、20種を残す
uv run lexical-acquisition-mechanisms evaluation-data select-neutral-contexts       # GPU

# 同じ組について、学習前の3モデルで、中立文脈20種と文脈なしの後に置いた語の内部表現の
# コサイン類似度を出す。同じカテゴリの組が上位、別のカテゴリの組が下位に入るかを確かめ、
# 人手確認用の data/words/review.csv（1行1語）を書き出す
uv run lexical-acquisition-mechanisms words measure-hidden-state-similarity        # GPU
```

**手作業**: `data/words/review.csv` の `approved` 列を埋める（語が、その行に書かれた用途・カテゴリ・語義に合えば 1、合わなければ 0）。

```bash
# approved を読み、0 の語があれば data/words/excluded.csv に書いて止まる。なければ、
# 以降のコマンドが使う語の一覧（語・用途・カテゴリ・語義・トークン列）を書き出す
uv run lexical-acquisition-mechanisms words finalize --review-csv data/words/review.csv
```

- `words select` は、条件を満たす語が足りないカテゴリがあると止まる。そのときは、`settings.yaml` の `words` でカテゴリを替えて、`words generate-noun-candidates` から実行し直す。
- `evaluation-data select-neutral-contexts` は、残る文脈が20種に満たないと止まる。そのときは、同じ形の書き出しを `settings.yaml` の `evaluation` の候補に足して、このコマンドから実行し直す。
- `words finalize` が止まったときは、`words select` から順に実行し直す（`words select` は、外れた語を予備の候補と入れ替え、ほかの語は変えない）。

#### 1-3. 用例

```bash
# Pile を先頭から読み、選んだ語を含む256トークンの文章（用例の候補）を、語ごとに必要な数
# （例: 名詞は学習用10,000件と評価用500件）の3倍まで切り出す。語の形は、先頭スペース付き・
# 小文字・単数に限り、実験で操作するほかの語を含む文章は使わない
uv run lexical-acquisition-mechanisms passages collect

# ほぼ同じ候補を除き、元の文書の単位で学習用と評価用に分ける
uv run lexical-acquisition-mechanisms passages remove-duplicates-and-split

# LLM が、候補の中の語の出現ごとに、意図した語義（名詞は、選んだときのカテゴリの語義）で
# 使われているか（慣用句・固有名詞の一部でないか。形容詞は形容詞として使われているか）を判定する
uv run lexical-acquisition-mechanisms passages judge-word-sense                     # LLM

# 名詞・形容詞・ターゲット語ごとに、合格した候補を20件抜き出し、
# 人手確認用の data/passages/review.csv（1行1件）に書き出す
uv run lexical-acquisition-mechanisms passages sample-for-review
```

**手作業**: `data/passages/review.csv` の `approved` 列を埋める（語が、その行に書かれた語義で使われていれば 1、そうでなければ 0）。

```bash
# 語ごとに、LLM の判定の精度（人手で 1 とした割合）を出す。精度と用例の数が足りれば、
# 合格した候補を、学習用・評価用の用例として必要な数だけ書き出す
uv run lexical-acquisition-mechanisms passages finalize --review-csv data/passages/review.csv
```

- `passages finalize` は、精度の足りない語があると止まる。そのときは、LLM への指示文（`llm_prompts/`）を直して、`passages judge-word-sense` から実行し直す。
- `passages finalize` は、用例が足りない語があると、`data/words/excluded.csv` に書いて止まる。そのときは、`words select` から順に実行し直す。

#### 1-4. 割り当てと、未知語の獲得の学習データ

```bash
# 未知語の獲得: 未知語8個と名詞8語を1組にして9組作り、8つの run で、各未知語を、どの名詞の
# 代わりに何回差し込むかを決める。回数は 0・1・10・100・1,000・3,000・5,000・10,000回で、
# 各未知語は、8つの run を通して、各回数と、同じ組の8つの名詞に1回ずつ当たる
uv run lexical-acquisition-mechanisms assignments build --experiment unknown-word-acquisition

# 語義の追加: 9つの run（足す語義の種類3つ × 数1・2・4）ごとに、
# 各ターゲット語に、どの名詞・形容詞の語義を足すかを決める
uv run lexical-acquisition-mechanisms assignments build --experiment sense-addition

# 割り当てに従い、名詞の用例の中の名詞を未知語に置き換えて基データに差し込み、9つの run
# （8つの run と、何も差し込まない対照の run）の学習データ（トークン列）を作る。
# 差し込んだ数などの確認の結果と、デコードした一部の用例も書き出す
uv run lexical-acquisition-mechanisms training-data build --experiment unknown-word-acquisition
```

- `training-data build` の後に、`data/training/<実験>/<run>/` にある確認の結果と、デコードした用例を見る（手順4でも同じ）。

#### 1-5. 評価データ

```bash
# 自然な語の用例から、その語を含む1文を取り出す（課題の例、プローブの学習、確認に使う）
uv run lexical-acquisition-mechanisms evaluation-data extract-natural-word-sentences

# 未知語の指標に使う語を作る: プローブの学習に使う実在語と非語（実在語どうしで2トークン目以降を
# 入れ替えた文字列）と、意味の指標の基準にする、名詞を人工的に2・3トークンに分けた形
# （例: pas|ta。| はトークンの区切り）
uv run lexical-acquisition-mechanisms evaluation-data build-probe-and-normalization-words

# 最小対（手がかりの1語だけが違い、ターゲット語の語義が変わる2文。例: "She peeled the car." /
# "She parked the car."）の候補を、元の語義と、足した1つの語義の組ごとに、テンプレートと LLM で作る
uv run lexical-acquisition-mechanisms evaluation-data generate-minimal-pairs         # LLM

# 課題の問題を作る（語義の同定と属性の判定。問題の前に例を見せる）。LLM は、
# ターゲット語の元の語義を表す言い換えの語（例: car → automobile）の候補を出す
uv run lexical-acquisition-mechanisms evaluation-data build-task-questions           # LLM

# 学習前の3モデルで、次を確かめる。最小対の各文で、手がかりが示す語義をもとから持つ語の方が
# 空欄に出やすいか。課題が自然な語で解けるか。言い換えの語が、元の語に対して選ばれるか。
# 通った最小対から一部を抜き出し、人手確認用の data/evaluation/minimal_pairs_review.csv に書き出す
uv run lexical-acquisition-mechanisms evaluation-data check-with-pretrained-models   # GPU
```

**手作業**:

1. 問題の前に見せる例の数（暫定: 3）を、`evaluation-data check-with-pretrained-models` が出す、自然な語での正解率を見て決め、`settings.yaml` の `evaluation` に書く。変えたら、`evaluation-data build-task-questions` から実行し直す。
2. `data/evaluation/minimal_pairs_review.csv` の `approved` 列を埋める（1行1組。それぞれの文で、ターゲット語の意味がその行に書かれた語義に決まれば 1、そうでなければ 0）。

```bash
# approved が 0 の最小対を除き、評価で使う最小対と課題の問題を書き出す
uv run lexical-acquisition-mechanisms evaluation-data finalize \
  --review-csv data/evaluation/minimal_pairs_review.csv
```

- 課題が自然な語で解けなかったモデルでは、その課題を使うコマンドは、そのことを記録するだけで終わる。

### 2. パイロット（7B）

学習の設定を確定するための、試しの学習（2つの run）です。`run-1` は未知語の獲得の run の1つ目、`control` は何も差し込まない（基データだけで学習する）対照の run です。

```bash
MODEL=OLMo-2-1124-7B
export CUDA_VISIBLE_DEVICES=0,1   # 使う GPU（LLM のサーバが使っていないもの）
NUMBER_OF_GPUS=2                  # 学習に使う GPU の数（上で選んだ GPU の数）

# run ごとの学習の設定ファイル（モデル・学習データ・学習率など）を書き出す
uv run lexical-acquisition-mechanisms training write-run-settings \
  --experiment unknown-word-acquisition --model "$MODEL"

# 2つの run を継続事前学習する
uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
  training run --run-settings "outputs/runs/unknown-word-acquisition/$MODEL/run-1.yaml"     # GPU
uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
  training run --run-settings "outputs/runs/unknown-word-acquisition/$MODEL/control.yaml"   # GPU

# 学習前のモデルで、内部表現から実在語かどうかを当てるプローブを、層ごと・トークン数ごとに学習し、
# 学習に使わなかった語での正解率を出す
uv run lexical-acquisition-mechanisms unknown-word-acquisition train-real-word-probes \
  --model "$MODEL"   # GPU
```

- 次のことを確かめる。
  - 学習の記録 `outputs/checkpoints/unknown-word-acquisition/OLMo-2-1124-7B/<run>/metrics.jsonl` で、パープレキシティの変化が1%以内で、差し込んだ用例の損失が下がり、損失が急に跳ね上がったり数値でない値（NaN）になったりしていない
  - プローブが、学習に使わなかった語でも当てられる
- 結果を見て、学習率・バッチサイズ・ウォームアップ・1トークンの未知語の埋め込みの初期値を確定し、`settings.yaml` の `training` に書く。学習率は、1B・13B にも当てはめる決め方（暫定: 各モデルの事前学習のステージ1の最後の学習率）として決める。
- 値を変えたら、`training write-run-settings` から2つの run をやり直す。

### 3. 未知語の獲得（7B）

9つの run（8つの run と対照の run）を学習し、3つの指標を測ります。

```bash
MODEL=OLMo-2-1124-7B
export CUDA_VISIBLE_DEVICES=0,1   # 使う GPU（LLM のサーバが使っていないもの）
NUMBER_OF_GPUS=2                  # 学習に使う GPU の数（上で選んだ GPU の数）

# 9つの run を継続事前学習する（学習済みの run は省かれる）
uv run lexical-acquisition-mechanisms training write-run-settings \
  --experiment unknown-word-acquisition --model "$MODEL"
for run_settings in outputs/runs/unknown-word-acquisition/"$MODEL"/*.yaml; do
  uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
    training run --run-settings "$run_settings" || break   # GPU
done
```

すべての run の学習が終わってから、次を実行します。

```bash
# 実在語かどうかを当てるプローブを学習する（パイロットで学習済みなら省かれる）
uv run lexical-acquisition-mechanisms unknown-word-acquisition train-real-word-probes \
  --model "$MODEL"   # GPU

# 各 run の学習後のモデルで、中立文脈20種と文脈なしの後に未知語を置き、3つの指標を層ごとに測る:
# 形への慣れ（2トークン目以降の予測しやすさ）、意味（元の名詞の内部表現とのコサイン類似度）、
# 語としての統合（プローブが実在語と判定する確率）。課題の答えも記録する
uv run lexical-acquisition-mechanisms unknown-word-acquisition measure-indicators \
  --model "$MODEL"   # GPU

# 3つの指標を、差し込み0回の未知語が0、既知の語が1になるように正規化し、差し込んだ回数に対する
# 曲線を当てはめて、3つの指標が獲得される順番を検定する
uv run lexical-acquisition-mechanisms unknown-word-acquisition analyze-acquisition-order \
  --model "$MODEL"

# 指標の推移の図と、トークン数・綴りの近さ・モデルサイズ別の比較の表を作る
uv run lexical-acquisition-mechanisms unknown-word-acquisition make-report
```

- 語義の追加で、足す語義1つあたりに差し込む用例の数 K を決める。`outputs/figures/unknown-word-acquisition/` の意味の曲線を見て、それ以上差し込んでもほとんど上がらない回数を選び、理由とともに `settings.yaml` の `sense_addition` に書く。

### 4. 語義の追加（7B）

11の run（足す語義の種類3つ × 数1・2・4の9つと、名詞用・形容詞用の対照2つ）を学習し、語義を選ぶ head を特定します。

```bash
MODEL=OLMo-2-1124-7B
export CUDA_VISIBLE_DEVICES=0,1   # 使う GPU（LLM のサーバが使っていないもの）
NUMBER_OF_GPUS=2                  # 学習に使う GPU の数（上で選んだ GPU の数）

# 形容詞の用例を、K に合わせて追加する（足りていれば省かれる）
uv run lexical-acquisition-mechanisms passages collect
uv run lexical-acquisition-mechanisms passages remove-duplicates-and-split
uv run lexical-acquisition-mechanisms passages judge-word-sense                     # LLM
uv run lexical-acquisition-mechanisms passages sample-for-review
```

**手作業**: `data/passages/review.csv` に足された行の `approved` 列を埋める（基準は 1-3 と同じ）。

```bash
# 1-3 と同じく、語ごとの判定の精度と用例の数を確かめ、足した用例を書き出す
uv run lexical-acquisition-mechanisms passages finalize --review-csv data/passages/review.csv
```

- ここで語が外れたときは、`words select` から順に実行し直してから、手順4の最初に戻る（`words select` は、外れた語だけを予備の候補と入れ替える）。このとき、未知語の獲得の学習データが作り直されたら（`training-data build --experiment unknown-word-acquisition` が省かれなかったら）、7B の手順3もやり直す。

```bash
# 名詞の run と名詞用の対照の run には名詞72語の用例を各 K 件、形容詞の run と形容詞用の対照の
# run には形容詞72語の用例を各 K 件、基データに差し込んだ学習データを作る。割り当てのある語の
# 用例だけ、その語をターゲット語に置き換える（対照の run では置き換えない）
uv run lexical-acquisition-mechanisms training-data build --experiment sense-addition

# 11の run を継続事前学習する
uv run lexical-acquisition-mechanisms training write-run-settings \
  --experiment sense-addition --model "$MODEL"
for run_settings in outputs/runs/sense-addition/"$MODEL"/*.yaml; do
  uv run torchrun --nproc-per-node "$NUMBER_OF_GPUS" -m lexical_acquisition_mechanisms \
    training run --run-settings "$run_settings" || break   # GPU
done
```

すべての run の学習が終わってから、次を実行します。

```bash
# 最小対の各文で語義の同定の正解率を測り、同じ用例を置き換えずに差し込んだ対照の run と比べる
# （使い分けられていない語義は、以降の分析から外す）
uv run lexical-acquisition-mechanisms sense-addition measure-sense-accuracy \
  --model "$MODEL"   # GPU

# 最小対の一方の文を処理するときに、各 attention head と MLP（各層の全結合のブロック）の、
# ターゲット語の位置の出力を、もう一方の文での出力に入れ替え、語義の同定の答えがどれだけ
# 切り替わるかを測り、対照の run での値を引く（補足として、ターゲット語より後の全位置でも測る）
uv run lexical-acquisition-mechanisms sense-addition measure-head-and-mlp-effects \
  --model "$MODEL"   # GPU

# 効果の信頼区間と多重比較の補正で、効いている head と MLP を選び、どの種類の語義を
# 足したときに効くか（3種類すべて、別のカテゴリの名詞と形容詞だけ、など）で分類する
uv run lexical-acquisition-mechanisms sense-addition select-significant-heads-and-mlps \
  --model "$MODEL"

# 選んだ head ごとに、学習前のモデルで、自然な語の文での head の出力から名詞のカテゴリと品詞を
# 当てるプローブを学習する。学習に使わなかった自然な語と、語義を足したモデルの最小対の文で
# 当てられるかを確かめる
uv run lexical-acquisition-mechanisms sense-addition train-attribute-probes \
  --model "$MODEL"   # GPU

# 属性の判定の問題を処理するときに、head の出力のうち、プローブの重みが示す属性の方向の成分
# だけを、もう一方の文での値に入れ替え、答えが変わる割合を、無作為な head・方向のときと比べる。
# プローブで当てられ、入れ替えで答えが変わり、その属性が違う種類の語義でだけ効く head を、
# 属性を判定する head として、属性の方向とともに保存する
uv run lexical-acquisition-mechanisms sense-addition test-attribute-directions \
  --model "$MODEL"   # GPU

# 層 × head の効果の地図と、足した語義の数・トークン数・モデルサイズ別の比較の表と図を作る
uv run lexical-acquisition-mechanisms sense-addition make-report
```

- 最小対の数（暫定: 元の語義と足した語義の組ごとに20組）は、`sense-addition select-significant-heads-and-mlps` が出す信頼区間の幅を見て、広すぎれば増やし、`settings.yaml` の `evaluation` に書く。増やしたら、`evaluation-data generate-minimal-pairs` から `evaluation-data finalize` までと、`sense-addition measure-sense-accuracy` 以降を実行し直す。

### 5. カテゴリ判定 head への未知語の依存（7B）

追加の学習はせず、手順3で学習したモデルと、手順4で保存した head を使います。

```bash
MODEL=OLMo-2-1124-7B
export CUDA_VISIBLE_DEVICES=0   # 使う GPU（LLM のサーバが使っていないもの）

# 未知語の獲得で学習した各 run のモデルで、中立文脈の文で属性の判定（カテゴリ）の問題を解かせる。
# 名詞のカテゴリを判定する head（手順4で保存）をすべて同時に無効化し、未知語と元の名詞の
# それぞれで、正しいカテゴリの確率がどれだけ下がるかを測る（依存度 = 未知語での低下 /
# 元の名詞での低下）。無作為な head・方向でも同じ操作をして比べる
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads measure-dependence \
  --model "$MODEL"   # GPU

# 依存度を、意味と語としての統合の正規化した値・トークン数・綴りの近さで説明する回帰
# （未知語・名詞・run ごとのばらつきを考慮した混合効果モデル）を当てはめる。
# 意味と統合の値を閾値で区切った状態ごとの比較もする
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads fit-regression \
  --model "$MODEL"

# 同じ名詞の意味を、未知語に付けた場合と既知語に足した場合とで、効いている head が同じかを比べる。
# 語義の同定の問題で、未知語の位置の各 head の出力を、差し込み0回の未知語での出力に入れ替えた
# ときの正解の確率の低下を測り、手順4の head の地図と、順位相関と上位の head の重なりで比べる
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads compare-head-maps \
  --model "$MODEL"   # GPU

# 意味・語としての統合と、依存度の関係の図と表を作る
uv run lexical-acquisition-mechanisms dependence-on-category-judging-heads make-report
```

- 手順4でカテゴリを判定する head が見つからなかったモデルでは、`dependence-on-category-judging-heads measure-dependence` と `dependence-on-category-judging-heads fit-regression` は、そのことを記録するだけで終わる。

### 6. 1B・13B

- 学習率は、パイロットで決めた決め方で 1B・13B の値を求め、`settings.yaml` の `training` に書く。
- 手順3〜5を、各手順の最初の変数を次のように替えて繰り返す（手順5は `MODEL` と `CUDA_VISIBLE_DEVICES` だけ）。
  - 1B: `MODEL=OLMo-2-0425-1B`、GPU 1枚（`CUDA_VISIBLE_DEVICES` に1枚、`NUMBER_OF_GPUS=1`）
  - 13B: `MODEL=OLMo-2-1124-13B`、GPU 4枚（`CUDA_VISIBLE_DEVICES` に4枚、`NUMBER_OF_GPUS=4`）
- パイロットと、K・最小対の数の決定は行わず、7B で決めた値を使う。手順4の用例の追加は省かれるので、LLM のサーバは要らない。

### 7. 集計

```bash
# 全結果を1つの表 outputs/results/all_results.parquet（Parquet 形式の表）にまとめ、
# 図を outputs/figures/ に作る
uv run lexical-acquisition-mechanisms results collect
```

## テスト

```bash
uv run pytest
```

乱数で初期化した小さなモデルと、LLM や外部のサービス（語の出現回数の検索、非語のデータベースなど）の代わりに決まった応答を返す部品を使い、GPU なしで全コマンドを通して実行します。
