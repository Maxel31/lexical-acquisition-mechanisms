"""ターゲット語と名詞の組ごとに内部表現の類似度を測り、確認用の一覧を書く
（words measure-hidden-state-similarity）。

1. 公開されている各モデルで、ターゲット語と名詞を中立文脈の各条件（文脈なしを含む）の直後に置き、
   語の最後のトークンの位置の、中間の層の残差ストリームを取り出す。
2. 組ごとに、条件ごとの cos 類似度を出し、条件で平均する。
3. 同じカテゴリの組が上位に入るかを、内部表現の類似度（モデルごと）と LLM の評定のそれぞれで
   確かめる。指標は、同じカテゴリの組と別のカテゴリの組を1つずつ選んだとき、同じカテゴリの組の
   値が大きい確率（同じなら半分と数える）。
4. 人が確かめる data/words/review.csv を書く。管理対象の実在語と自然な語を1行ずつ並べ、ターゲット語
   と名詞には、類似度の大きい相手のうち、同じカテゴリのものの割合を付ける（カテゴリの誤りの手がかり）。
"""

import argparse
import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lexical_acquisition_mechanisms import neutral_contexts, similarity_ratings, word_selection
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_review_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.similarity_ratings import SimilarityRating
from lexical_acquisition_mechanisms.word_selection import SelectedWord

COMMAND = "words measure-hidden-state-similarity"
SIMILARITY_PATH = Path("data/words/hidden_state_similarity.csv")
CHECKS_PATH = Path("data/words/similarity_checks.csv")
REVIEW_PATH = Path("data/words/review.csv")
BATCH_SIZE = 64
REVIEWED_ROLES = ("noun", "target", "adjective", "natural-noun", "natural-adjective")


@dataclass
class HiddenStateSimilarity:
    target: str
    target_category: str
    noun: str
    noun_category: str
    model: str
    # 中立文脈の各条件で平均した、中間の層の残差ストリームの cos 類似度
    cosine_similarity: float


@dataclass
class SimilarityCheck:
    # 類似度の種類（モデルの名前、または llm）
    measure: str
    # 同じカテゴリの組の値が、別のカテゴリの組の値より大きい確率
    area_under_curve: float
    mean_same_category: float
    mean_other_category: float


@dataclass
class ReviewRow:
    word: str
    role: str
    category: str
    synset: str
    definition: str
    # ターゲット語と名詞だけ: 類似度（全モデルの平均）の大きい相手のうち、同じカテゴリのものの割合
    # （相手は、同じカテゴリの相手の数だけ取る）
    same_category_share_of_nearest: str = ""
    # 人が記入する（その行のカテゴリ・語義に合えば 1、合わなければ 0）
    approved: str = ""


def measure_hidden_state_similarity(settings: Settings, arguments: argparse.Namespace) -> None:
    step = Step(
        COMMAND,
        inputs={
            word_selection.OUTPUT_PATH: word_selection.COMMAND,
            neutral_contexts.CONTEXTS_PATH: neutral_contexts.COMMAND,
            similarity_ratings.OUTPUT_PATH: similarity_ratings.COMMAND,
        },
        outputs=[SIMILARITY_PATH, CHECKS_PATH, REVIEW_PATH],
        settings={
            "models": {name: source.model_dump() for name, source in settings.models.items()}
        },
    )
    run_unless_unchanged(
        step,
        lambda: build_hidden_state_similarity(
            settings,
            _load_model,
            word_selection.OUTPUT_PATH,
            neutral_contexts.CONTEXTS_PATH,
            similarity_ratings.OUTPUT_PATH,
            (SIMILARITY_PATH, CHECKS_PATH, REVIEW_PATH),
        ),
        arguments.recompute,
    )


def build_hidden_state_similarity(
    settings: Settings,
    load_model: Callable[[ModelSource], LanguageModel],
    selected_words_path: Path,
    contexts_path: Path,
    ratings_path: Path,
    output_paths: tuple[Path, Path, Path],
) -> None:
    similarity_path, checks_path, review_path = output_paths
    words = read_rows(selected_words_path, SelectedWord)
    targets = [word for word in words if word.role == "target"]
    nouns = [word for word in words if word.role == "noun"]
    contexts = [row.context for row in read_rows(contexts_path, NeutralContext)]
    same = np.array([[target.category == noun.category for noun in nouns] for target in targets])
    cosines = {}
    for name, source in settings.models.items():
        model = load_model(source)
        target_states = _middle_layer_states(model, targets, contexts)
        noun_states = _middle_layer_states(model, nouns, contexts)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        # 形は ターゲット語 × 名詞 × 条件。条件で平均する
        cosines[name] = np.einsum("tcd,ncd->tnc", target_states, noun_states).mean(axis=2)
    write_rows(
        similarity_path,
        HiddenStateSimilarity,
        [
            HiddenStateSimilarity(
                target.word,
                target.category,
                noun.word,
                noun.category,
                name,
                float(values[target_index, noun_index]),
            )
            for name, values in cosines.items()
            for target_index, target in enumerate(targets)
            for noun_index, noun in enumerate(nouns)
        ],
    )
    ratings = {
        (rating.target, rating.noun): rating.llm_similarity
        for rating in read_rows(ratings_path, SimilarityRating)
    }
    llm = np.array(
        [[ratings[(target.word, noun.word)] for noun in nouns] for target in targets], dtype=float
    )
    measures = {**cosines, "llm": np.where(llm > 0, llm, np.nan)}
    checks = [_check(name, values, same) for name, values in measures.items()]
    write_rows(checks_path, SimilarityCheck, checks)
    for check in checks:
        print(
            f"{check.measure}: 同じカテゴリの組の値が、別のカテゴリの組の値より大きい確率"
            f" {check.area_under_curve:.3f}"
        )
    mean_cosine = np.mean(list(cosines.values()), axis=0)
    shares = {
        **_nearest_same_category_shares(targets, mean_cosine, same),
        **_nearest_same_category_shares(nouns, mean_cosine.T, same.T),
    }
    review = [
        ReviewRow(
            word.word,
            word.role,
            word.category,
            word.synset,
            word.definition,
            shares.get(word.word, ""),
        )
        for word in words
        if word.role in REVIEWED_ROLES
    ]
    write_review_rows(review_path, ReviewRow, review, ("word", "role", "category", "synset"))


def _middle_layer_states(
    model: LanguageModel, words: Sequence[SelectedWord], contexts: Sequence[str]
) -> np.ndarray:
    """語ごと・条件ごとの、中間の層の残差ストリーム（長さ1に正規化。形は 語 × 条件 × 次元）。"""
    inputs = []
    positions = []
    for word in words:
        token_ids = [int(token) for token in word.token_ids.split()]
        for context in contexts:
            parts = [context, token_ids] if context else [token_ids]
            ids, spans = model.tokenizer.encode_with_spans(parts)
            inputs.append(ids)
            positions.append(spans[-1][-1])
    states = model.residual_streams(inputs, positions)[:, model.number_of_layers // 2]
    states = states / np.linalg.norm(states, axis=1, keepdims=True)
    return states.reshape(len(words), len(contexts), -1)


def _check(measure: str, values: np.ndarray, same: np.ndarray) -> SimilarityCheck:
    same_values = values[same & ~np.isnan(values)]
    other_values = values[~same & ~np.isnan(values)]
    greater = (same_values[:, None] > other_values[None, :]).mean()
    ties = (same_values[:, None] == other_values[None, :]).mean()
    return SimilarityCheck(
        measure,
        float(greater + ties / 2),
        float(same_values.mean()),
        float(other_values.mean()),
    )


def _nearest_same_category_shares(
    words: Sequence[SelectedWord], values: np.ndarray, same: np.ndarray
) -> dict[str, str]:
    """語ごとに、類似度の大きい相手（同じカテゴリの相手の数だけ）のうち、同じカテゴリの割合。"""
    shares = {}
    for index, word in enumerate(words):
        count = int(same[index].sum())
        nearest = np.argsort(-values[index])[:count]
        shares[word.word] = f"{same[index, nearest].mean():.2f}" if count else ""
    return shares


def _load_model(source: ModelSource) -> LanguageModel:
    return LanguageModel.load(source, batch_size=BATCH_SIZE)
