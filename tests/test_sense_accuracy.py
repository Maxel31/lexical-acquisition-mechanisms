"""語義の使い分けの正解率のテスト。モデルは、決まった答えを返す部品で置き換える。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.sense_accuracy import SenseAccuracy, write_sense_accuracy
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.task_prompts import TaskItem, prompt_template
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer


class ModelReadingCues:
    """語義を足したモデルは手がかりどおりに、対照のモデルはいつも元の語義を答える。"""

    def __init__(self, tokenizer: WordTokenizer, learned: bool) -> None:
        self.tokenizer = tokenizer
        self.learned = learned

    def next_token_log_probabilities(self, inputs, positions, candidate_ids) -> np.ndarray:
        values = np.zeros((len(inputs), len(candidate_ids)), dtype=np.float32)
        pasta, automobile = (
            self.tokenizer.word_token_ids(word)[0] for word in ("pasta", "automobile")
        )
        for row, token_ids in enumerate(inputs):
            last = self.tokenizer.decode(token_ids).split("Sentence:")[-1]
            wanted = pasta if self.learned and "cooked" in last else automobile
            values[row, candidate_ids.index(wanted)] = 1.0
        return values


def test_accuracy_is_compared_with_the_control(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    car = " ".join(map(str, tokenizer.word_token_ids("car")))
    write_rows(tmp_path / "words.csv", SelectedWord, [SelectedWord("car", "target", token_ids=car)])
    questions = []
    for donor_sentence, target_sentence in [
        ("She cooked the {word}.", "She parked the {word}."),
        ("We cooked the {word}.", "We parked the {word}."),
    ]:
        for cue, sentence in (("donor", donor_sentence), ("target", target_sentence)):
            answer = "pasta" if cue == "donor" else "automobile"
            item = TaskItem("sense", sentence, ("automobile", "pasta"), answer)
            questions.append(
                {
                    "kind": "other-category",
                    "number_of_senses": 1,
                    "target": "car",
                    "donor": "pasta",
                    "donor_sentence": donor_sentence,
                    "cue": cue,
                    "task": "sense",
                    "prompt_template": prompt_template([], item),
                    "choices": ["automobile", "pasta"],
                    "answer": answer,
                }
            )
    parquet.write_table(arrow.Table.from_pylist(questions), tmp_path / "questions.parquet")
    checkpoints = {"other-category-1": Path("learned"), "noun-control": Path("control")}

    write_sense_accuracy(
        checkpoints,
        lambda source: ModelReadingCues(tokenizer, learned=source.repository == "learned"),
        tmp_path / "words.csv",
        tmp_path / "questions.parquet",
        (tmp_path / "answers.parquet", tmp_path / "accuracy.csv"),
    )

    accuracy = {row.sense: row for row in read_rows(tmp_path / "accuracy.csv", SenseAccuracy)}
    # 足した語義（pasta）は、語義を足したモデルでだけ当たる
    assert (accuracy["pasta"].accuracy, accuracy["pasta"].control_accuracy) == (1.0, 0.0)
    assert accuracy["pasta"].difference == 1.0
    assert accuracy["original"].accuracy == 1.0 and accuracy["original"].control_accuracy == 1.0
    assert accuracy["pasta"].chance == pytest.approx(0.5)
    answers = parquet.read_table(tmp_path / "answers.parquet").to_pylist()
    assert len(answers) == 2 * len(questions)
