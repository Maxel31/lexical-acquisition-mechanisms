"""OLMo 2 の読み込みと、内部表現・確率の取り出し。

入力は、トークン番号の列で渡す（標準とは違う区切りや、1トークンの未知語のトークン番号を差し込めるように）。
各入力の先頭には、文書の区切り <|endoftext|> を置いてからモデルに入れる（OLMo 2 は、文書をこの
トークンで区切って学習しているため）。位置は、先頭に置いた区切りを数えずに、渡した列の中で数える。
"""

from collections.abc import Iterator, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer


class LanguageModel:
    def __init__(self, model: PreTrainedModel, tokenizer: WordTokenizer, batch_size: int = 16):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    @classmethod
    def load(
        cls,
        source: ModelSource,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 16,
    ) -> "LanguageModel":
        model = AutoModelForCausalLM.from_pretrained(
            source.repository, revision=source.revision, dtype=dtype
        )
        return cls(model.to(device), WordTokenizer.from_model(source), batch_size)

    @property
    def number_of_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def residual_streams(
        self, inputs: Sequence[Sequence[int]], positions: Sequence[int]
    ) -> np.ndarray:
        """各入力の指定した位置での、全層の残差ストリーム。

        形は (入力の数, 層の数 + 1, 隠れ次元)。0番目は埋め込み、i 番目は第 i 層の出力
        （最後の層だけは、Transformers の実装どおり、最終の正規化の後の値）。
        """
        _check_positions(inputs, positions)
        results = np.zeros(
            (len(inputs), self.number_of_layers + 1, self.model.config.hidden_size),
            dtype=np.float32,
        )
        for indices, outputs in self._run_in_batches(inputs, output_hidden_states=True):
            layers = torch.stack(outputs.hidden_states, dim=1)  # (入力, 層 + 1, 位置, 次元)
            rows = torch.arange(len(indices))
            columns = torch.tensor([positions[index] + 1 for index in indices])
            results[indices] = layers[rows, :, columns].float().cpu().numpy()
        return results

    def next_token_log_probabilities(
        self,
        inputs: Sequence[Sequence[int]],
        positions: Sequence[int],
        candidate_token_ids: Sequence[int],
    ) -> np.ndarray:
        """各入力の指定した位置の次に来るトークンの対数確率を、候補ごとに返す。

        形は (入力の数, 候補の数)。例えば、課題の "Answer:" の位置を指定して、選択肢の語を比べる。
        """
        _check_positions(inputs, positions)
        candidates = torch.tensor(candidate_token_ids)
        results = np.zeros((len(inputs), len(candidate_token_ids)), dtype=np.float32)
        for indices, outputs in self._run_in_batches(inputs):
            rows = torch.arange(len(indices))
            columns = torch.tensor([positions[index] + 1 for index in indices])
            log_probabilities = torch.log_softmax(outputs.logits[rows, columns].float(), dim=-1)
            results[indices] = log_probabilities[:, candidates.to(log_probabilities.device)].cpu()
        return results

    def token_log_probabilities(self, inputs: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """各入力の各トークンの対数確率 log p(t_k | 区切り, t_1, …, t_{k-1})。

        入力ごとに、入力と同じ長さの配列を返す。驚き度は、この値の符号を変えたもの。
        """
        results: list[np.ndarray] = [np.zeros(0, dtype=np.float32)] * len(inputs)
        for indices, outputs in self._run_in_batches(inputs):
            log_probabilities = torch.log_softmax(outputs.logits.float(), dim=-1)
            for row, index in enumerate(indices):
                length = len(inputs[index])
                targets = torch.tensor(inputs[index], device=log_probabilities.device)
                # 入力の k 番目のトークンは、区切りを先頭に足した列の k 番目の出力が予測する
                predicted = log_probabilities[row, torch.arange(length), targets]
                results[index] = predicted.cpu().numpy()
        return results

    def _run_in_batches(
        self, inputs: Sequence[Sequence[int]], output_hidden_states: bool = False
    ) -> Iterator[tuple[list[int], CausalLMOutputWithPast]]:
        """入力を長さの順に並べ、batch_size ずつ、右側を詰め物で埋めてモデルに通す。"""
        order = sorted(range(len(inputs)), key=lambda index: len(inputs[index]))
        end_of_text = self.tokenizer.end_of_text_id
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            sequences = [[end_of_text, *inputs[index]] for index in indices]
            longest = max(len(sequence) for sequence in sequences)
            token_ids = torch.full((len(sequences), longest), end_of_text, dtype=torch.long)
            attention_mask = torch.zeros((len(sequences), longest), dtype=torch.long)
            for row, sequence in enumerate(sequences):
                self.tokenizer.check_token_ids(sequence)
                token_ids[row, : len(sequence)] = torch.tensor(sequence)
                attention_mask[row, : len(sequence)] = 1
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=token_ids.to(self.model.device),
                    attention_mask=attention_mask.to(self.model.device),
                    output_hidden_states=output_hidden_states,
                )
            yield indices, outputs


def _check_positions(inputs: Sequence[Sequence[int]], positions: Sequence[int]) -> None:
    if len(inputs) != len(positions):
        raise ValueError(f"入力の数 {len(inputs)} と、位置の数 {len(positions)} が違います。")
    for token_ids, position in zip(inputs, positions, strict=True):
        if not 0 <= position < len(token_ids):
            raise ValueError(f"位置 {position} が、長さ {len(token_ids)} の入力の外にあります。")
