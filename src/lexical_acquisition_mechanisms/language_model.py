"""OLMo 2 の読み込みと、内部表現・確率の取り出し。

入力は、トークン番号の列で渡す（標準とは違う区切りや、1トークンの未知語のトークン番号を差し込めるように）。
各入力の先頭には、文書の区切り <|endoftext|> を置いてからモデルに入れる（OLMo 2 は、文書をこの
トークンで区切って学習しているため）。位置は、先頭に置いた区切りを数えずに、渡した列の中で数える。
測定は、既定で float32 で行う（bf16 では対数確率が 0.06 程度丸められ、小さな効果を測れないため）。
"""

from collections.abc import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, PreTrainedModel

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
        dtype: torch.dtype = torch.float32,
        batch_size: int = 16,
    ) -> "LanguageModel":
        model = AutoModelForCausalLM.from_pretrained(
            source.repository, revision=source.revision, dtype=dtype
        )
        return cls(model.to(device), WordTokenizer.from_model(source), batch_size)

    @property
    def number_of_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def maximum_input_length(self) -> int:
        """渡せる入力の長さの上限（先頭に置く区切りの分を除く）。"""
        return self.model.config.max_position_embeddings - 1

    def residual_streams(
        self, inputs: Sequence[Sequence[int]], positions: Sequence[int]
    ) -> np.ndarray:
        """各入力の指定した位置での、全層の残差ストリーム。

        形は (入力の数, 層の数 + 1, 隠れ次元)。0番目は埋め込み、i 番目は第 i 層の出力（最後の層も、
        最終の正規化の前の値）。
        """
        self._check_inputs(inputs, positions)
        results = np.zeros(
            (len(inputs), self.number_of_layers + 1, self.model.config.hidden_size),
            dtype=np.float32,
        )
        for indices, token_ids, attention_mask in self._batches(inputs):
            rows, columns = self._rows_and_columns(indices, positions)
            outputs = self._block_outputs(token_ids, attention_mask, rows, columns)
            results[indices] = outputs.numpy()
        return results

    def _block_outputs(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rows: torch.Tensor,
        columns: torch.Tensor,
    ) -> torch.Tensor:
        """埋め込みと各層の出力を、指定した行と位置で取り出す（形は 行 × 層の数 + 1 × 次元）。"""
        captured: list[torch.Tensor] = []

        def capture(module, arguments, output: torch.Tensor) -> None:
            captured.append(output[rows, columns].float().cpu())

        blocks = [self.model.model.embed_tokens, *self.model.model.layers]
        handles = [block.register_forward_hook(capture) for block in blocks]
        try:
            with torch.inference_mode():
                self.model.model(
                    input_ids=token_ids, attention_mask=attention_mask, use_cache=False
                )
        finally:
            for handle in handles:
                handle.remove()
        return torch.stack(captured, dim=1)

    def next_token_log_probabilities(
        self,
        inputs: Sequence[Sequence[int]],
        positions: Sequence[int],
        candidate_token_ids: Sequence[int],
    ) -> np.ndarray:
        """各入力の指定した位置の次に来るトークンの対数確率を、候補ごとに返す。

        形は (入力の数, 候補の数)。例えば、課題の "Answer:" の位置を指定して、選択肢の語を比べる。
        """
        self._check_inputs(inputs, positions)
        if not candidate_token_ids:
            raise ValueError("候補のトークン番号がありません。")
        self.tokenizer.check_token_ids(candidate_token_ids)
        candidates = torch.tensor(candidate_token_ids, dtype=torch.long, device=self.model.device)
        results = np.zeros((len(inputs), len(candidate_token_ids)), dtype=np.float32)
        for indices, token_ids, attention_mask in self._batches(inputs):
            rows, columns = self._rows_and_columns(indices, positions)
            with torch.inference_mode():
                hidden = self._last_hidden_state(token_ids, attention_mask)
                logits = self.model.lm_head(hidden[rows, columns]).float()
                log_probabilities = torch.log_softmax(logits, dim=-1)[:, candidates]
            results[indices] = log_probabilities.cpu().numpy()
        return results

    def token_log_probabilities(self, inputs: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """各入力の各トークンの対数確率 log p(t_k | 区切り, t_1, …, t_{k-1})。

        入力ごとに、入力と同じ長さの配列を返す。驚き度は、この値の符号を変えたもの。
        """
        self._check_inputs(inputs, [0] * len(inputs))
        results: list[np.ndarray] = [np.zeros(0, dtype=np.float32)] * len(inputs)
        for indices, token_ids, attention_mask in self._batches(inputs):
            with torch.inference_mode():
                hidden = self._last_hidden_state(token_ids, attention_mask)
                for row, index in enumerate(indices):
                    length = len(inputs[index])
                    # 入力の k 番目のトークンは、区切りを先頭に足した列の k 番目の出力が予測する
                    logits = self.model.lm_head(hidden[row, :length]).float()
                    targets = token_ids[row, 1 : length + 1]
                    negative = functional.cross_entropy(logits, targets, reduction="none")
                    results[index] = (-negative).cpu().numpy()
        return results

    def _last_hidden_state(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model.model(
            input_ids=token_ids, attention_mask=attention_mask, use_cache=False
        )
        return outputs.last_hidden_state

    def _rows_and_columns(
        self, indices: Sequence[int], positions: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """まとめた入力の中の行と、取り出す位置（先頭の区切りの分、1つずらす）。"""
        rows = torch.arange(len(indices), device=self.model.device)
        columns = torch.tensor(
            [positions[index] + 1 for index in indices], dtype=torch.long, device=self.model.device
        )
        return rows, columns

    def _batches(
        self, inputs: Sequence[Sequence[int]]
    ) -> Iterator[tuple[list[int], torch.Tensor, torch.Tensor]]:
        """入力を長さの順に並べ、batch_size ずつ、右側を詰め物で埋めて返す。"""
        order = sorted(range(len(inputs)), key=lambda index: len(inputs[index]))
        end_of_text = self.tokenizer.end_of_text_id
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            longest = max(len(inputs[index]) for index in indices) + 1
            token_ids = torch.full((len(indices), longest), end_of_text, dtype=torch.long)
            attention_mask = torch.zeros((len(indices), longest), dtype=torch.long)
            for row, index in enumerate(indices):
                sequence = [end_of_text, *inputs[index]]
                token_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
                attention_mask[row, : len(sequence)] = 1
            yield indices, token_ids.to(self.model.device), attention_mask.to(self.model.device)

    def _check_inputs(self, inputs: Sequence[Sequence[int]], positions: Sequence[int]) -> None:
        """全部の入力を、モデルに入れる前に確かめる。"""
        if len(inputs) != len(positions):
            raise ValueError(f"入力の数 {len(inputs)} と、位置の数 {len(positions)} が違います。")
        for token_ids, position in zip(inputs, positions, strict=True):
            if not 1 <= len(token_ids) <= self.maximum_input_length:
                raise ValueError(
                    f"入力の長さ {len(token_ids)} が、1〜{self.maximum_input_length} にありません。"
                )
            if not 0 <= position < len(token_ids):
                raise ValueError(
                    f"位置 {position} が、長さ {len(token_ids)} の入力の外にあります。"
                )
            self.tokenizer.check_token_ids(token_ids)
