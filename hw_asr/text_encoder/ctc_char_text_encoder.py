from typing import List, Tuple

import torch
import numpy as np

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from pyctcdecode import build_ctcdecoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.decoder = build_ctcdecoder(
            [""] + [a.upper() for a in alphabet],
            "3-gram.arpa",
            alpha=1,
            beta=0,
        )
    def ctc_decode(self, inds: List[int]) -> str:
        prev = 0
        result = ""
        for i in inds:
            if i != 0 and i != prev:
                result += self.ind2char[i.item()]
            prev = i
        return result

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2, probs.shape
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = []

        res = self.decoder.decode_beams(probs[:probs_length, :], beam_width=beam_size)
        for row in res:
            hypos.append((row[0], row[-1]))
        return hypos
