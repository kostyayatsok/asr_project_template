from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length:Tensor, text: List[str], *args, **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1)
        predictions = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(predictions, log_probs_length)
        ]
        for log_prob_vec, target_text in zip(predictions, text):
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec.numpy())
            else:
                pred_text = self.text_encoder.decode(log_prob_vec.numpy())
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)

class BeamsearchCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length:Tensor, text: List[str], *args, **kwargs):
        cers = []
        for log_prob, log_prob_len, target_text in zip(log_probs.cpu().numpy(), log_probs_length, text):
            pred_text = self.text_encoder.ctc_beam_search(log_prob, log_prob_len)[0][0]
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)