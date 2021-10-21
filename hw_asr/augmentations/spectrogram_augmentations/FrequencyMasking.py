import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.FrequencyMasking(*args, **kwargs) #(freq_mask_param=15)

    def __call__(self, data: Tensor):
        return self._aug(data)