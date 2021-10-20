import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeMasking(*args, **kwargs) #(time_mask_param=35)

    def __call__(self, data: Tensor):
        return self._aug(data)
