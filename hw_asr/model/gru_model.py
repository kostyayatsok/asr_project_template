from torch import nn
from torch.nn import Sequential, GRU

from hw_asr.base import BaseModel


class GRUModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = GRU(input_size=n_feats, hidden_size=fc_hidden, num_layers=2, dropout=0, batch_first=True)
        self.tail = Sequential(
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden//2),
            nn.ReLU(),
            nn.Linear(fc_hidden//2, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        out, h = self.net(spectrogram)
        return {"logits":self.tail(out)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
