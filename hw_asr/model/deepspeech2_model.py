import torch
from torch import nn
import torch.nn.functional as F

from hw_asr.base import BaseModel

class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
              nn.Conv2d(ch, ch, 3, padding='same')
            , nn.BatchNorm2d(ch)
            , nn.ReLU()
            , nn.Dropout(0.1)
            , nn.Conv2d(ch, ch, 3, padding='same')
            , nn.BatchNorm2d(ch)
            , nn.ReLU()
            , nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return x + self.net(x)

class RNNBlock(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=512, num_layers=1, bidirectional=True)
        self.dp = nn.Dropout(0.1)
    
    def forward(self, x):
        x, _ = self.gru(x)
        return self.dp(x)

class DeepSpeech2Model(BaseModel):
    def __init__(self, n_feats, n_class):
        super().__init__(n_feats, n_class)
       
        self.cnn = nn.Sequential(
              nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
            , ResBlock(32)
            , ResBlock(32)
            , ResBlock(32)
        )

        self.rnn = nn.Sequential(
            RNNBlock(32*n_feats//2),
            RNNBlock(2*512),
            RNNBlock(2*512),
            RNNBlock(2*512),
            RNNBlock(2*512),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        x = spectrogram.unsqueeze(1)
        x = x.transpose(2, 3).contiguous() # (batch, channel, feature, time)
        x = self.cnn(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]) # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1) # (time, batch, features)
        x = self.rnn(x)
        x = self.classifier(x)
        x = x.transpose(0, 1)
        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths//2
