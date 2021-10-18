import torch
from torch import nn
import torch.nn.functional as F

from hw_asr.base import BaseModel

class Block(nn.Module):
    def __init__(self, n_subblocks, p, in_ch, out_ch, kernel_size, stride=1, dilation=1): #TODO:stride default?
        super().__init__()
        self.last_conv1d = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size,
            stride=stride, dilation=dilation) #TODO: padding?
        self.last_batch_norm = nn.BatchNorm1d(out_ch)
        self.last_relu = nn.ReLU()
        self.last_dropout = nn.Dropout(p)
        
        self.residual_process = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch)
        )
        layers = []
        for _ in range(n_subblocks - 1):
            layers.append(nn.Conv1d(
                in_ch, out_ch, kernel_size=kernel_size,
                stride=stride, dilation=dilation)) #TODO: padding?
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p))
            in_ch = out_ch
        self.gobrrrrr = nn.Sequential(*layers)

    def forward(self, input):
        x_residual = self.residual_process(input)
        x = self.gobrrrrr(input)

        x = self.last_conv1d(x)
        x = self.last_batch_norm(x)
        # x = torch.cat((x, x_residual), 1)
        # print(x.shape, x_residual.shape[2] - x.shape[2])
        x = F.pad(x, (0, 0, 0, 0, 0, x_residual.shape[2] - x.shape[2]))
        # print(x.shape, x_residual.shape[2] - x.shape[2])
        
        x = x + x_residual
        x = self.last_relu(x)
        x = self.last_dropout(x)
        return x


class JasperModel(BaseModel):
    #TODO: 20ms windows with a 10ms overlap; We use 40 features for WSJ and 64 for LibriSpeech and F+S
    def __init__(self, n_feats, n_class):
        super().__init__(n_feats, n_class)
        self.Conv1 = Block(n_subblocks=1, p=0.2, in_ch=1   , out_ch=256 , kernel_size=11, stride=2) #TODO: in_ch?
        self.B1    = Block(n_subblocks=5, p=0.2, in_ch=256 , out_ch=256 , kernel_size=11)
        self.B2    = Block(n_subblocks=5, p=0.2, in_ch=256 , out_ch=384 , kernel_size=13)
        self.B3    = Block(n_subblocks=5, p=0.2, in_ch=384 , out_ch=512 , kernel_size=17)
        self.B4    = Block(n_subblocks=5, p=0.3, in_ch=512 , out_ch=640 , kernel_size=21)
        self.B5    = Block(n_subblocks=5, p=0.3, in_ch=640 , out_ch=768 , kernel_size=25)
        self.Conv2 = Block(n_subblocks=1, p=0.4, in_ch=768 , out_ch=896 , kernel_size=29, dilation=2)
        self.Conv3 = Block(n_subblocks=1, p=0.4, in_ch=768 , out_ch=1024, kernel_size=1 )
        self.Conv4 = Block(n_subblocks=1, p=0.0, in_ch=1024, out_ch=n_class, kernel_size=1)

        self.B1_   = Block(n_subblocks=5, p=0.2, in_ch=256 , out_ch=256 , kernel_size=11)
        self.B2_   = Block(n_subblocks=5, p=0.2, in_ch=384 , out_ch=384 , kernel_size=13)
        self.B3_   = Block(n_subblocks=5, p=0.2, in_ch=512 , out_ch=512 , kernel_size=17)
        self.B4_   = Block(n_subblocks=5, p=0.3, in_ch=640 , out_ch=640 , kernel_size=21)
        self.B5_   = Block(n_subblocks=5, p=0.3, in_ch=768 , out_ch=768 , kernel_size=25)

        self.gobrrrrr = nn.Sequential(
              self.Conv1 
            , self.B1
            , self.B1_
            , self.B2
            , self.B2_
            , self.B3
            , self.B3_
            , self.B4
            , self.B4_
            , self.B5
            , self.B5_
            , self.Conv2 
            , self.Conv3 
            , self.Conv4
        )

    def forward(self, spectrogram, *args, **kwargs):
        spectrogram = torch.unsqueeze(spectrogram, 1)
        spectrogram = torch.flatten(spectrogram, start_dim=2)
        return self.gobrrrrr(spectrogram)

    def transform_input_lengths(self, input_lengths):
        return input_lengths  #TODO: Do we reduce time dimension here?
