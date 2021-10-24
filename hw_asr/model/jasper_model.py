import torch
from torch import nn
import torch.nn.functional as F

from hw_asr.base import BaseModel

class Block(nn.Module):
    def __init__(self, n_subblocks, p, in_ch, out_ch, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size - 1)//2
        if stride == 1 and dilation == 1:
            # padding = (kernel_size // 2) * dilation
            self.use_residual = True
        else:
            self.use_residual = False
        self.residual_process = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.1)
        )
        layers = []
        for _ in range(n_subblocks - 1):
            layers.append(nn.Conv1d(
                in_ch, out_ch, kernel_size=kernel_size,
                stride=stride, dilation=dilation, padding=padding)) #TODO: padding?
            layers.append(nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p))
            in_ch = out_ch
        self.gobrrrrr = nn.Sequential(*layers)

        self.last_conv1d = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size,
            stride=stride, dilation=dilation, padding=padding) #TODO: padding?
        self.last_batch_norm = nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.1)
        self.last_relu = nn.ReLU()
        self.last_dropout = nn.Dropout(p)

    def forward(self, input):
        x = self.gobrrrrr(input)

        x = self.last_conv1d(x)
        x = self.last_batch_norm(x)
        if self.use_residual:
            x_residual = self.residual_process(input)
            x = x + x_residual
        x = self.last_relu(x)
        x = self.last_dropout(x)
        return x


class JasperModel(BaseModel):
    #TODO: 20ms windows with a 10ms overlap; We use 40 features for WSJ and 64 for LibriSpeech and F+S
    def __init__(self, n_feats, n_class):
        super().__init__(n_feats, n_class)
        ch, inc = 128, 128 
        self.Conv1 = Block(n_subblocks=1, p=0.2, in_ch=n_feats, out_ch=ch+inc, kernel_size=11, stride=2); ch = ch+inc
        self.B1    = Block(n_subblocks=3, p=0.2, in_ch=ch, out_ch=ch+inc, kernel_size=11); ch = ch+inc
        self.B2    = Block(n_subblocks=3, p=0.2, in_ch=ch, out_ch=ch+inc, kernel_size=13); ch = ch+inc
        self.B3    = Block(n_subblocks=3, p=0.2, in_ch=ch, out_ch=ch+inc, kernel_size=17); ch = ch+inc
        self.B4    = Block(n_subblocks=3, p=0.3, in_ch=ch, out_ch=ch+inc, kernel_size=21); ch = ch+inc
        self.B5    = Block(n_subblocks=3, p=0.3, in_ch=ch, out_ch=ch+inc, kernel_size=25); ch = ch+inc
        self.Conv2 = Block(n_subblocks=1, p=0.4, in_ch=ch, out_ch=ch+inc, kernel_size=29, dilation=2); ch = ch+inc
        self.Conv3 = Block(n_subblocks=1, p=0.4, in_ch=ch, out_ch=ch+inc, kernel_size=1);  ch = ch+inc
        self.Conv4 = Block(n_subblocks=1, p=0.0, in_ch=ch, out_ch=n_class, kernel_size=1)

        
        # self.B1_   = Block(n_subblocks=5, p=0.2, in_ch=256 , out_ch=256 , kernel_size=11)
        
        # self.B2_   = Block(n_subblocks=5, p=0.2, in_ch=384 , out_ch=384 , kernel_size=13)
        
        # self.B3_   = Block(n_subblocks=5, p=0.2, in_ch=512 , out_ch=512 , kernel_size=17)
        
        # self.B4_   = Block(n_subblocks=5, p=0.3, in_ch=640 , out_ch=640 , kernel_size=21)
        
        # self.B5_   = Block(n_subblocks=5, p=0.3, in_ch=768 , out_ch=768 , kernel_size=25)

        self.gobrrrrr = nn.Sequential(
              self.Conv1 
            , self.B1
            # , self.B1_
            , self.B2
            # , self.B2_
            , self.B3
            # , self.B3_
            , self.B4
            # , self.B4_
            , self.B5
            # , self.B5_
            , self.Conv2 
            , self.Conv3 
            , self.Conv4
        )

    def forward(self, spectrogram, *args, **kwargs):
        x = torch.transpose(spectrogram, 1, 2)
        x = self.gobrrrrr(x)
        out = torch.transpose(x, 1, 2)
        return out

    def transform_input_lengths(self, input_lengths):
        return input_lengths//4 #TODO: Do we reduce time dimension here?
