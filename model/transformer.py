import torch
from torch import nn
from torch.nn import Transformer



class TransFormer(nn.Module):
    def __init__(self, nhead=2, num_encoder_layers=1) -> None:
        super().__init__()
        self.transformer = Transformer(nhead=nhead, num_encoder_layers=num_encoder_layers)


    def forward(self, src, tgt):
        return self.transformer(src, tgt)