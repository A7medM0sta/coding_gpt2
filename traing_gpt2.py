from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect





class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init()
        self.weight = nn.Paramter(torch.ones(ndim))
        self.bias = nn.Paramter()

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layers: int = 6
    n_head: int = 6
    n_embed: int = 384

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config


        self.transformer = nn.ModuleDict(dict(
            wte
        ))
