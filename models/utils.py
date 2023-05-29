import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional, Tuple, Union
import numpy as np

# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, positions) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        return super().forward(positions)

def glorot_uniform_init(weight, fan_in, fan_out):
    v = 6 if (fan_in != 0 and fan_out != 0) else 3
    bound = float(math.sqrt(v / (fan_in + fan_out)))
    nn.init.uniform_(weight, a=-bound, b=bound)


def generate_absolute_positional_embeddings(max_len, d_model, freeze=True):
    with T.no_grad():
        # Compute the positional encodings once in log space.
        pe = T.zeros(max_len, d_model)
        position = T.arange(1, max_len+1).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        assert pe.size() == (max_len, d_model)
        pe = pe / math.sqrt(d_model)
    return pe.unsqueeze(0), nn.Embedding.from_pretrained(pe,
                                                         freeze=freeze)


def generate_relative_positional_embeddings(max_len, d_model):
    with T.no_grad():
        # Compute the positional encodings once in log space.
        pe = T.zeros(2 * max_len + 1, d_model)
        position = T.arange(-max_len, max_len + 1).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        assert pe.size() == (2 * max_len + 1, d_model)
        pe = nn.Embedding.from_pretrained(pe,
                                          freeze=True)
    return pe


def generate_temporal_encodings(time, d_model):
    with T.no_grad():
        pe = T.zeros(d_model).float()
        div_term = T.exp(T.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[0::2] = T.sin(time * div_term)
        pe[1::2] = T.cos(time * div_term)

        pe = pe.view(1, 1, d_model)

    return pe
