import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.models import resnet18

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import random
import os

from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelArgs:
    embeds_dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 4096


class RMSNorm(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeds_dim = args.embeds_dim
        self.eps = args.norm_eps
        self.gamma = nn.Parameter(torch.ones(args.embeds_dim), requires_grad=True)

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)


def get_counted_thetas(
    dim: int,
    theta: float = 10000.0
):
    thetas = 1.0 / (np.pow(theta, torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    repeated_thetas = torch.repeat_interleave(thetas, 2)
    return thetas


def get_rotary_matrices(
    seq_len: int,
    thetas: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos_matrix = np.cos(thetas * (torch.arange(seq_len) + 1)[:, None])
    sin_matrix = np.sin(thetas * (torch.arange(seq_len) + 1)[:, None])
    return cos_matrix, sin_matrix


def apply_rope(
    x: torch.Tensor,
    matrices: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    odd_indices = -x[1::2]
    even_indices = x[0::2]

    x_for_sin = torch.empty_like(x)
    x_for_sin[0::2] = odd_indices
    x_for_sin[1::2] = even_indices

    rope_x = x * matrices[0] + x_for_sin * matrices[1]
    return rope_x

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        self.head_embeds_dim = args.embeds_dim // args.n_heads
        self.wq = nn.Linear(args.embeds_dim, args.n_heads * self.head_embeds_dim, bias=False)
        self.wk = nn.Linear(args.embeds_dim, self.n_heads * self.head_embeds_dim, bias=False)
        self.wv = nn.Linear(args.embeds_dim, self.n_heads * self.head_embeds_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_embeds_dim, args.dim, bias=False)

    def process_qkv(
        self,
        x: torch.Tensor,
        batch_size: int,
        sequence_len: int
    ) -> torch.tensor:
        x = x.view(batch_size, sequence_len, self.n_heads, self.head_embeds_dim)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor, rope_matrices: Tuple[torch.Tensor, torch.Tensor]):
        batch_size, sequence_len, initial_embeds = x.shape

        #(B, seq_len, N_heads * Head_Dim) -> (B, N_heads, seq_len, Head_Embed_Dim)
        xq = self.process_qkv(self.wq(x), batch_size, sequence_len)
        xk = self.process_qkv(self.wk(x), batch_size, sequence_len)
        xv = self.process_qkv(self.wv(x), batch_size, sequence_len)

        #apply rope
        rope_matrices = (el[:sequence_len, :] for el in rope_matrices)
        xq = apply_rope(xq, rope_matrices)
        xk = apply_rope(xk, rope_matrices)

        attention = F.scaled_dot_product_attention(xq, xk, xv, is_casual=True)
        softmax_attention = F.softmax(attention.float(), dim=-1)
        return softmax_attention.transpose(1, 2).contiguous().view(batch_size, sequence_len, -1)


class FeedForwardSwiGLU(nn.Module):
    def __init__(
            self,
            args: ModelArgs
    ):
        super().__init__()
        hidden_dim = args.hidden_dim
        self.w1 = nn.Linear(args.embeds_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.embeds_dim, bias=False)
        self.w3 = nn.Linear(args.embeds_dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        return self.w2(swish * self.w3(x))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForwardSwiGLU(args)
        self.rms_norm_attention = RMSNorm(args)
        self.rms_norm_ffn = RMSNorm(args)

    def forward(self, x: torch.Tensor, rope_matrices: Tuple[torch.Tensor, torch.Tensor]):
        x = x + self.attention(self.rms_norm_attention, rope_matrices)
        x = x + self.feed_forward(self.rms_norm.ffn(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        self.args = args

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.embeddings = nn.Embedding(args.vocab_size, args.embeds_dim)
        self.rms_norm = RMSNorm(self.args.embeds_dim, eps=self.args.norm_eps)
        self.linear = nn.Linear(self.args.embeds_dim, self.args.vocab_size, bias=False)

        self.rope_matrices = get_rotary_matrices(
            args.max_seq_len, get_counted_thetas(args.embeds_dim // args.n_heads)
        )

    def forward(self, x: torch.Tensor, rope_matrices: Tuple[torch.Tensor, torch.Tensor]):
        batch_size, seq_len = x.shape
        h = self.embeddings(x)

        for layer in self.layers:
            h = layer(h, rope_matrices)
        h = self.rms_norm(h)
        output = F.softmax(self.linear(h).float(), dim=-1)
        return output

