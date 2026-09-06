"""Transformer decoder mínimo para el curso."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .atencion import causal_mask, combine_heads, scaled_dot_product_attention, split_heads


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd debe ser divisible por n_head")
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", causal_mask(block_size))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        _, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = split_heads(q, self.n_head), split_heads(k, self.n_head), split_heads(v, self.n_head)
        out, weights = scaled_dot_product_attention(q, k, v, mask=self.mask[:t, :t])
        weights = self.attn_drop(weights)
        out = combine_heads(weights @ v)
        out = self.resid_drop(self.proj(out))
        if return_attention:
            return out, weights
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if return_attention:
            attn_out, weights = self.attn(self.ln1(x), return_attention=True)
            x = x + attn_out
            x = x + self.ffwd(self.ln2(x))
            return x, weights
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int = 64,
        n_head: int = 4,
        n_layer: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_attention: bool = False):
        b, t = idx.shape
        if t > self.block_size:
            raise ValueError("la longitud de contexto excede block_size")
        pos = torch.arange(t, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        x = self.drop(x)
        attention = None
        for i, block in enumerate(self.blocks):
            if return_attention and i == 0:
                x, attention = block(x, return_attention=True)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(b * t, self.vocab_size), targets.view(b * t))
        if return_attention:
            return logits, loss, attention
        return logits, loss


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
