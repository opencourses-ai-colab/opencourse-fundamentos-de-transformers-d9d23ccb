"""Operaciones mínimas de atención."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if mask is not None:
        while mask.ndim < scores.ndim:
            mask = mask.unsqueeze(0)
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


def split_heads(x: torch.Tensor, n_head: int) -> torch.Tensor:
    b, t, c = x.shape
    if c % n_head != 0:
        raise ValueError("La dimensión de canales debe ser divisible por n_head")
    return x.view(b, t, n_head, c // n_head).transpose(1, 2)


def combine_heads(x: torch.Tensor) -> torch.Tensor:
    b, h, t, d = x.shape
    return x.transpose(1, 2).contiguous().view(b, t, h * d)
