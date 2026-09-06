"""Datasets autoregresivos pequeños."""

from __future__ import annotations

import torch


class TextoAutoregresivo:
    def __init__(self, ids: torch.Tensor, block_size: int) -> None:
        if ids.ndim != 1:
            raise ValueError("ids debe ser un tensor 1D")
        if len(ids) <= block_size:
            raise ValueError("ids debe tener más elementos que block_size")
        self.ids = ids.long()
        self.block_size = block_size

    def sample_batch(self, batch_size: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.ids.device)
            generator.manual_seed(seed)
        max_start = len(self.ids) - self.block_size - 1
        ix = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
        x = torch.stack([self.ids[i : i + self.block_size] for i in ix])
        y = torch.stack([self.ids[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y


def crear_batch_autoregresivo(ids: torch.Tensor, block_size: int, batch_size: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    return TextoAutoregresivo(ids, block_size).sample_batch(batch_size, seed=seed)
