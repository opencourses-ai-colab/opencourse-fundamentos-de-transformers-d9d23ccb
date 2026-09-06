"""Generación autoregresiva mínima."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _aplicar_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove_sorted = cumulative > top_p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(logits, dtype=torch.bool)
    remove.scatter_(dim=-1, index=sorted_indices, src=remove_sorted)
    return logits.masked_fill(remove, -float("inf"))


@torch.no_grad()
def generar_ids(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
        if top_p is not None:
            if not 0 < top_p <= 1:
                raise ValueError("top_p debe estar en el intervalo (0, 1]")
            logits = _aplicar_top_p(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def generar_texto(
    model,
    prompt: str,
    token_a_id: dict[str, int],
    id_a_token: dict[int, str],
    max_new_tokens: int = 80,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
    ids = [token_a_id[c] for c in prompt if c in token_a_id]
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long)
    out = generar_ids(model, idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)[0].tolist()
    return "".join(id_a_token[int(i)] for i in out)
