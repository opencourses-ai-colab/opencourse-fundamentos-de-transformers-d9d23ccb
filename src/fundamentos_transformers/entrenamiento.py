"""Rutinas pequeñas de entrenamiento."""

from __future__ import annotations

import torch


@torch.no_grad()
def estimar_perdida(model, dataset, batch_size: int = 16, eval_iters: int = 4) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for i in range(eval_iters):
        x, y = dataset.sample_batch(batch_size, seed=10 + i)
        _, loss = model(x, y)
        losses.append(float(loss))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def entrenar_pasos(model, train_data, val_data, steps: int = 50, batch_size: int = 16, lr: float = 3e-3, eval_every: int = 10) -> dict[str, list[float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"step": [], "train": [], "val": []}
    model.train()
    for step in range(steps + 1):
        if step % eval_every == 0:
            history["step"].append(step)
            history["train"].append(estimar_perdida(model, train_data, batch_size=batch_size))
            history["val"].append(estimar_perdida(model, val_data, batch_size=batch_size))
        if step == steps:
            break
        x, y = train_data.sample_batch(batch_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return history
