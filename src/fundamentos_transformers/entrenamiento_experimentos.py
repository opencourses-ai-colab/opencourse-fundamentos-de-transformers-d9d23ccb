"""Experimentos auxiliares para entrenamiento autoregresivo."""

from __future__ import annotations

from dataclasses import dataclass
import math

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .datos import TextoAutoregresivo
from .entrenamiento import entrenar_pasos
from .modelo import MiniTransformerLM
from .tokenizacion import codificar, construir_vocabulario, tokenizar_caracteres


@dataclass
class ResultadoCorpusEntrenamiento:
    corpus: str
    tokens: list[str]
    token_a_id: dict[str, int]
    id_a_token: dict[int, str]
    ids: torch.Tensor
    block_size: int
    split: int
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    train_data: TextoAutoregresivo
    val_data: TextoAutoregresivo
    resumen: pd.DataFrame
    particiones: pd.DataFrame


@dataclass
class ResultadoBatchEntrenamiento:
    x_batch: torch.Tensor
    y_batch: torch.Tensor
    batch: pd.DataFrame
    alineacion: pd.DataFrame
    formas: pd.DataFrame


@dataclass
class ResultadoPasoManual:
    model: MiniTransformerLM
    auditoria: pd.DataFrame
    optimizacion: pd.DataFrame
    gradientes: pd.DataFrame


@dataclass
class ResultadoEntrenamientoBreve:
    model: MiniTransformerLM
    history: dict[str, list[float]]
    historial: pd.DataFrame
    lectura: pd.DataFrame
    figura: plt.Figure


@dataclass
class ResultadoComparacionEntrenamiento:
    comparacion: pd.DataFrame
    lectura: pd.DataFrame


def preparar_corpus_entrenamiento() -> ResultadoCorpusEntrenamiento:
    corpus = ("atencion mezcla contexto. transformers predicen tokens. " * 8).lower()
    tokens = tokenizar_caracteres(corpus)
    token_a_id, id_a_token = construir_vocabulario(tokens)
    ids = torch.tensor(codificar(tokens, token_a_id), dtype=torch.long)
    block_size = 16
    split = int(0.8 * len(ids))
    train_ids = ids[:split]
    val_ids = ids[split - block_size :]
    train_data = TextoAutoregresivo(train_ids, block_size=block_size)
    val_data = TextoAutoregresivo(val_ids, block_size=block_size)
    resumen = pd.DataFrame(
        [
            {"objeto": "caracteres del corpus", "valor": len(corpus), "lectura": "texto bruto usado solo para demostracion"},
            {"objeto": "tokens", "valor": len(tokens), "lectura": "tokenizacion por caracteres"},
            {"objeto": "Vocab", "valor": len(token_a_id), "lectura": "simbolos distintos observados"},
            {"objeto": "tokens train", "valor": len(train_ids), "lectura": "se usan para actualizar parametros"},
            {"objeto": "tokens valid", "valor": len(val_ids), "lectura": "se usan para estimar perdida sin actualizar"},
            {"objeto": "block_size", "valor": block_size, "lectura": "longitud de contexto T"},
        ]
    )
    particiones = pd.DataFrame(
        [
            {"particion": "train", "inicio": 0, "fin": int(split), "ventanas disponibles": int(len(train_ids) - block_size)},
            {"particion": "valid", "inicio": int(split - block_size), "fin": int(len(ids)), "ventanas disponibles": int(len(val_ids) - block_size)},
        ]
    )
    return ResultadoCorpusEntrenamiento(corpus, tokens, token_a_id, id_a_token, ids, block_size, split, train_ids, val_ids, train_data, val_data, resumen, particiones)


def _decodificar_ids(secuencia: torch.Tensor, id_a_token: dict[int, str]) -> str:
    return "".join(id_a_token[int(i)] for i in secuencia)


def preparar_batch_entrenamiento(train_data: TextoAutoregresivo, id_a_token: dict[int, str]) -> ResultadoBatchEntrenamiento:
    x_batch, y_batch = train_data.sample_batch(batch_size=4, seed=1)
    batch = pd.DataFrame(
        [
            {"ejemplo": b, "x decodificado": _decodificar_ids(x_batch[b], id_a_token), "y decodificado": _decodificar_ids(y_batch[b], id_a_token), "lectura": "y es x desplazado una posicion"}
            for b in range(x_batch.shape[0])
        ]
    )
    alineacion = pd.DataFrame(
        [
            {"posicion t": t, "x[0,t]": repr(id_a_token[int(x_batch[0, t])]), "y[0,t]": repr(id_a_token[int(y_batch[0, t])]), "lectura": "target siguiente respecto al contexto"}
            for t in range(x_batch.shape[1])
        ]
    )
    formas = pd.DataFrame(
        [
            {"objeto": "x_batch", "forma": tuple(x_batch.shape), "lectura": "entrada del modelo"},
            {"objeto": "y_batch", "forma": tuple(y_batch.shape), "lectura": "targets para cross-entropy"},
        ]
    )
    return ResultadoBatchEntrenamiento(x_batch, y_batch, batch, alineacion, formas)


def _grad_norm(modulo) -> float:
    total = 0.0
    for parametro in modulo.parameters():
        if parametro.grad is not None:
            total += float(parametro.grad.detach().pow(2).sum())
    return math.sqrt(total)


def preparar_paso_manual(config: dict, train_data: TextoAutoregresivo, seed: int = 7) -> ResultadoPasoManual:
    torch.manual_seed(seed)
    model = MiniTransformerLM(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x_step, y_step = train_data.sample_batch(batch_size=8, seed=3)
    lm_head_before = model.lm_head.weight.detach().clone()
    logits, loss = model(x_step, y_step)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm_lm = _grad_norm(model.lm_head)
    optimizer.step()
    cambio_lm = float((model.lm_head.weight.detach() - lm_head_before).abs().max())
    auditoria = pd.DataFrame(
        [
            {"objeto": "x_step", "forma": tuple(x_step.shape), "lectura": "batch de entrada"},
            {"objeto": "y_step", "forma": tuple(y_step.shape), "lectura": "targets desplazados"},
            {"objeto": "logits", "forma": tuple(logits.shape), "lectura": "puntajes B,T,V"},
            {"objeto": "loss", "forma": "escalar", "lectura": "cross-entropy media"},
        ]
    )
    optimizacion = pd.DataFrame(
        [
            {"paso": "forward", "que calcula": "logits y loss", "que cambia": "nada"},
            {"paso": "backward", "que calcula": "gradientes", "que cambia": "parametro.grad"},
            {"paso": "optimizer.step", "que calcula": "actualizacion AdamW", "que cambia": "parametros"},
            {"paso": "validacion sin gradiente", "que calcula": "perdida de referencia", "que cambia": "nada"},
        ]
    )
    gradientes = pd.DataFrame(
        [
            {"medida": "loss antes del step", "valor": round(float(loss.detach()), 4), "lectura": "senal que se retropropaga"},
            {"medida": "norma grad lm_head", "valor": round(grad_norm_lm, 6), "lectura": "magnitud de gradiente en la cabeza"},
            {"medida": "max cambio lm_head", "valor": round(cambio_lm, 8), "lectura": "evidencia de actualizacion"},
        ]
    )
    return ResultadoPasoManual(model, auditoria, optimizacion, gradientes)


def preparar_entrenamiento_breve(config: dict, train_data: TextoAutoregresivo, val_data: TextoAutoregresivo, seed: int = 7) -> ResultadoEntrenamientoBreve:
    torch.manual_seed(seed)
    model = MiniTransformerLM(**config)
    history = entrenar_pasos(model, train_data, val_data, steps=35, batch_size=16, lr=3e-3, eval_every=5)
    historial = pd.DataFrame(history).rename(columns={"train": "train_loss", "val": "val_loss"})
    historial["train_ppl"] = historial["train_loss"].map(lambda v: round(math.exp(v), 3))
    historial["val_ppl"] = historial["val_loss"].map(lambda v: round(math.exp(v), 3))
    historial["brecha val-train"] = (historial["val_loss"] - historial["train_loss"]).round(4)
    historial["train_loss"] = historial["train_loss"].round(4)
    historial["val_loss"] = historial["val_loss"].round(4)
    lectura = pd.DataFrame(
        [
            {"senal": "train_loss baja", "lectura": "el modelo se ajusta a batches de entrenamiento", "cautela": "puede memorizar patrones repetidos"},
            {"senal": "val_loss baja", "lectura": "parte del ajuste se transfiere a ventanas no actualizadas", "cautela": "validacion muy pequena no prueba generalizacion amplia"},
            {"senal": "brecha val-train", "lectura": "diferencia entre ajuste y validacion", "cautela": "si crece mucho puede indicar sobreajuste"},
            {"senal": "ppl = exp(loss)", "lectura": "incertidumbre promedio aproximada", "cautela": "no mide comprension ni calidad generativa"},
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    axes[0].plot(historial["step"], historial["train_loss"], marker="o", label="train")
    axes[0].plot(historial["step"], historial["val_loss"], marker="o", label="valid")
    axes[0].set(title="Cross-entropy media por token", xlabel="step", ylabel="loss")
    axes[0].legend(frameon=False)
    axes[1].plot(historial["step"], historial["train_ppl"], marker="o", label="train")
    axes[1].plot(historial["step"], historial["val_ppl"], marker="o", label="valid")
    axes[1].set(title="Perplexity aproximada", xlabel="step", ylabel="exp(loss)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return ResultadoEntrenamientoBreve(model, history, historial, lectura, fig)


def preparar_comparacion_entrenamiento(config: dict, ids: torch.Tensor, split: int, train_ids: torch.Tensor) -> ResultadoComparacionEntrenamiento:
    comparaciones = [
        {"nombre": "base", "block_size": 16, "steps": 20, "lr": 3e-3},
        {"nombre": "lr menor", "block_size": 16, "steps": 20, "lr": 1e-3},
        {"nombre": "menos pasos", "block_size": 16, "steps": 10, "lr": 3e-3},
    ]
    registros = []
    for cfg in comparaciones:
        train_tmp = TextoAutoregresivo(train_ids, block_size=cfg["block_size"])
        val_tmp = TextoAutoregresivo(ids[split - cfg["block_size"] :], block_size=cfg["block_size"])
        model_tmp = MiniTransformerLM(
            vocab_size=config["vocab_size"],
            block_size=cfg["block_size"],
            n_embd=config["n_embd"],
            n_head=config["n_head"],
            n_layer=config["n_layer"],
            dropout=0.0,
        )
        hist = entrenar_pasos(model_tmp, train_tmp, val_tmp, steps=cfg["steps"], batch_size=16, lr=cfg["lr"], eval_every=max(5, cfg["steps"] // 2))
        registros.append(
            {
                "prueba": cfg["nombre"],
                "block_size": cfg["block_size"],
                "steps": cfg["steps"],
                "lr": cfg["lr"],
                "train_loss final": round(hist["train"][-1], 4),
                "val_loss final": round(hist["val"][-1], 4),
                "val_ppl final": round(math.exp(hist["val"][-1]), 3),
            }
        )
    lectura = pd.DataFrame(
        [
            {"criterio": "una variable", "lectura": "comparar cambios controlados"},
            {"criterio": "loss final", "lectura": "mide ajuste promedio al token observado"},
            {"criterio": "ppl final", "lectura": "reexpresa la loss, no mide inteligencia"},
        ]
    )
    return ResultadoComparacionEntrenamiento(pd.DataFrame(registros), lectura)
