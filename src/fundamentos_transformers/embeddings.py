"""Experimentos auxiliares para el notebook de embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .tokenizacion import construir_vocabulario


@dataclass
class ResultadoLookupEmbeddings:
    """Tablas y figura del experimento de lookup de embeddings."""

    token_a_id: dict[str, int]
    id_a_token: dict[int, str]
    formas: pd.DataFrame
    seleccion: pd.DataFrame
    figura: plt.Figure


@dataclass
class ResultadoGradientesEmbeddings:
    """Tablas del experimento de actualización potencial de embeddings."""

    resumen: pd.DataFrame
    gradientes: pd.DataFrame
    parametros: pd.DataFrame


def _token_visible(token: str) -> str:
    return "<espacio>" if token == " " else token


def preparar_experimento_lookup_embeddings(
    *,
    vocab: list[str] | None = None,
    dimension: int = 2,
    indice_consulta: int = 3,
    seed: int = 7,
) -> ResultadoLookupEmbeddings:
    """Construye tablas y figura para estudiar lookup de embeddings."""

    if vocab is None:
        vocab = list("atencion")

    torch.manual_seed(seed)
    token_a_id, id_a_token = construir_vocabulario(vocab)
    emb = torch.nn.Embedding(len(token_a_id), dimension)
    ids = torch.arange(len(token_a_id))
    vectores = emb(ids).detach()

    idx = torch.tensor([indice_consulta])
    one_hot = torch.nn.functional.one_hot(idx, num_classes=len(token_a_id)).float()
    seleccion_embedding = emb(idx).detach()
    seleccion_manual = emb.weight.detach()[idx]
    seleccion_one_hot = one_hot @ emb.weight.detach()

    formas = pd.DataFrame(
        [
            ["ids", tuple(ids.shape), "índices del vocabulario", "enteros"],
            ["E.weight", tuple(emb.weight.shape), "matriz entrenable de embeddings", "parámetros"],
            ["Embedding(ids)", tuple(vectores.shape), "filas seleccionadas", "vectores reales"],
            ["one_hot @ E", tuple(seleccion_one_hot.shape), "selección explícita de una fila", "vector real"],
        ],
        columns=["objeto", "forma", "interpretación", "tipo"],
    )

    seleccion = pd.DataFrame(
        [
            [
                int(idx.item()),
                _token_visible(id_a_token[int(idx.item())]),
                seleccion_embedding.numpy().round(4).tolist(),
                torch.allclose(seleccion_embedding, seleccion_manual),
                torch.allclose(seleccion_embedding, seleccion_one_hot),
            ]
        ],
        columns=[
            "índice consultado",
            "token",
            "vector seleccionado",
            "coincide con E[idx]",
            "coincide con one_hot @ E",
        ],
    )

    puntos = vectores.numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(puntos[:, 0], puntos[:, 1], s=90, color="#2a9d8f")
    for idx_plot, (x0, y0) in enumerate(puntos):
        ax.text(x0 + 0.03, y0 + 0.03, _token_visible(id_a_token[int(idx_plot)]), fontsize=12)
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set(title="Filas iniciales de E cuando C=2", xlabel="canal 1", ylabel="canal 2")
    fig.tight_layout()

    return ResultadoLookupEmbeddings(
        token_a_id=token_a_id,
        id_a_token=id_a_token,
        formas=formas,
        seleccion=seleccion,
        figura=fig,
    )


def preparar_experimento_gradientes_embeddings(
    token_a_id: dict[str, int],
    id_a_token: dict[int, str],
    *,
    dimension: int = 3,
    batch_tokens: tuple[str, ...] = ("a", "t", "a", "n"),
    dimensiones_parametros: tuple[int, ...] = (2, 4, 16, 64),
    seed: int = 11,
) -> ResultadoGradientesEmbeddings:
    """Construye tablas para observar qué filas de embeddings reciben gradiente."""

    torch.manual_seed(seed)
    emb_grad = torch.nn.Embedding(len(token_a_id), dimension)
    ids_batch = torch.tensor([[token_a_id[token] for token in batch_tokens]])
    vectores_batch = emb_grad(ids_batch)
    objetivo = torch.zeros_like(vectores_batch)
    loss = ((vectores_batch - objetivo) ** 2).mean()
    loss.backward()

    ids_batch_planos = ids_batch.flatten().tolist()
    consultados = set(ids_batch_planos)

    gradientes = pd.DataFrame(
        [
            {
                "fila": idx_fila,
                "token": _token_visible(id_a_token[idx_fila]),
                "veces_en_batch": ids_batch_planos.count(idx_fila),
                "aparece_en_batch": "sí" if idx_fila in consultados else "no",
                "norma_gradiente": round(float(grad.norm()), 6),
            }
            for idx_fila, grad in enumerate(emb_grad.weight.grad)
        ]
    )

    resumen = pd.DataFrame(
        {
            "ids_batch": [ids_batch.tolist()],
            "tokens_batch": [[_token_visible(token) for token in batch_tokens]],
            "loss": [round(float(loss.detach()), 6)],
        }
    )

    parametros = pd.DataFrame(
        [
            {
                "C": c,
                "vocabulario": len(token_a_id),
                "parámetros en E": len(token_a_id) * c,
                "forma de E": (len(token_a_id), c),
            }
            for c in dimensiones_parametros
        ]
    )

    return ResultadoGradientesEmbeddings(
        resumen=resumen,
        gradientes=gradientes,
        parametros=parametros,
    )
