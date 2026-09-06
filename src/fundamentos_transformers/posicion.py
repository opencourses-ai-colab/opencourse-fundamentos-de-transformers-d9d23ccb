"""Experimentos auxiliares para el notebook de posición y orden."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch


@dataclass
class ResultadoPosicionOrden:
    """Tablas y figura del experimento contenido más posición."""

    vocab: dict[str, int]
    id_a_token: dict[int, str]
    block_size: int
    dimension: int
    formas: pd.DataFrame
    comparacion: pd.DataFrame
    repeticion: pd.DataFrame
    figura: plt.Figure


@dataclass
class ResultadoLimitePosicional:
    """Tablas para estudiar el límite de una tabla posicional aprendida."""

    limites: pd.DataFrame
    consulta: pd.DataFrame


@torch.no_grad()
def preparar_experimento_posicion_orden(
    *,
    seed: int = 7,
    dimension: int = 4,
    seq_1_tokens: tuple[str, ...] = ("a", "b", "c", "a"),
    seq_2_tokens: tuple[str, ...] = ("c", "a", "a", "b"),
) -> ResultadoPosicionOrden:
    """Construye tablas y figura para comparar contenido con contenido ubicado."""

    torch.manual_seed(seed)
    vocab_tokens = sorted(set(seq_1_tokens) | set(seq_2_tokens))
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
    id_a_token = {idx: token for token, idx in vocab.items()}
    block_size = len(seq_1_tokens)
    vocab_size = len(vocab)

    token_embedding = torch.nn.Embedding(vocab_size, dimension)
    position_embedding = torch.nn.Embedding(block_size, dimension)

    seq_1 = torch.tensor([[vocab[token] for token in seq_1_tokens]])
    seq_2 = torch.tensor([[vocab[token] for token in seq_2_tokens]])
    pos = torch.arange(seq_1.shape[1])

    tok_1 = token_embedding(seq_1)
    tok_2 = token_embedding(seq_2)
    pos_vecs = position_embedding(pos)[None, :, :]
    h_1 = tok_1 + pos_vecs
    h_2 = tok_2 + pos_vecs

    formas = pd.DataFrame(
        [
            ["seq_1", tuple(seq_1.shape), "índices de token"],
            ["E_tok[seq_1]", tuple(tok_1.shape), "contenido por posición"],
            ["E_pos[0:T][None,:,:]", tuple(pos_vecs.shape), "posición compartida por batch"],
            ["h_1 = token + posición", tuple(h_1.shape), "representación inicial"],
        ],
        columns=["objeto", "forma", "interpretación"],
    )

    comparacion = pd.DataFrame(
        [
            {
                "posición t": t,
                "seq_1 token": id_a_token[int(seq_1[0, t])],
                "seq_2 token": id_a_token[int(seq_2[0, t])],
                "mismo token en t": "sí" if bool(seq_1[0, t] == seq_2[0, t]) else "no",
                "misma fila posicional": "sí",
                "distancia sin posición": round(float((tok_1[0, t] - tok_2[0, t]).norm()), 4),
                "distancia con posición": round(float((h_1[0, t] - h_2[0, t]).norm()), 4),
                "cambia h_t^(0)": "sí" if not torch.allclose(h_1[0, t], h_2[0, t]) else "no",
            }
            for t in range(seq_1.shape[1])
        ]
    )

    token_repetido = seq_1_tokens[0]
    posiciones_repetidas = [i for i, token in enumerate(seq_1_tokens) if token == token_repetido]
    pos_1, pos_2 = posiciones_repetidas[0], posiciones_repetidas[-1]
    repeticion = pd.DataFrame(
        [
            {
                "caso": "mismo token en posiciones distintas",
                "token": token_repetido,
                "posición 1": pos_1,
                "posición 2": pos_2,
                "misma fila E_tok": "sí",
                "misma fila E_pos": "no",
                "lectura": "mismo contenido, distinta ubicación",
            }
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].imshow(tok_1[0].detach().numpy(), aspect="auto", cmap="viridis")
    axes[0].set(title="Solo contenido: E_tok[seq_1]", xlabel="canal C", ylabel="posición t")
    im = axes[1].imshow(h_1[0].detach().numpy(), aspect="auto", cmap="viridis")
    axes[1].set(title="Contenido + posición: h_1^(0)", xlabel="canal C", ylabel="posición t")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)

    return ResultadoPosicionOrden(
        vocab=vocab,
        id_a_token=id_a_token,
        block_size=block_size,
        dimension=dimension,
        formas=formas,
        comparacion=comparacion,
        repeticion=repeticion,
        figura=fig,
    )


def preparar_limite_tabla_posicional(block_size: int) -> ResultadoLimitePosicional:
    """Construye tablas para mostrar el límite de una tabla posicional aprendida."""

    limites = pd.DataFrame(
        [
            {
                "posición solicitada": posicion,
                "block_size": block_size,
                "existe en E_pos": posicion < block_size,
                "lectura": "fila disponible" if posicion < block_size else "fuera de la tabla posicional",
            }
            for posicion in range(block_size + 3)
        ]
    )

    consulta = pd.DataFrame(
        [
            {
                "consulta": f"E_pos[{block_size}]",
                "resultado conceptual": "no existe esa fila",
                "consecuencia": "la arquitectura no representa directamente esa posición",
            }
        ]
    )

    return ResultadoLimitePosicional(limites=limites, consulta=consulta)
