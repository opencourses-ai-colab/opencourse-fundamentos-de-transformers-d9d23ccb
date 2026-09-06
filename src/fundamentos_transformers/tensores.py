"""Experimentos auxiliares para auditoria de formas tensoriales."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch


@dataclass
class ResultadoAuditoriaFormas:
    auditoria: pd.DataFrame
    tensores: pd.DataFrame
    figura: plt.Figure


@dataclass
class ResultadoDivisibilidad:
    divisibilidad: pd.DataFrame
    diagnostico: pd.DataFrame


def preparar_auditoria_formas_transformer(
    batch_size: int = 2,
    seq_len: int = 5,
    n_embd: int = 12,
    n_head: int = 3,
    seed: int = 7,
) -> ResultadoAuditoriaFormas:
    """Construye una auditoria compacta del flujo X -> Q/K/V -> heads."""

    if n_embd % n_head != 0:
        raise ValueError("n_embd debe ser divisible por n_head para separar cabezas uniformes.")

    torch.manual_seed(seed)
    head_dim = n_embd // n_head
    x = torch.randn(batch_size, seq_len, n_embd)
    qkv_proj = torch.nn.Linear(n_embd, 3 * n_embd, bias=False)
    qkv = qkv_proj(x)
    q, k, v = qkv.chunk(3, dim=-1)
    q_view = q.view(batch_size, seq_len, n_head, head_dim)
    k_view = k.view(batch_size, seq_len, n_head, head_dim)
    v_view = v.view(batch_size, seq_len, n_head, head_dim)
    q_heads = q_view.transpose(1, 2)

    auditoria = pd.DataFrame(
        [
            ["entrada", "-", tuple(x.shape), "se crea X", "define B, T y C", "-"],
            ["proyeccion QKV", tuple(x.shape), tuple(qkv.shape), "ultimo eje C -> 3C", "B y T", "Linear(C,3C)"],
            ["separar Q/K/V", tuple(qkv.shape), "3 tensores " + str(tuple(q.shape)), "3C -> C + C + C", "B y T", "chunk(3, dim=-1)"],
            ["separar heads", tuple(q.shape), tuple(q_view.shape), "C -> H,D", "B y T", "view(B,T,H,D)"],
            ["heads antes de tiempo", tuple(q_view.shape), tuple(q_heads.shape), "orden T,H -> H,T", "B y valores del tensor", "transpose(1,2)"],
        ],
        columns=["operacion", "forma antes", "forma despues", "que eje cambia", "que se conserva", "operacion PyTorch"],
    )

    tensores = pd.DataFrame(
        [
            ["X", tuple(x.shape), "$B,T,C$", "entrada con batch, posiciones y canales"],
            ["QKV", tuple(qkv.shape), "$B,T,3C$", "proyeccion conjunta por posicion"],
            ["Q", tuple(q.shape), "$B,T,C$", "consultas"],
            ["K", tuple(k.shape), "$B,T,C$", "llaves"],
            ["V", tuple(v.shape), "$B,T,C$", "valores"],
            ["Q/K/V separados por heads", tuple(q_view.shape), "$B,T,H,D$", "canales repartidos en cabezas"],
            ["Q/K/V por heads", tuple(q_heads.shape), "$B,H,T,D$", "eje de heads listo para atencion"],
        ],
        columns=["objeto", "forma", "lectura matematica", "interpretacion"],
    )

    fig, ax = plt.subplots(figsize=(8, 3.4))
    nombres = ["X", "QKV", "Q", "Q separado", "Q por heads"]
    formas = [x.shape, qkv.shape, q.shape, q_view.shape, q_heads.shape]
    num_ejes = [len(forma) for forma in formas]
    ax.bar(nombres, num_ejes, color=["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#457b9d"])
    for i, forma in enumerate(formas):
        ax.text(i, num_ejes[i] + 0.08, str(tuple(forma)), ha="center", fontsize=10)
    ax.set(ylim=(0, 5), ylabel="numero de ejes", title="Cambios de forma en el Transformer")
    fig.tight_layout()

    # Referenciar variables para dejar claro que el flujo Q/K/V se construye simetricamente.
    _ = (k_view, v_view)
    return ResultadoAuditoriaFormas(auditoria=auditoria, tensores=tensores, figura=fig)


def preparar_verificacion_divisibilidad(
    casos: tuple[tuple[int, int], ...] = ((12, 3), (12, 4), (10, 3), (16, 8)),
) -> ResultadoDivisibilidad:
    """Prepara una tabla sobre la condicion C = H * D."""

    divisibilidad = pd.DataFrame(
        [
            {
                "C": c_val,
                "H": h_val,
                "C divisible por H": "si" if c_val % h_val == 0 else "no",
                "D = C/H si aplica": c_val // h_val if c_val % h_val == 0 else "no definido",
                "lectura": "particion uniforme posible" if c_val % h_val == 0 else "no se puede dividir en cabezas iguales",
            }
            for c_val, h_val in casos
        ]
    )

    try:
        c_mal, h_mal = 10, 3
        d_mal = c_mal // h_mal
        torch.randn(1, 2, c_mal).view(1, 2, h_mal, d_mal)
        resultado_error = "sin error"
    except RuntimeError:
        resultado_error = "RuntimeError capturado: no hay D entero que conserve todos los valores"

    diagnostico = pd.DataFrame(
        [
            {
                "caso": "C=10, H=3",
                "intento": "view(1,2,3,3)",
                "problema": "3 x 3 = 9 canales, pero C=10",
                "resultado": resultado_error,
                "correccion": "elegir otro C o cambiar H",
            }
        ]
    )

    return ResultadoDivisibilidad(divisibilidad=divisibilidad, diagnostico=diagnostico)
