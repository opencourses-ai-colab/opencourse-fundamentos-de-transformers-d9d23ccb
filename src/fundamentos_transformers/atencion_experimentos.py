"""Experimentos auxiliares para notebooks de atencion.

Las funciones de este modulo no muestran salidas. Devuelven tablas, tensores
clave y figuras para que cada notebook controle explicitamente que se despliega.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .atencion import causal_mask, combine_heads, scaled_dot_product_attention, split_heads


@dataclass
class ResultadoMezclaPonderada:
    formas: pd.DataFrame
    pesos: pd.DataFrame
    resumen_casos: pd.DataFrame
    aportes: pd.DataFrame
    figura: plt.Figure


@dataclass
class ResultadoQKV:
    auditoria: pd.DataFrame
    fila_atencion: pd.DataFrame
    verificacion: pd.DataFrame
    roles: pd.DataFrame
    figura: plt.Figure


@dataclass
class ResultadoSelfAttention:
    etapas: pd.DataFrame
    fila: pd.DataFrame
    verificaciones: pd.DataFrame
    lectura_fila: pd.DataFrame
    figura: plt.Figure


@dataclass
class ResultadoAtencionCausal:
    resumen: pd.DataFrame
    comparacion: pd.DataFrame
    fila: pd.DataFrame
    verificaciones: pd.DataFrame
    figura: plt.Figure


@dataclass
class ResultadoMultiHead:
    auditoria: pd.DataFrame
    traza_fila: pd.DataFrame
    configuracion: pd.DataFrame
    verificaciones: pd.DataFrame
    figura: plt.Figure
    n_head: int


@torch.no_grad()
def preparar_mezcla_ponderada() -> ResultadoMezclaPonderada:
    """Prepara el experimento de softmax y mezcla ponderada de valores."""

    values = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.3],
            [0.2, 1.1],
            [1.3, 1.0],
        ],
        dtype=torch.float32,
    )
    logits = torch.tensor([0.2, 1.2, 0.4, -0.1], dtype=torch.float32)
    logits_concentrados = torch.tensor([-4.0, 4.0, -4.0, -4.0], dtype=torch.float32)
    pesos_por_caso = {
        "softmax": torch.softmax(logits, dim=0),
        "uniforme": torch.full_like(logits, 1.0 / logits.numel()),
        "concentrada": torch.softmax(logits_concentrados, dim=0),
    }
    weights = pesos_por_caso["softmax"]
    out = weights @ values

    formas = pd.DataFrame(
        [
            ["values / V", tuple(values.shape), "matriz de valores con n filas y C canales"],
            ["logits / s", tuple(logits.shape), "puntajes sin normalizar; todavia no son pesos"],
            ["weights / alpha", tuple(weights.shape), "pesos normalizados por softmax"],
            ["out / y", tuple(out.shape), "salida mezclada con C canales"],
        ],
        columns=["objeto", "forma", "interpretacion"],
    )

    pesos = pd.DataFrame(
        [
            {
                "caso": nombre,
                "v0": float(p[0]),
                "v1": float(p[1]),
                "v2": float(p[2]),
                "v3": float(p[3]),
                "suma pesos": float(p.sum()),
            }
            for nombre, p in pesos_por_caso.items()
        ]
    ).round(3)

    resumen_casos = []
    for nombre, p in pesos_por_caso.items():
        salida = p @ values
        distancias = torch.linalg.norm(values - salida, dim=1)
        resumen_casos.append(
            {
                "caso": nombre,
                "salida y": tuple(round(float(x), 3) for x in salida),
                "peso maximo": round(float(p.max()), 3),
                "valor con mayor peso": f"v{int(p.argmax())}",
                "valor mas cercano a y": f"v{int(distancias.argmin())}",
                "dist v0": round(float(distancias[0]), 3),
                "dist v1": round(float(distancias[1]), 3),
                "dist v2": round(float(distancias[2]), 3),
                "dist v3": round(float(distancias[3]), 3),
            }
        )
    resumen_casos_df = pd.DataFrame(resumen_casos)

    aportes = pd.DataFrame(
        [
            {
                "valor": f"v{i}",
                "logit": float(logits[i]),
                "peso softmax": float(weights[i]),
                "canal 1": float(values[i, 0]),
                "canal 2": float(values[i, 1]),
                "aporte c1": float(weights[i] * values[i, 0]),
                "aporte c2": float(weights[i] * values[i, 1]),
            }
            for i in range(values.shape[0])
        ]
    ).round(3)
    aportes.loc[len(aportes)] = {
        "valor": "salida",
        "logit": np.nan,
        "peso softmax": round(float(weights.sum()), 3),
        "canal 1": np.nan,
        "canal 2": np.nan,
        "aporte c1": round(float(out[0]), 3),
        "aporte c2": round(float(out[1]), 3),
    }

    fig = _figura_mezcla_ponderada(values, weights, out)
    return ResultadoMezclaPonderada(formas, pesos, resumen_casos_df, aportes, fig)


@torch.no_grad()
def preparar_experimento_qkv(seed: int = 7) -> ResultadoQKV:
    """Prepara tablas y figura para el experimento Q/K/V."""

    torch.manual_seed(seed)
    x = torch.tensor(
        [
            [
                [1.0, 0.0, 0.3, 0.2],
                [0.2, 1.0, 0.0, 0.4],
                [0.3, 0.1, 1.0, 0.5],
            ]
        ],
        dtype=torch.float32,
    )
    wq = torch.nn.Linear(4, 4, bias=False)
    wk = torch.nn.Linear(4, 4, bias=False)
    wv = torch.nn.Linear(4, 4, bias=False)
    q, k, v = wq(x), wk(x), wv(x)
    d_model = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_model)
    weights = torch.softmax(scores, dim=-1)
    out = weights @ v
    out_func, weights_func = scaled_dot_product_attention(q, k, v)

    t_ref = 1
    alpha_t = weights[0, t_ref]
    salida_manual = alpha_t @ v[0]
    salida_atencion = out[0, t_ref]
    max_diff = torch.max(torch.abs(salida_manual - salida_atencion))

    auditoria = pd.DataFrame(
        [
            ["X", tuple(x.shape), "B,T,C", "entrada con batch, posiciones y canales"],
            ["Q = XW_Q", tuple(q.shape), "B,T,D", "consultas; participan en scores"],
            ["K = XW_K", tuple(k.shape), "B,T,D", "llaves; participan en scores"],
            ["V = XW_V", tuple(v.shape), "B,T,D", "valores; no participan en scores"],
            ["S = QK^T/sqrt(D)", tuple(scores.shape), "B,T,T", "compatibilidad consulta-llave"],
            ["A = softmax(S)", tuple(weights.shape), "B,T,T", "pesos por fila; cada fila suma 1"],
            ["Y = AV", tuple(out.shape), "B,T,D", "salida contextual mezclando valores"],
        ],
        columns=["objeto", "forma", "lectura", "rol computacional"],
    )

    fila_atencion = pd.DataFrame(
        [
            {
                "j": j,
                "score s_1j": round(float(scores[0, t_ref, j]), 3),
                "peso alpha_1j": round(float(alpha_t[j]), 3),
                "V participa en score": "no",
                "valor usado luego": f"v_{j}",
                "aporte c1": round(float(alpha_t[j] * v[0, j, 0]), 3),
            }
            for j in range(x.shape[1])
        ]
    )

    verificacion = pd.DataFrame(
        [
            ["suma de pesos fila t=1", round(float(alpha_t.sum()), 6), "softmax normaliza la fila"],
            ["alpha_t @ V", tuple(round(float(z), 4) for z in salida_manual), "calculo manual de la salida"],
            ["out[0,t]", tuple(round(float(z), 4) for z in salida_atencion), "salida producida por atencion"],
            ["maxima diferencia", float(max_diff), "debe ser 0 o error numerico minimo"],
            ["coincide con helper", bool(torch.allclose(out, out_func) and torch.allclose(weights, weights_func)), "misma operacion que scaled_dot_product_attention"],
        ],
        columns=["verificacion", "valor", "lectura"],
    )

    roles = pd.DataFrame(
        [
            ["Q y K", "calculan scores", "si", "no"],
            ["softmax", "convierte scores en pesos", "usa S", "no"],
            ["V", "aporta contenido a la mezcla", "no", "si"],
        ],
        columns=["objeto", "papel", "antes de softmax", "despues de softmax"],
    )

    return ResultadoQKV(auditoria, fila_atencion, verificacion, roles, _figura_pesos(weights, "Pesos derivados de QK^T", "llave j", "consulta t"))


@torch.no_grad()
def preparar_self_attention_matricial(seed: int = 2) -> ResultadoSelfAttention:
    """Prepara el experimento matricial completo Q,K,V,S,A,Y."""

    torch.manual_seed(seed)
    x = torch.randn(1, 5, 6)
    q_proj = torch.nn.Linear(6, 6, bias=False)
    k_proj = torch.nn.Linear(6, 6, bias=False)
    v_proj = torch.nn.Linear(6, 6, bias=False)
    q, k, v = q_proj(x), k_proj(x), v_proj(x)
    d_model = q.size(-1)
    puntajes_sin_escala = q @ k.transpose(-2, -1)
    puntajes = puntajes_sin_escala / math.sqrt(d_model)
    pesos = torch.softmax(puntajes, dim=-1)
    salida = pesos @ v
    salida_helper, pesos_helper = scaled_dot_product_attention(q, k, v)

    t_ref = 2
    fila_scores = puntajes[0, t_ref]
    fila_pesos = pesos[0, t_ref]
    salida_manual = fila_pesos @ v[0]
    salida_tensor = salida[0, t_ref]

    etapas = pd.DataFrame(
        [
            ["proyectar Q,K,V", tuple(x.shape), tuple(q.shape), "C -> D", "crea roles separados"],
            ["QK^T", f"{tuple(q.shape)} x {(q.shape[0], q.shape[2], q.shape[1])}", tuple(puntajes_sin_escala.shape), "D desaparece", "scores crudos entre t y j"],
            ["escala / sqrt(D)", tuple(puntajes_sin_escala.shape), tuple(puntajes.shape), "no cambia forma", "controla magnitud de scores"],
            ["softmax por fila", tuple(puntajes.shape), tuple(pesos.shape), "normaliza eje j", "convierte scores en pesos"],
            ["AV", f"{tuple(pesos.shape)} x {tuple(v.shape)}", tuple(salida.shape), "T consultado -> D", "mezcla valores"],
        ],
        columns=["etapa", "forma antes", "forma despues", "eje que cambia", "significado"],
    )

    fila = pd.DataFrame(
        [
            {
                "j": j,
                "S[2,j] score": round(float(fila_scores[j]), 3),
                "A[2,j] peso": round(float(fila_pesos[j]), 3),
                "V[j] canal 1": round(float(v[0, j, 0]), 3),
                "aporte c1": round(float(fila_pesos[j] * v[0, j, 0]), 3),
            }
            for j in range(x.shape[1])
        ]
    )
    fila.loc[len(fila)] = {
        "j": "suma/salida",
        "S[2,j] score": np.nan,
        "A[2,j] peso": round(float(fila_pesos.sum()), 6),
        "V[j] canal 1": np.nan,
        "aporte c1": round(float(salida_tensor[0]), 6),
    }

    verificaciones = pd.DataFrame(
        [
            ["todas las filas de A suman 1", bool(torch.allclose(pesos[0].sum(dim=-1), torch.ones(x.shape[1]), atol=1e-6))],
            ["Y[2] coincide con A[2,:] @ V", bool(torch.allclose(salida_manual, salida_tensor, atol=1e-6))],
            ["helper coincide con calculo explicito", bool(torch.allclose(salida, salida_helper) and torch.allclose(pesos, pesos_helper))],
            ["forma S", tuple(puntajes.shape)],
            ["forma A", tuple(pesos.shape)],
            ["forma Y", tuple(salida.shape)],
        ],
        columns=["verificacion", "resultado"],
    )

    lectura_fila = pd.DataFrame(
        [
            ["S[2,:]", tuple(round(float(z), 3) for z in fila_scores), "puntajes sin normalizar"],
            ["A[2,:]", tuple(round(float(z), 3) for z in fila_pesos), "distribucion sobre posiciones j"],
            ["A[2,:] @ V", tuple(round(float(z), 4) for z in salida_manual), "reconstruccion manual de Y[2]"],
            ["Y[2]", tuple(round(float(z), 4) for z in salida_tensor), "fila calculada por la operacion matricial"],
        ],
        columns=["objeto", "valor", "lectura"],
    )

    return ResultadoSelfAttention(etapas, fila, verificaciones, lectura_fila, _figura_pesos(pesos, "Matriz de autoatencion A", "posicion consultada j", "consulta t", colorbar_label="peso A[t,j]"))


@torch.no_grad()
def preparar_atencion_causal(seed: int = 4) -> ResultadoAtencionCausal:
    """Prepara comparacion entre atencion causal y no causal."""

    torch.manual_seed(seed)
    x = torch.randn(1, 6, 8)
    q = k = v = x
    seq_len = x.shape[1]
    mask = causal_mask(seq_len)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    scores_causales = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    out_causal, pesos_causales = scaled_dot_product_attention(q, k, v, mask=mask)
    out_libre, pesos_libres = scaled_dot_product_attention(q, k, v, mask=None)

    resumen = pd.DataFrame(
        [
            {"objeto": "S", "forma": tuple(scores.shape), "lectura": "puntajes antes de softmax"},
            {"objeto": "M", "forma": tuple(mask.shape), "lectura": "permitida si j <= t"},
            {"objeto": "S con mascara", "forma": tuple(scores_causales.shape), "lectura": "futuro reemplazado por -inf"},
            {"objeto": "A causal", "forma": tuple(pesos_causales.shape), "lectura": "pesos futuros iguales a cero"},
            {"objeto": "Y causal", "forma": tuple(out_causal.shape), "lectura": "salida con restriccion autoregresiva"},
        ]
    )

    filas = []
    for t_idx in range(seq_len):
        masa_futura_causal = float(pesos_causales[0, t_idx, t_idx + 1 :].sum()) if t_idx + 1 < seq_len else 0.0
        masa_futura_libre = float(pesos_libres[0, t_idx, t_idx + 1 :].sum()) if t_idx + 1 < seq_len else 0.0
        filas.append(
            {
                "fila t": t_idx,
                "permitidas": f"0..{t_idx}",
                "bloqueadas": "ninguna" if t_idx == seq_len - 1 else f"{t_idx + 1}..{seq_len - 1}",
                "masa futura causal": masa_futura_causal,
                "masa futura sin mascara": masa_futura_libre,
                "suma fila causal": float(pesos_causales[0, t_idx].sum()),
            }
        )
    comparacion = pd.DataFrame(filas).round(3)

    fila_t = 3
    lectura_fila = []
    for j in range(seq_len):
        permitido = j <= fila_t
        s_masked = "-inf" if not permitido else f"{float(scores_causales[0, fila_t, j]):.3f}"
        lectura_fila.append(
            {
                "j": j,
                "relacion": "diagonal" if j == fila_t else ("pasado" if j < fila_t else "futuro"),
                "S[t,j]": f"{float(scores[0, fila_t, j]):.3f}",
                "S_masked[t,j]": s_masked,
                "A causal": f"{float(pesos_causales[0, fila_t, j]):.3f}",
                "A sin mascara": f"{float(pesos_libres[0, fila_t, j]):.3f}",
            }
        )
    fila = pd.DataFrame(lectura_fila)

    verificaciones = pd.DataFrame(
        [
            {
                "verificacion": "maximo peso causal en futuro",
                "valor": f"{float(pesos_causales[0].triu(1).max()):.6f}",
                "cumple": bool(torch.allclose(pesos_causales[0].triu(1), torch.zeros_like(pesos_causales[0].triu(1)), atol=1e-7)),
            },
            {
                "verificacion": "cada fila causal suma 1",
                "valor": f"{float(pesos_causales[0].sum(dim=-1).mean()):.6f}",
                "cumple": bool(torch.allclose(pesos_causales[0].sum(dim=-1), torch.ones(seq_len), atol=1e-6)),
            },
            {
                "verificacion": "la diagonal queda permitida",
                "valor": f"minimo diagonal A = {float(torch.diagonal(pesos_causales[0]).min()):.6f}",
                "cumple": bool(torch.all(mask.diag())),
            },
        ]
    )

    _ = out_libre
    figura = _figura_atencion_causal(mask, pesos_causales, pesos_libres)
    return ResultadoAtencionCausal(resumen, comparacion, fila, verificaciones, figura)


@torch.no_grad()
def preparar_multi_head_attention(
    batch_size: int = 1,
    seq_len: int = 6,
    n_embd: int = 12,
    n_head: int = 3,
    seed: int = 7,
) -> ResultadoMultiHead:
    """Prepara auditoria y mapas de una multi-head attention causal."""

    if n_embd % n_head != 0:
        raise ValueError("n_embd debe ser divisible por n_head.")

    torch.manual_seed(seed)
    x = torch.randn(batch_size, seq_len, n_embd)
    proyeccion_qkv = torch.nn.Linear(n_embd, 3 * n_embd, bias=False)
    qkv = proyeccion_qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)
    qh, kh, vh = split_heads(q, n_head), split_heads(k, n_head), split_heads(v, n_head)
    mascara = causal_mask(seq_len)
    out_heads, pesos = scaled_dot_product_attention(qh, kh, vh, mask=mascara)
    out_concat = combine_heads(out_heads)
    proyeccion_salida = torch.nn.Linear(n_embd, n_embd, bias=False)
    out = proyeccion_salida(out_concat)

    t_ref = 4
    mascara_futuro = torch.arange(seq_len) > t_ref
    auditoria = pd.DataFrame(
        [
            {"operacion": "entrada", "forma antes": "-", "forma despues": tuple(x.shape), "eje que cambia": "-", "eje que se conserva": "B,T,C"},
            {"operacion": "proyeccion QKV", "forma antes": tuple(x.shape), "forma despues": tuple(qkv.shape), "eje que cambia": "C -> 3C", "eje que se conserva": "B,T"},
            {"operacion": "separar Q/K/V", "forma antes": tuple(qkv.shape), "forma despues": tuple(q.shape), "eje que cambia": "3C -> C", "eje que se conserva": "B,T"},
            {"operacion": "Q en cabezas", "forma antes": tuple(q.shape), "forma despues": tuple(qh.shape), "eje que cambia": "C -> H,D", "eje que se conserva": "B,T"},
            {"operacion": "K en cabezas", "forma antes": tuple(k.shape), "forma despues": tuple(kh.shape), "eje que cambia": "C -> H,D", "eje que se conserva": "B,T"},
            {"operacion": "V en cabezas", "forma antes": tuple(v.shape), "forma despues": tuple(vh.shape), "eje que cambia": "C -> H,D", "eje que se conserva": "B,T"},
            {"operacion": "atencion por cabeza", "forma antes": tuple(qh.shape), "forma despues": tuple(pesos.shape), "eje que cambia": "D -> T", "eje que se conserva": "B,H,T"},
            {"operacion": "mezcla de valores", "forma antes": tuple(pesos.shape), "forma despues": tuple(out_heads.shape), "eje que cambia": "T -> D", "eje que se conserva": "B,H,T"},
            {"operacion": "concatenar cabezas", "forma antes": tuple(out_heads.shape), "forma despues": tuple(out_concat.shape), "eje que cambia": "H,D -> C", "eje que se conserva": "B,T"},
            {"operacion": "proyeccion final W_O", "forma antes": tuple(out_concat.shape), "forma despues": tuple(out.shape), "eje que cambia": "canales mezclados", "eje que se conserva": "B,T,C"},
        ]
    )

    configuracion = pd.DataFrame(
        [
            {"cantidad": "B", "valor": batch_size, "interpretacion": "un ejemplo en el lote"},
            {"cantidad": "T", "valor": seq_len, "interpretacion": "seis posiciones"},
            {"cantidad": "C", "valor": n_embd, "interpretacion": "doce canales totales"},
            {"cantidad": "H", "valor": n_head, "interpretacion": "tres cabezas"},
            {"cantidad": "D=C/H", "valor": n_embd // n_head, "interpretacion": "cuatro canales por cabeza"},
        ]
    )

    traza_fila = pd.DataFrame(
        [
            {
                "cabeza": cabeza,
                "fila A[h,t,:]": [round(float(valor), 3) for valor in pesos[0, cabeza, t_ref]],
                "suma fila": round(float(pesos[0, cabeza, t_ref].sum()), 6),
                "masa futura": round(float(pesos[0, cabeza, t_ref, mascara_futuro].sum()), 6),
                "mayor peso permitido": int(torch.argmax(pesos[0, cabeza, t_ref].masked_fill(mascara_futuro, -1))),
            }
            for cabeza in range(n_head)
        ]
    )

    verificaciones = pd.DataFrame(
        [
            {"prueba": "filas de A suman 1", "valor": round(float((pesos.sum(dim=-1) - 1).abs().max()), 8), "lectura": "error maximo cercano a cero"},
            {"prueba": "masa futura causal", "valor": round(float(pesos.masked_fill(mascara.unsqueeze(0).unsqueeze(0), 0).sum()), 8), "lectura": "debe ser cero"},
            {"prueba": "combine_heads", "valor": str(tuple(out_concat.shape)), "lectura": "recupera B,T,C"},
            {"prueba": "W_O", "valor": str(tuple(out.shape)), "lectura": "conserva la interfaz final"},
        ]
    )

    figura = _figura_multihead(pesos, t_ref)
    return ResultadoMultiHead(auditoria, traza_fila, configuracion, verificaciones, figura, n_head)


def _figura_mezcla_ponderada(values: torch.Tensor, weights: torch.Tensor, out: torch.Tensor) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    colores = ["#255c99", "#2a9d8f", "#577590", "#8ecae6"]
    ax.scatter(
        values[:, 0],
        values[:, 1],
        s=(weights.numpy() * 900) + 110,
        color=colores,
        alpha=0.86,
        edgecolor="#1f2328",
        linewidth=0.8,
        label="valores v_i",
    )
    for i, (x0, y0) in enumerate(values):
        ax.text(float(x0) + 0.035, float(y0) + 0.035, f"v{i}  α={float(weights[i]):.2f}", fontsize=10, color="#1f2a44")
        ax.plot([float(x0), float(out[0])], [float(y0), float(out[1])], color="#c9d1d9", lw=1.2, zorder=0)
    ax.scatter([float(out[0])], [float(out[1])], marker="*", s=360, color="#b91c1c", edgecolor="#7f1d1d", linewidth=0.8, label="salida y", zorder=5)
    ax.annotate(
        "la salida combina, no elige",
        xy=(float(out[0]), float(out[1])),
        xytext=(float(out[0]) + 0.12, float(out[1]) + 0.18),
        arrowprops={"arrowstyle": "->", "color": "#7f1d1d", "lw": 1.2},
        fontsize=10,
        color="#7f1d1d",
    )
    ax.legend(loc="lower right")
    ax.set(title="Salida como promedio ponderado de valores", xlabel="canal 1", ylabel="canal 2", xlim=(-0.12, 1.48), ylim=(-0.08, 1.28))
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def _figura_pesos(
    weights: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str = "peso de atencion",
) -> plt.Figure:
    matriz = weights[0].detach().numpy()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    im = ax.imshow(matriz, vmin=0, vmax=1, cmap="Blues")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks(range(matriz.shape[1]))
    ax.set_xticklabels([f"j={j}" for j in range(matriz.shape[1])])
    ax.set_yticks(range(matriz.shape[0]))
    ax.set_yticklabels([f"t={row}" for row in range(matriz.shape[0])])
    for row in range(matriz.shape[0]):
        for col in range(matriz.shape[1]):
            ax.text(col, row, f"{float(matriz[row, col]):.2f}", ha="center", va="center", color="#1f2a44", fontsize=9)
        ax.text(matriz.shape[1] - 0.1, row + 0.38, f"suma={float(weights[0, row].sum()):.1f}", ha="right", va="center", fontsize=9, color="#57606a")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    return fig


def _figura_atencion_causal(mask: torch.Tensor, pesos_causales: torch.Tensor, pesos_libres: torch.Tensor) -> plt.Figure:
    seq_len = mask.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))
    axes[0].imshow(mask.numpy(), cmap="Greys", vmin=0, vmax=1)
    axes[0].set(title="Mascara causal M", xlabel="posicion consultada j", ylabel="posicion que consulta t")
    im1 = axes[1].imshow(pesos_causales[0].detach().numpy(), cmap="Blues", vmin=0, vmax=1)
    axes[1].set(title="A causal: futuro bloqueado", xlabel="posicion consultada j", ylabel="posicion que consulta t")
    im2 = axes[2].imshow(pesos_libres[0].detach().numpy(), cmap="YlOrRd", vmin=0, vmax=1)
    axes[2].set(title="A sin mascara: futuro visible", xlabel="posicion consultada j", ylabel="posicion que consulta t")
    for ax in axes:
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.axline((-0.5, -0.5), slope=1, color="#0f766e", linewidth=1.2, linestyle="--", alpha=0.8)
    for row in range(seq_len):
        for col in range(seq_len):
            if col > row:
                axes[0].text(col, row, "bloq.", ha="center", va="center", fontsize=7, color="#991b1b")
            axes[1].text(col, row, f"{float(pesos_causales[0, row, col]):.2f}", ha="center", va="center", fontsize=8, color="#1f2a44")
            axes[2].text(col, row, f"{float(pesos_libres[0, row, col]):.2f}", ha="center", va="center", fontsize=8, color="#1f2a44")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _figura_multihead(pesos: torch.Tensor, t_ref: int) -> plt.Figure:
    _, n_head, seq_len, _ = pesos.shape
    fig, axes = plt.subplots(1, n_head, figsize=(10.5, 3.3), sharex=True, sharey=True)
    if n_head == 1:
        axes = [axes]
    im = None
    for i, ax in enumerate(axes):
        matriz = pesos[0, i].detach().numpy()
        im = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"cabeza {i}")
        ax.set_xlabel("posicion consultada j")
        if i == 0:
            ax.set_ylabel("posicion que consulta t")
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.axhline(t_ref - 0.5, color="#0f766e", linewidth=1.4)
        ax.axhline(t_ref + 0.5, color="#0f766e", linewidth=1.4)
        for fila in range(seq_len):
            for columna in range(seq_len):
                color = "white" if matriz[fila, columna] > 0.55 else "#1f2a44"
                ax.text(columna, fila, f"{matriz[fila, columna]:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.suptitle(f"Cada cabeza produce su propio mapa causal T×T; fila resaltada: t={t_ref}", y=1.04)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    return fig
