"""Experimentos auxiliares para arquitectura Transformer."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .modelo import TransformerBlock


@dataclass
class ResultadoResidualesLayerNormMLP:
    operaciones: pd.DataFrame
    estadisticas: pd.DataFrame
    comparacion: pd.DataFrame
    parametros: pd.DataFrame
    figura: plt.Figure


@dataclass
class ResultadoBloqueDecoder:
    traza: pd.DataFrame
    parametros: pd.DataFrame
    verificaciones: pd.DataFrame
    fila_atencion: pd.DataFrame
    figura: plt.Figure


@torch.no_grad()
def preparar_residuales_layernorm_mlp(seed: int = 7) -> ResultadoResidualesLayerNormMLP:
    """Construye la auditoria de LayerNorm, MLP y residual."""

    torch.manual_seed(seed)
    b, t, c = 3, 5, 8
    x = torch.randn(b, t, c) * 4 + 10
    ln = torch.nn.LayerNorm(c)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(c, 4 * c),
        torch.nn.GELU(),
        torch.nn.Linear(4 * c, c),
    )
    x_norm = ln(x)
    cambio = mlp(x_norm)
    y_con_residual = x + cambio
    y_sin_residual = cambio

    operaciones = pd.DataFrame(
        [
            {"operacion": "entrada", "forma antes": "-", "forma despues": tuple(x.shape), "eje que cambia": "-", "eje que se conserva": "B,T,C"},
            {"operacion": "LayerNorm", "forma antes": tuple(x.shape), "forma despues": tuple(x_norm.shape), "eje que cambia": "escala de C", "eje que se conserva": "B,T,C"},
            {"operacion": "MLP C->4C->C", "forma antes": tuple(x_norm.shape), "forma despues": tuple(cambio.shape), "eje que cambia": "canales por posicion", "eje que se conserva": "B,T"},
            {"operacion": "residual", "forma antes": f"{tuple(x.shape)} + {tuple(cambio.shape)}", "forma despues": tuple(y_con_residual.shape), "eje que cambia": "valor de activacion", "eje que se conserva": "B,T,C"},
        ]
    )
    estadisticas = pd.DataFrame(
        [
            {"medida": "media global antes LN", "valor": round(float(x.mean()), 4), "lectura": "escala original del tensor"},
            {"medida": "std global antes LN", "valor": round(float(x.std()), 4), "lectura": "dispersion original"},
            {"medida": "|media por posicion| despues LN", "valor": round(float(x_norm.mean(dim=-1).abs().max()), 8), "lectura": "maximo debe quedar cerca de 0"},
            {"medida": "std por posicion despues LN", "valor": round(float(x_norm.std(dim=-1, unbiased=False).mean()), 8), "lectura": "promedio debe quedar cerca de 1"},
            {"medida": "forma conservada", "valor": "si" if x.shape == y_con_residual.shape else "no", "lectura": "la suma residual esta definida"},
        ]
    )
    comparacion = pd.DataFrame(
        [
            {"salida": "X", "norma": round(float(x.norm()), 4), "lectura": "senal que puede viajar por residual"},
            {"salida": "MLP(LN(X))", "norma": round(float(cambio.norm()), 4), "lectura": "correccion aprendible"},
            {"salida": "con residual", "norma": round(float(y_con_residual.norm()), 4), "lectura": "entrada mas correccion"},
            {"salida": "sin residual", "norma": round(float(y_sin_residual.norm()), 4), "lectura": "solo correccion; no implica ser mejor"},
        ]
    )
    parametros = pd.DataFrame(
        [
            {"componente": "LayerNorm", "parametros": sum(p.numel() for p in ln.parameters()), "lectura": "gamma y beta por canal"},
            {"componente": "MLP", "parametros": sum(p.numel() for p in mlp.parameters()), "lectura": "expansion C->4C y retorno 4C->C"},
            {"componente": "Residual", "parametros": 0, "lectura": "suma directa; no anade pesos"},
        ]
    )

    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    ax.hist(x.flatten().detach().numpy(), bins=25, alpha=0.58, label="antes de LayerNorm", color="#4f7fd5", edgecolor="white")
    ax.hist(x_norm.flatten().detach().numpy(), bins=25, alpha=0.72, label="despues de LayerNorm", color="#0f766e", edgecolor="white")
    ax.axvline(float(x_norm.mean()), color="#0f766e", linestyle="--", linewidth=1.2)
    ax.legend(frameon=False)
    ax.set(title="LayerNorm estabiliza escala por posicion", xlabel="valor de activacion", ylabel="frecuencia")
    fig.tight_layout()

    return ResultadoResidualesLayerNormMLP(operaciones, estadisticas, comparacion, parametros, fig)


@torch.no_grad()
def preparar_bloque_decoder(seed: int = 7) -> ResultadoBloqueDecoder:
    """Construye una traza interna de un bloque Transformer decoder."""

    torch.manual_seed(seed)
    b, t, c, h = 2, 8, 16, 4
    block = TransformerBlock(n_embd=c, n_head=h, block_size=t, dropout=0.0)
    x = torch.randn(b, t, c)
    x_ln1 = block.ln1(x)
    attn_out, att = block.attn(x_ln1, return_attention=True)
    u = x + attn_out
    u_ln2 = block.ln2(u)
    mlp_out = block.ffwd(u_ln2)
    y = u + mlp_out

    t_ref = 5
    mascara = torch.tril(torch.ones(t, t, dtype=torch.bool))
    traza = pd.DataFrame(
        [
            {"paso": "X", "forma": tuple(x.shape), "eje que opera": "-", "lectura": "entrada del bloque"},
            {"paso": "LN1(X)", "forma": tuple(x_ln1.shape), "eje que opera": "C", "lectura": "normaliza antes de atencion"},
            {"paso": "MHA causal", "forma": tuple(attn_out.shape), "eje que opera": "T", "lectura": "mezcla posiciones permitidas"},
            {"paso": "U = X + MHA", "forma": tuple(u.shape), "eje que opera": "B,T,C", "lectura": "residual 1"},
            {"paso": "LN2(U)", "forma": tuple(u_ln2.shape), "eje que opera": "C", "lectura": "normaliza antes de MLP"},
            {"paso": "MLP", "forma": tuple(mlp_out.shape), "eje que opera": "C", "lectura": "transforma canales por posicion"},
            {"paso": "Y = U + MLP", "forma": tuple(y.shape), "eje que opera": "B,T,C", "lectura": "residual 2"},
            {"paso": "A", "forma": tuple(att.shape), "eje que opera": "H,T,T", "lectura": "pesos de atencion por cabeza"},
        ]
    )
    parametros = pd.DataFrame(
        [
            {"componente": "ln1", "parametros": sum(p.numel() for p in block.ln1.parameters()), "lectura": "normalizacion antes de atencion"},
            {"componente": "attn", "parametros": sum(p.numel() for p in block.attn.parameters()), "lectura": "QKV, proyeccion final y sesgos"},
            {"componente": "ln2", "parametros": sum(p.numel() for p in block.ln2.parameters()), "lectura": "normalizacion antes de MLP"},
            {"componente": "ffwd", "parametros": sum(p.numel() for p in block.ffwd.parameters()), "lectura": "MLP C->4C->C"},
        ]
    )
    masa_futura = float(att.masked_fill(mascara.unsqueeze(0).unsqueeze(0), 0).sum().item())
    verificaciones = pd.DataFrame(
        [
            {"verificacion": "X, U y Y conservan forma", "resultado": "si" if x.shape == u.shape == y.shape else "no", "lectura": "necesario para residuales y apilamiento"},
            {"verificacion": "C = H · D", "resultado": f"{c} = {h} · {c // h}", "lectura": "canales por cabeza enteros"},
            {"verificacion": "masa futura total", "resultado": round(masa_futura, 6), "lectura": "debe ser 0 por mascara causal"},
            {"verificacion": "suma media por fila", "resultado": round(float(att.sum(dim=-1).mean()), 6), "lectura": "cada fila suma 1 con dropout=0"},
        ]
    )
    fila_atencion = pd.DataFrame(
        [
            {"j": j, "A[0,0,t,j]": round(float(att[0, 0, t_ref, j]), 4), "estado": "permitida" if j <= t_ref else "futuro bloqueado"}
            for j in range(t)
        ]
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    matriz = att[0, 0].detach().numpy()
    im = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=1)
    ax.set(title="Atencion causal interna (dropout=0)", xlabel="posicion consultada j", ylabel="posicion que consulta t")
    ax.set_xticks(range(t))
    ax.set_yticks(range(t))
    for fila in range(t):
        for columna in range(t):
            color = "white" if matriz[fila, columna] > 0.55 else "#1f2a44"
            ax.text(columna, fila, f"{matriz[fila, columna]:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    return ResultadoBloqueDecoder(traza, parametros, verificaciones, fila_atencion, fig)
