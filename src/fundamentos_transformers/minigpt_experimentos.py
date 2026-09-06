"""Experimentos auxiliares para integrar un Mini-GPT."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from .modelo import MiniTransformerLM, contar_parametros


CONFIG_MINIGPT = {
    "vocab_size": 20,
    "block_size": 8,
    "n_embd": 16,
    "n_head": 4,
    "n_layer": 2,
    "dropout": 0.0,
}


@dataclass
class ResultadoConfiguracionMiniGPT:
    config: dict
    model: MiniTransformerLM
    configuracion: pd.DataFrame
    restricciones: pd.DataFrame


@dataclass
class ResultadoBatchMiniGPT:
    idx: torch.Tensor
    targets: torch.Tensor
    batch: pd.DataFrame
    formas: pd.DataFrame


@dataclass
class ResultadoForwardMiniGPT:
    logits: torch.Tensor
    loss: torch.Tensor
    attention: torch.Tensor
    traza: pd.DataFrame
    verificaciones: pd.DataFrame
    perdida: pd.DataFrame


@dataclass
class ResultadoParametrosMiniGPT:
    parametros: pd.DataFrame
    decisiones: pd.DataFrame


@dataclass
class ResultadoInspeccionLogits:
    top_tokens: pd.DataFrame
    objetivo: pd.DataFrame
    pasos: pd.DataFrame
    figura: plt.Figure


@dataclass
class ResultadoVariantesMiniGPT:
    variantes: pd.DataFrame
    lectura: pd.DataFrame


def preparar_configuracion_minigpt(config: dict | None = None) -> ResultadoConfiguracionMiniGPT:
    cfg = dict(CONFIG_MINIGPT if config is None else config)
    model = MiniTransformerLM(**cfg)
    model.eval()
    configuracion = pd.DataFrame(
        [
            {"parametro": "vocab_size", "valor": cfg["vocab_size"], "interpretacion": "numero de tokens posibles en la salida"},
            {"parametro": "block_size", "valor": cfg["block_size"], "interpretacion": "longitud maxima de contexto"},
            {"parametro": "n_embd", "valor": cfg["n_embd"], "interpretacion": "dimension interna C"},
            {"parametro": "n_head", "valor": cfg["n_head"], "interpretacion": "cabezas de atencion por bloque"},
            {"parametro": "n_layer", "valor": cfg["n_layer"], "interpretacion": "bloques decoder apilados"},
            {"parametro": "dropout", "valor": cfg["dropout"], "interpretacion": "desactivado para inspeccion deterministica"},
        ]
    )
    restricciones = pd.DataFrame(
        [
            {"restriccion": "C = H·D", "valor": f"{cfg['n_embd']} = {cfg['n_head']}·{cfg['n_embd'] // cfg['n_head']}", "lectura": "las cabezas dividen canales"},
            {"restriccion": "T <= block_size", "valor": f"T <= {cfg['block_size']}", "lectura": "la tabla posicional tiene longitud finita"},
            {"restriccion": "dropout=0", "valor": "si", "lectura": "las filas de atencion se pueden leer sin ruido"},
        ]
    )
    return ResultadoConfiguracionMiniGPT(cfg, model, configuracion, restricciones)


def preparar_batch_sintetico_minigpt() -> ResultadoBatchMiniGPT:
    idx = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [3, 4, 5, 6, 7, 8, 9, 10],
            [5, 6, 7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [2, 3, 4, 5, 6, 7, 8, 0],
            [4, 5, 6, 7, 8, 9, 10, 0],
            [6, 7, 8, 9, 10, 11, 12, 0],
        ],
        dtype=torch.long,
    )
    batch = pd.DataFrame(
        [
            {"ejemplo": b, "posicion": t, "idx[b,t]": int(idx[b, t]), "target[b,t]": int(targets[b, t]), "lectura": "target es el token observado que se penaliza en esa posicion"}
            for b in range(idx.shape[0])
            for t in range(idx.shape[1])
        ]
    )
    formas = pd.DataFrame(
        [
            {"objeto": "idx", "forma": tuple(idx.shape), "lectura": "contextos de entrada"},
            {"objeto": "targets", "forma": tuple(targets.shape), "lectura": "objetivos alineados posicion por posicion"},
            {"objeto": "B·T", "forma": int(idx.numel()), "lectura": "numero de decisiones locales evaluadas por la perdida"},
        ]
    )
    return ResultadoBatchMiniGPT(idx, targets, batch, formas)


def preparar_forward_minigpt(model: MiniTransformerLM, config: dict, idx: torch.Tensor, targets: torch.Tensor) -> ResultadoForwardMiniGPT:
    with torch.no_grad():
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        tok_emb = model.token_embedding(idx)
        pos_emb = model.position_embedding(pos)[None, :, :]
        x_inicial = tok_emb + pos_emb
        x_drop = model.drop(x_inicial)
        x_bloques = x_drop
        attention = None
        for i, block in enumerate(model.blocks):
            if i == 0:
                x_bloques, attention = block(x_bloques, return_attention=True)
            else:
                x_bloques = block(x_bloques)
        x_ln = model.ln_f(x_bloques)
        logits_manual = model.lm_head(x_ln)
        loss_manual = F.cross_entropy(logits_manual.reshape(b * t, config["vocab_size"]), targets.reshape(b * t))
        logits, loss, attention_forward = model(idx, targets, return_attention=True)

    future_mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    future_mass = float(attention.masked_select(future_mask[None, None, :, :]).sum())
    row_sums = attention.sum(dim=-1)
    traza = pd.DataFrame(
        [
            {"objeto": "idx", "forma": tuple(idx.shape), "ejes": "B,T", "lectura": "indices discretos"},
            {"objeto": "E_tok(idx)", "forma": tuple(tok_emb.shape), "ejes": "B,T,C", "lectura": "contenido tokenizado"},
            {"objeto": "E_pos[0:T]", "forma": tuple(pos_emb.shape), "ejes": "1,T,C", "lectura": "posicion compartida sobre batch"},
            {"objeto": "X inicial", "forma": tuple(x_inicial.shape), "ejes": "B,T,C", "lectura": "contenido mas posicion"},
            {"objeto": "dropout(X)", "forma": tuple(x_drop.shape), "ejes": "B,T,C", "lectura": "sin cambio de forma"},
            {"objeto": "bloques decoder", "forma": tuple(x_bloques.shape), "ejes": "B,T,C", "lectura": "contextualizan y conservan interfaz"},
            {"objeto": "LN final", "forma": tuple(x_ln.shape), "ejes": "B,T,C", "lectura": "normaliza antes del vocabulario"},
            {"objeto": "logits", "forma": tuple(logits.shape), "ejes": "B,T,Vocab", "lectura": "puntajes por token candidato"},
            {"objeto": "attention", "forma": tuple(attention.shape), "ejes": "B,H,T,T", "lectura": "pesos de la primera capa"},
        ]
    )
    verificaciones = pd.DataFrame(
        [
            {"verificacion": "logits manuales = forward", "resultado": bool(torch.allclose(logits_manual, logits)), "lectura": "la traza reproduce el modelo"},
            {"verificacion": "loss manual = forward", "resultado": bool(torch.allclose(loss_manual, loss)), "lectura": "cross-entropy usa logits aplanados"},
            {"verificacion": "targets.shape == idx.shape", "resultado": bool(targets.shape == idx.shape), "lectura": "un objetivo por posicion"},
            {"verificacion": "C divisible por H", "resultado": bool(config["n_embd"] % config["n_head"] == 0), "lectura": "existe D=C/H"},
            {"verificacion": "filas de atencion suman 1", "resultado": f"{float(row_sums.min()):.3f} a {float(row_sums.max()):.3f}", "lectura": "dropout=0 permite leer softmax"},
            {"verificacion": "masa futura causal", "resultado": f"{future_mass:.3f}", "lectura": "las posiciones futuras quedan bloqueadas"},
        ]
    )
    perdida = pd.DataFrame(
        [
            {"metrica": "loss", "valor": round(float(loss.detach()), 4), "lectura": "promedio sobre B·T decisiones locales"},
            {"metrica": "logits para CE", "valor": str((b * t, config["vocab_size"])), "lectura": "forma conceptual que recibe cross_entropy"},
            {"metrica": "targets para CE", "valor": str((b * t,)), "lectura": "indice observado por decision local"},
            {"metrica": "parametros entrenables", "valor": contar_parametros(model), "lectura": "pesos ajustables durante entrenamiento"},
        ]
    )
    _ = attention_forward
    return ResultadoForwardMiniGPT(logits, loss, attention, traza, verificaciones, perdida)


def _contar_parametros_modulo(modulo) -> int:
    return sum(p.numel() for p in modulo.parameters() if p.requires_grad)


def preparar_parametros_minigpt(model: MiniTransformerLM) -> ResultadoParametrosMiniGPT:
    total = contar_parametros(model)
    parametros = pd.DataFrame(
        [
            {"componente": "token_embedding", "parametros": _contar_parametros_modulo(model.token_embedding), "lectura": "tabla Vocab x C para contenido de tokens"},
            {"componente": "position_embedding", "parametros": _contar_parametros_modulo(model.position_embedding), "lectura": "tabla block_size x C para posiciones"},
            {"componente": "blocks", "parametros": _contar_parametros_modulo(model.blocks), "lectura": "atencion causal, MLP, LayerNorm y residuales aprendibles"},
            {"componente": "ln_f", "parametros": _contar_parametros_modulo(model.ln_f), "lectura": "normalizacion final"},
            {"componente": "lm_head", "parametros": _contar_parametros_modulo(model.lm_head), "lectura": "proyeccion C -> Vocab para logits"},
        ]
    )
    parametros["porcentaje"] = (100 * parametros["parametros"] / total).round(2)
    parametros.loc[len(parametros)] = ["total", total, "todos los pesos entrenables", 100.0]
    decisiones = pd.DataFrame(
        [
            {"si aumenta": "vocab_size", "cambian": "token_embedding y lm_head", "lectura": "mas tokens posibles de entrada/salida"},
            {"si aumenta": "block_size", "cambian": "position_embedding", "lectura": "mas posiciones absolutas aprendidas"},
            {"si aumenta": "n_layer", "cambian": "blocks", "lectura": "mas bloques decoder apilados"},
            {"si aumenta": "n_embd", "cambian": "casi todos", "lectura": "mas canales internos y mas matrices grandes"},
        ]
    )
    return ResultadoParametrosMiniGPT(parametros, decisiones)


def preparar_inspeccion_logits(logits: torch.Tensor, targets: torch.Tensor, b_sel: int = 0, t_sel: int = 3) -> ResultadoInspeccionLogits:
    logits_pos = logits[b_sel, t_sel].detach()
    probs_pos = torch.softmax(logits_pos, dim=-1)
    target_id = int(targets[b_sel, t_sel])
    valores, indices = torch.topk(probs_pos, k=5)
    loss_local = F.cross_entropy(logits_pos[None, :], targets[b_sel, t_sel][None])
    top_tokens = pd.DataFrame(
        [
            {"token candidato": int(token_id), "logit": round(float(logits_pos[token_id]), 4), "probabilidad softmax": round(float(prob), 4), "es target observado": "si" if int(token_id) == target_id else "no"}
            for token_id, prob in zip(indices.tolist(), valores.tolist())
        ]
    )
    objetivo = pd.DataFrame(
        [
            {"batch": b_sel, "posicion": t_sel, "target observado": target_id, "logit target": round(float(logits_pos[target_id]), 4), "probabilidad target": round(float(probs_pos[target_id]), 4), "loss local": round(float(loss_local.detach()), 4)}
        ]
    )
    pasos = pd.DataFrame(
        [
            {"paso": "logits", "objeto": "Z[b,t,:]", "lectura": "puntajes sin normalizar"},
            {"paso": "softmax", "objeto": "p[b,t,:]", "lectura": "distribucion sobre Vocab"},
            {"paso": "perdida", "objeto": "target[b,t]", "lectura": "penaliza baja probabilidad del token observado"},
            {"paso": "seleccion", "objeto": "posterior", "lectura": "argmax o muestreo se estudian despues"},
        ]
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colores = ["#9b2226" if int(token_id) == target_id else "#1f77b4" for token_id in indices.tolist()]
    ax.bar([str(int(i)) for i in indices], valores.detach().numpy(), color=colores)
    ax.set(title=f"Top-5 de probabilidades en b={b_sel}, t={t_sel}", xlabel="token candidato", ylabel="probabilidad")
    fig.tight_layout()
    return ResultadoInspeccionLogits(top_tokens, objetivo, pasos, fig)


def preparar_variantes_minigpt(config: dict) -> ResultadoVariantesMiniGPT:
    variantes = [
        {"nombre": "base", **config},
        {"nombre": "mas vocabulario", **{**config, "vocab_size": 32}},
        {"nombre": "mas contexto", **{**config, "block_size": 12}},
        {"nombre": "mas canales", **{**config, "n_embd": 32, "n_head": 4}},
        {"nombre": "mas capas", **{**config, "n_layer": 3}},
    ]
    registros = []
    for especificacion in variantes:
        cfg = dict(especificacion)
        nombre = cfg.pop("nombre")
        variante = MiniTransformerLM(**cfg)
        b_var, t_var = 2, min(8, cfg["block_size"])
        entrada = torch.randint(0, cfg["vocab_size"], (b_var, t_var))
        objetivo = torch.randint(0, cfg["vocab_size"], (b_var, t_var))
        with torch.no_grad():
            z, perdida = variante(entrada, objetivo)
        registros.append(
            {
                "variante": nombre,
                "vocab_size": cfg["vocab_size"],
                "block_size": cfg["block_size"],
                "n_embd": cfg["n_embd"],
                "n_head": cfg["n_head"],
                "n_layer": cfg["n_layer"],
                "logits": tuple(z.shape),
                "loss": round(float(perdida.detach()), 4),
                "parametros": contar_parametros(variante),
            }
        )
    lectura = pd.DataFrame(
        [
            {"cambio": "vocab_size", "lectura": "cambia la dimension V de logits"},
            {"cambio": "block_size", "lectura": "cambia el maximo contexto permitido"},
            {"cambio": "n_embd", "lectura": "aumenta canales internos y costo"},
            {"cambio": "n_layer", "lectura": "aumenta profundidad y parametros"},
        ]
    )
    return ResultadoVariantesMiniGPT(pd.DataFrame(registros), lectura)
