"""Experimentos auxiliares para generación autoregresiva."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from .datos import TextoAutoregresivo
from .entrenamiento import entrenar_pasos
from .generacion import generar_texto
from .modelo import MiniTransformerLM
from .tokenizacion import codificar, construir_vocabulario, tokenizar_caracteres


@dataclass
class ResultadoPreparacionGeneracion:
    corpus: str
    token_a_id: dict[str, int]
    id_a_token: dict[int, str]
    ids: torch.Tensor
    block_size: int
    config: dict
    model: MiniTransformerLM
    history: dict[str, list[float]]
    resumen: pd.DataFrame
    configuracion: pd.DataFrame
    lectura: pd.DataFrame


@dataclass
class ResultadoGeneraciones:
    prompt: str
    configuraciones: list[dict]
    configuraciones_df: pd.DataFrame
    generaciones: pd.DataFrame


@dataclass
class ResultadoAuditoriaGenerativa:
    prompt_ids: torch.Tensor
    idx_cond: torch.Tensor
    logits: torch.Tensor
    last_logits: torch.Tensor
    attention: torch.Tensor
    auditoria: pd.DataFrame
    top_base: pd.DataFrame
    lectura: pd.DataFrame


@dataclass
class ResultadoFiltrosGeneracion:
    resumen: pd.DataFrame
    candidatos: pd.DataFrame


@dataclass
class ResultadoTrazaGenerativa:
    traza: pd.DataFrame
    lectura: pd.DataFrame


@dataclass
class ResultadoAtencionGeneracion:
    figura: plt.Figure
    lectura: pd.DataFrame


@dataclass
class ResultadoPruebasMuestreo:
    pruebas: pd.DataFrame
    lectura: pd.DataFrame


def preparar_demo_generacion(seed: int = 7) -> ResultadoPreparacionGeneracion:
    torch.manual_seed(seed)
    corpus = ("atencion mezcla contexto. transformer aprende patrones. " * 8).lower()
    tokens = tokenizar_caracteres(corpus)
    token_a_id, id_a_token = construir_vocabulario(tokens)
    ids = torch.tensor(codificar(tokens, token_a_id), dtype=torch.long)
    block_size = 16
    config = {"vocab_size": len(token_a_id), "block_size": block_size, "n_embd": 24, "n_head": 4, "n_layer": 1, "dropout": 0.0}
    data = TextoAutoregresivo(ids, block_size=block_size)
    model = MiniTransformerLM(**config)
    history = entrenar_pasos(model, data, data, steps=25, batch_size=16, lr=3e-3, eval_every=25)
    resumen = pd.DataFrame(
        [
            {"objeto": "caracteres del corpus", "valor": len(corpus), "lectura": "texto sintetico y repetitivo"},
            {"objeto": "tokens", "valor": len(tokens), "lectura": "tokenizacion por caracteres"},
            {"objeto": "Vocab", "valor": len(token_a_id), "lectura": "candidatos posibles en cada paso"},
            {"objeto": "block_size", "valor": block_size, "lectura": "longitud maxima de contexto"},
            {"objeto": "train_loss inicial", "valor": round(history["train"][0], 4), "lectura": "antes de las actualizaciones principales"},
            {"objeto": "train_loss final", "valor": round(history["train"][-1], 4), "lectura": "despues del entrenamiento breve"},
        ]
    )
    configuracion = pd.DataFrame([{"parametro": clave, "valor": valor} for clave, valor in config.items()])
    lectura = pd.DataFrame(
        [
            {"decision": "modelo pequeno", "efecto": "ejecuta rapido", "cautela": "no produce lenguaje robusto"},
            {"decision": "corpus repetitivo", "efecto": "permite observar patrones", "cautela": "favorece repeticion/memorizacion"},
            {"decision": "mismo dato train/eval", "efecto": "simplifica la demo", "cautela": "no estima generalizacion real"},
        ]
    )
    return ResultadoPreparacionGeneracion(corpus, token_a_id, id_a_token, ids, block_size, config, model, history, resumen, configuracion, lectura)


def configuraciones_muestreo_base() -> list[dict]:
    return [
        {"estrategia": "greedy / top_k=1", "temperature": 1.0, "top_k": 1, "top_p": None, "lectura": "un solo candidato disponible"},
        {"estrategia": "temperatura + top_k", "temperature": 0.7, "top_k": 6, "top_p": None, "lectura": "concentra y limita a k candidatos"},
        {"estrategia": "temperatura + top_p", "temperature": 1.1, "top_k": None, "top_p": 0.85, "lectura": "conserva una masa acumulada"},
    ]


def preparar_generaciones_comparadas(model, token_a_id, id_a_token, prompt: str = "atencion") -> ResultadoGeneraciones:
    configuraciones = configuraciones_muestreo_base()
    generaciones = []
    for cfg in configuraciones:
        torch.manual_seed(11)
        texto = generar_texto(model, prompt, token_a_id, id_a_token, max_new_tokens=45, temperature=cfg["temperature"], top_k=cfg["top_k"], top_p=cfg["top_p"])
        generaciones.append(
            {
                "estrategia": cfg["estrategia"],
                "temperature": cfg["temperature"],
                "top_k": cfg["top_k"] if cfg["top_k"] is not None else "-",
                "top_p": cfg["top_p"] if cfg["top_p"] is not None else "-",
                "prompt": prompt,
                "texto generado": repr(texto),
                "longitud": len(texto),
            }
        )
    configuraciones_df = pd.DataFrame(
        [
            {"estrategia": cfg["estrategia"], "temperature": cfg["temperature"], "top_k": cfg["top_k"] if cfg["top_k"] is not None else "-", "top_p": cfg["top_p"] if cfg["top_p"] is not None else "-", "lectura": cfg["lectura"]}
            for cfg in configuraciones
        ]
    )
    return ResultadoGeneraciones(prompt, configuraciones, configuraciones_df, pd.DataFrame(generaciones))


def preparar_auditoria_vuelta_generativa(model, token_a_id, id_a_token, block_size: int, prompt: str) -> ResultadoAuditoriaGenerativa:
    prompt_ids = torch.tensor([[token_a_id[c] for c in prompt[-block_size:] if c in token_a_id]], dtype=torch.long)
    idx_cond = prompt_ids[:, -model.block_size :]
    with torch.no_grad():
        logits, _, att = model(idx_cond, return_attention=True)
    last_logits = logits[:, -1, :]
    base_probs = F.softmax(last_logits, dim=-1)
    base_top_probs, base_top_ids = torch.topk(base_probs, k=8, dim=-1)
    auditoria = pd.DataFrame(
        [
            {"objeto": "prompt", "forma": (1, len(prompt_ids[0])), "lectura": "ids iniciales del prompt"},
            {"objeto": "idx_cond", "forma": tuple(idx_cond.shape), "lectura": "contexto recortado usado por el modelo"},
            {"objeto": "logits", "forma": tuple(logits.shape), "lectura": "puntajes para todas las posiciones"},
            {"objeto": "last_logits", "forma": tuple(last_logits.shape), "lectura": "unica fila usada para el siguiente token"},
            {"objeto": "base_probs", "forma": tuple(base_probs.shape), "lectura": "distribucion sin filtros adicionales"},
            {"objeto": "attention", "forma": tuple(att.shape), "lectura": "atencion de la primera capa"},
        ]
    )
    top_base = pd.DataFrame(
        [
            {"rank": rank + 1, "token id": int(token_id), "token": repr(id_a_token[int(token_id)]), "probabilidad base": round(float(prob), 4)}
            for rank, (prob, token_id) in enumerate(zip(base_top_probs[0], base_top_ids[0]))
        ]
    )
    lectura = pd.DataFrame(
        [
            {"paso": "recorte", "objeto": "idx[:, -block_size:]", "lectura": "solo entra el contexto permitido"},
            {"paso": "modelo", "objeto": "logits", "lectura": "produce puntajes para todas las posiciones visibles"},
            {"paso": "seleccion de fila", "objeto": "logits[:, -1, :]", "lectura": "solo la ultima posicion predice el siguiente token"},
            {"paso": "softmax/filtros", "objeto": "probs", "lectura": "define distribucion de muestreo"},
        ]
    )
    return ResultadoAuditoriaGenerativa(prompt_ids, idx_cond, logits, last_logits, att, auditoria, top_base, lectura)


def distribucion_filtrada(logits_1v: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    z = logits_1v.clone() / max(float(temperature), 1e-6)
    if top_k is not None:
        values, _ = torch.topk(z, min(int(top_k), z.size(-1)))
        z = z.masked_fill(z < values[:, [-1]], -float("inf"))
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(z, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_sorted = cumulative > float(top_p)
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(z, dtype=torch.bool)
        remove.scatter_(dim=-1, index=sorted_indices, src=remove_sorted)
        z = z.masked_fill(remove, -float("inf"))
    return F.softmax(z, dim=-1)


def entropia(probs: torch.Tensor) -> float:
    p = probs[probs > 0]
    return float(-(p * torch.log(p)).sum())


def preparar_filtros_generacion(last_logits: torch.Tensor, id_a_token: dict[int, str], configuraciones: list[dict]) -> ResultadoFiltrosGeneracion:
    filas, resumen_filtros = [], []
    for cfg in configuraciones:
        probs = distribucion_filtrada(last_logits, temperature=cfg["temperature"], top_k=cfg["top_k"], top_p=cfg["top_p"])
        top_probs, top_ids = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1)
        resumen_filtros.append(
            {
                "estrategia": cfg["estrategia"],
                "candidatos > 0": int((probs > 0).sum()),
                "masa top-3": round(float(torch.topk(probs, k=min(3, probs.shape[-1]), dim=-1).values.sum()), 4),
                "entropia aprox.": round(entropia(probs), 4),
                "lectura": cfg["lectura"],
            }
        )
        for rank, (prob, token_id) in enumerate(zip(top_probs[0], top_ids[0]), start=1):
            if float(prob) > 0:
                filas.append({"estrategia": cfg["estrategia"], "rank": rank, "token": repr(id_a_token[int(token_id)]), "token id": int(token_id), "probabilidad": round(float(prob), 4)})
    return ResultadoFiltrosGeneracion(pd.DataFrame(resumen_filtros), pd.DataFrame(filas))


def preparar_traza_generativa(model, token_a_id, id_a_token, prompt: str) -> ResultadoTrazaGenerativa:
    torch.manual_seed(19)
    idx_traza = torch.tensor([[token_a_id[c] for c in prompt if c in token_a_id]], dtype=torch.long)
    filas_traza = []
    for paso in range(1, 4):
        idx_cond_paso = idx_traza[:, -model.block_size :]
        contexto_visible = "".join(id_a_token[int(i)] for i in idx_cond_paso[0])
        hubo_recorte = idx_traza.shape[1] > model.block_size
        with torch.no_grad():
            logits_paso, _ = model(idx_cond_paso)
        probs_paso = distribucion_filtrada(logits_paso[:, -1, :], temperature=0.9, top_k=6, top_p=None)
        next_id = torch.multinomial(probs_paso, num_samples=1)
        token_elegido = id_a_token[int(next_id.item())]
        prob_elegida = float(probs_paso[0, next_id.item()])
        idx_traza = torch.cat([idx_traza, next_id], dim=1)
        contexto_actualizado = "".join(id_a_token[int(i)] for i in idx_traza[0])
        filas_traza.append(
            {
                "paso": paso,
                "contexto visible": repr(contexto_visible),
                "forma idx_cond": tuple(idx_cond_paso.shape),
                "token elegido": repr(token_elegido),
                "prob. token": round(prob_elegida, 4),
                "contexto actualizado": repr(contexto_actualizado),
                "recorte block_size": "si" if hubo_recorte else "no",
            }
        )
    lectura = pd.DataFrame(
        [
            {"criterio": "ultima posicion", "lectura": "cada paso usa logits[:, -1, :]"},
            {"criterio": "realimentacion", "lectura": "el token elegido cambia el siguiente contexto"},
            {"criterio": "block_size", "lectura": "si el contexto crece, se conservan las ultimas posiciones"},
        ]
    )
    return ResultadoTrazaGenerativa(pd.DataFrame(filas_traza), lectura)


def preparar_atencion_generacion(att: torch.Tensor, idx_cond: torch.Tensor, id_a_token: dict[int, str]) -> ResultadoAtencionGeneracion:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    matriz = att[0, 0].detach().numpy()
    im = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=1)
    labels = [repr(id_a_token[int(i)]) for i in idx_cond[0]]
    ax.set(title="Atencion de una cabeza durante generacion", xlabel="posicion consultada j", ylabel="posicion que consulta t")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for row in range(matriz.shape[0]):
        for col in range(matriz.shape[1]):
            color = "white" if matriz[row, col] > 0.55 else "#1f2a44"
            ax.text(col, row, f"{matriz[row, col]:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="peso")
    fig.tight_layout()
    future_mask = torch.triu(torch.ones_like(att[0, 0], dtype=torch.bool), diagonal=1)
    lectura = pd.DataFrame(
        [
            {"criterio": "forma attention", "valor": tuple(att.shape), "lectura": "B,H,T,T"},
            {"criterio": "cabeza visualizada", "valor": "h=0", "lectura": "una sola cabeza de la primera capa"},
            {"criterio": "suma de filas", "valor": f"{float(att[0, 0].sum(dim=-1).min()):.3f} a {float(att[0, 0].sum(dim=-1).max()):.3f}", "lectura": "cada fila se normaliza"},
            {"criterio": "masa futura", "valor": f"{float(att[0, 0].masked_select(future_mask).sum()):.3f}", "lectura": "la mascara causal bloquea j>t"},
            {"criterio": "cautela", "valor": "necesaria", "lectura": "el mapa no explica por si solo la salida"},
        ]
    )
    return ResultadoAtencionGeneracion(fig, lectura)


def preparar_pruebas_muestreo(model, token_a_id, id_a_token, prompt: str, last_logits: torch.Tensor) -> ResultadoPruebasMuestreo:
    pruebas = [
        {"nombre": "temperatura baja", "temperature": 0.6, "top_k": 6, "top_p": None},
        {"nombre": "temperatura alta", "temperature": 1.3, "top_k": 6, "top_p": None},
        {"nombre": "top_p estricto", "temperature": 1.0, "top_k": None, "top_p": 0.70},
    ]
    registros = []
    for cfg in pruebas:
        torch.manual_seed(13)
        texto = generar_texto(model, prompt, token_a_id, id_a_token, max_new_tokens=35, temperature=cfg["temperature"], top_k=cfg["top_k"], top_p=cfg["top_p"])
        probs = distribucion_filtrada(last_logits, cfg["temperature"], cfg["top_k"], cfg["top_p"])
        registros.append(
            {
                "prueba": cfg["nombre"],
                "temperature": cfg["temperature"],
                "top_k": cfg["top_k"] if cfg["top_k"] is not None else "-",
                "top_p": cfg["top_p"] if cfg["top_p"] is not None else "-",
                "candidatos > 0": int((probs > 0).sum()),
                "masa top-3": round(float(torch.topk(probs, k=3, dim=-1).values.sum()), 4),
                "entropia aprox.": round(entropia(probs), 4),
                "texto generado": repr(texto),
            }
        )
    lectura = pd.DataFrame(
        [
            {"criterio": "prompt fijo", "lectura": "permite atribuir cambios a la estrategia"},
            {"criterio": "una variable", "lectura": "evita comparar configuraciones mezcladas"},
            {"criterio": "evidencia minima", "lectura": "candidatos, masa top-3, entropia y salida"},
            {"criterio": "cautela", "lectura": "una salida llamativa no prueba comprension"},
        ]
    )
    return ResultadoPruebasMuestreo(pd.DataFrame(registros), lectura)
