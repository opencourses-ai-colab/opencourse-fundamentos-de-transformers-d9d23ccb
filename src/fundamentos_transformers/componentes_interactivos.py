"""Componentes interactivos reutilizables para los notebooks del curso."""

from __future__ import annotations

import io
import math
import warnings
from collections.abc import Sequence

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Pandas requires version .*", category=UserWarning)
    import pandas as pd

from .tokenizacion import codificar, construir_vocabulario, tokenizar_caracteres


def _tabla_df(filas: list[dict], columnas: list[tuple[str, str]]):
    nombres = [nombre for nombre, _ in columnas]
    claves = [clave for _, clave in columnas]
    df = pd.DataFrame(filas, columns=claves).rename(columns=dict(zip(claves, nombres)))
    return df.style.hide(axis="index").set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1f2a44"),
                    ("color", "white"),
                    ("text-align", "left"),
                    ("padding", "8px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border-bottom", "1px solid #d0d7de"),
                    ("padding", "8px"),
                    ("text-align", "left"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("font-size", "14px"),
                ],
            },
        ]
    )


def _panel_value(titulo: str, contenido_html: str) -> str:
    return f"""
        <div>
            <h4 style="margin: 0 0 8px 0;">{titulo}</h4>
            {contenido_html}
        </div>
        """


def _panel_html(titulo: str, contenido_html: str) -> widgets.HTML:
    return widgets.HTML(value=_panel_value(titulo, contenido_html))


def _fila_posicion(tokens: Sequence[str], t: int) -> dict[str, object]:
    return {
        "t": t,
        "contexto_visible": repr("".join(tokens[:t])),
        "token_objetivo": repr(tokens[t]),
        "futuro_prohibido": repr("".join(tokens[t + 1 : t + 8])),
    }


def _top_probabilidades(
    token_a_id: dict[str, int],
    id_a_token: dict[int, str],
    ids: Sequence[int],
    t: int,
    k: int = 5,
) -> list[dict[str, object]]:
    generador = torch.Generator().manual_seed(100 + int(t))
    logits = torch.randn(len(token_a_id), generator=generador)
    objetivo_id = ids[t]
    logits[objetivo_id] += 1.2
    probs = torch.softmax(logits, dim=0)
    topk = torch.topk(probs, k=k)

    filas_probs = []
    for prob, idx in zip(topk.values, topk.indices):
        token = id_a_token[int(idx)]
        filas_probs.append(
            {
                "token": repr(token),
                "id": int(idx),
                "probabilidad": f"{float(prob):.3f}",
                "es_objetivo": "s\u00ed" if int(idx) == objetivo_id else "no",
            }
        )
    return filas_probs


def _crear_figura_frontera(tokens: Sequence[str], t: int):
    n_visible = min(len(tokens), max(14, t + 6))
    pos = np.arange(n_visible)

    fig, ax = plt.subplots(figsize=(max(10, 0.62 * n_visible), 3.2))
    ax.axvspan(-0.5, t - 0.5, color="#dbeafe", alpha=0.55, zorder=0)
    ax.axvspan(t - 0.5, t + 0.5, color="#fee2e2", alpha=0.75, zorder=0)
    if t < n_visible - 1:
        ax.axvspan(t + 0.5, n_visible - 0.5, color="#f6f8fa", alpha=0.95, zorder=0)

    for p, tok in zip(pos, tokens[:n_visible]):
        if p < t:
            color, texto_color = "#255c99", "white"
        elif p == t:
            color, texto_color = "#9b2226", "white"
        else:
            color, texto_color = "#c9d1d9", "#24292f"
        ax.scatter([p], [0], s=440, color=color, edgecolor="#1f2328", linewidth=0.8, zorder=3)
        ax.text(
            p,
            0,
            tok if tok != " " else "sp",
            ha="center",
            va="center",
            color=texto_color,
            fontsize=9,
            zorder=4,
        )

    if t > 1:
        ax.annotate(
            "contexto visible",
            xy=((t - 1) / 2, 0.48),
            ha="center",
            color="#255c99",
            fontsize=10,
            fontweight="bold",
        )
    else:
        ax.annotate(
            "contexto visible",
            xy=(0.35, 0.48),
            ha="left",
            color="#255c99",
            fontsize=10,
            fontweight="bold",
        )
    ax.annotate(
        "token objetivo",
        xy=(t, -0.48),
        ha="center",
        color="#9b2226",
        fontsize=10,
        fontweight="bold",
    )
    if t < n_visible - 1:
        ax.annotate(
            "futuro no visible",
            xy=((t + 1 + n_visible - 1) / 2, 0.48),
            ha="center",
            color="#57606a",
            fontsize=10,
            fontweight="bold",
        )
    ax.annotate(
        "",
        xy=(t - 0.35, 0.25),
        xytext=(0.2, 0.25),
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#255c99"},
    )
    ax.axvline(t - 0.5, color="#9b2226", linestyle="--", linewidth=1)
    ax.set_xlim(-0.6, n_visible - 0.4)
    ax.set_ylim(-0.75, 0.82)
    ax.set(yticks=[], xlabel="posici\u00f3n", title=f"Frontera autoregresiva para t={t}")
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def crear_explorador_autoregresivo(
    texto: str = "transformers modelan contexto",
    posiciones_referencia: Sequence[int] | None = None,
    posicion_inicial: int = 8,
) -> widgets.VBox:
    """Construye el explorador interactivo del problema autoregresivo."""

    tokens = tokenizar_caracteres(texto)
    token_a_id, id_a_token = construir_vocabulario(tokens)
    ids = codificar(tokens, token_a_id)

    if posiciones_referencia is None:
        posiciones_referencia = [2, 4, 8, 12, 16, 20, 24]
    posiciones_referencia = [t for t in posiciones_referencia if 1 <= int(t) < len(tokens)]
    posicion_inicial = min(max(1, int(posicion_inicial)), len(tokens) - 1)

    vocabulario_df = pd.DataFrame(
        [
            {"token": repr(token), "id": token_id}
            for token, token_id in sorted(token_a_id.items(), key=lambda item: item[1])
        ]
    )
    filas_referencia = [_fila_posicion(tokens, int(t)) for t in posiciones_referencia]

    panel_vocabulario = _panel_html(
        "Vocabulario del ejemplo",
        vocabulario_df.style.hide(axis="index").to_html(),
    )
    panel_referencias = _panel_html(
        "Lectura por posiciones de referencia",
        _tabla_df(
            filas_referencia,
            [
                ("posici\u00f3n t", "t"),
                ("contexto visible", "contexto_visible"),
                ("token objetivo", "token_objetivo"),
                ("futuro no visible", "futuro_prohibido"),
            ],
        ).to_html(),
    )

    panel_vocabulario.layout = widgets.Layout(width="260px")
    panel_referencias.layout = widgets.Layout(width="calc(100% - 300px)", min_width="520px")
    resumen_inicial = widgets.HBox(
        [panel_vocabulario, panel_referencias],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="18px",
            width="100%",
        ),
    )

    selector_t = widgets.IntSlider(
        value=posicion_inicial,
        min=1,
        max=len(tokens) - 1,
        step=1,
        description="posici\u00f3n t",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=widgets.Layout(width="560px"),
    )
    panel_posicion = widgets.HTML()
    panel_distribucion = widgets.HTML()
    figura_imagen = widgets.Image(format="png")
    figura_imagen.layout = widgets.Layout(width="100%")
    panel_posicion.layout = widgets.Layout(width="52%", min_width="480px")
    panel_distribucion.layout = widgets.Layout(width="44%", min_width="360px")

    paneles_dinamicos = widgets.HBox(
        [panel_posicion, panel_distribucion],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="18px",
            width="100%",
        ),
    )

    def actualizar_contenido(t: int) -> None:
        objetivo_id = ids[t]
        tabla_posicion = _tabla_df(
            [_fila_posicion(tokens, t)],
            [
                ("posici\u00f3n t", "t"),
                ("contexto visible", "contexto_visible"),
                ("token objetivo", "token_objetivo"),
                ("futuro no visible", "futuro_prohibido"),
            ],
        ).to_html()
        token_html = (
            f"<p><strong>Token observado:</strong> {repr(tokens[t])} "
            f"&nbsp; <strong>id:</strong> {objetivo_id}</p>"
        )
        panel_posicion.value = _panel_value(
            f"Lectura para la posici\u00f3n t={t}", tabla_posicion + token_html
        )
        panel_distribucion.value = _panel_value(
            "Top 5 de una distribuci\u00f3n simulada sobre el vocabulario",
            _tabla_df(
                _top_probabilidades(token_a_id, id_a_token, ids, t),
                [
                    ("token", "token"),
                    ("id", "id"),
                    ("probabilidad", "probabilidad"),
                    ("\u00bfes el objetivo observado?", "es_objetivo"),
                ],
            ).to_html(),
        )

        fig = _crear_figura_frontera(tokens, t)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        figura_imagen.value = buffer.getvalue()

    def actualizar_explorador(change=None) -> None:
        actualizar_contenido(t=selector_t.value)

    selector_t.observe(actualizar_explorador, names="value")
    actualizar_contenido(posicion_inicial)

    explorador = widgets.VBox(
        [
            widgets.HTML("<h4>Explorador autoregresivo</h4>"),
            selector_t,
            paneles_dinamicos,
            figura_imagen,
        ],
        layout=widgets.Layout(
            gap="8px",
            width="100%",
        )
    )
    return widgets.VBox([resumen_inicial, explorador])


def crear_explorador_batches_autoregresivos(
    textos: dict[str, str] | None = None,
    texto_inicial: str | None = None,
    block_size_inicial: int = 8,
    start_inicial: int = 0,
) -> widgets.VBox:
    """Construye un explorador de tokenización, vocabulario y ventanas X/Y."""

    if textos is None:
        textos = {
            "atención": "atencion transforma tokens en contexto. atencion mezcla informacion.",
            "secuencias": "transformers modelan secuencias. cada posicion predice el siguiente token.",
            "mínimo": "texto pequeno para ver indices y batches.",
        }
    if not textos:
        raise ValueError("textos debe contener al menos un ejemplo")
    if texto_inicial is None or texto_inicial not in textos:
        texto_inicial = next(iter(textos))

    selector_texto = widgets.Dropdown(
        options=list(textos.keys()),
        value=texto_inicial,
        description="texto",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="360px"),
    )
    selector_block = widgets.IntSlider(
        value=max(2, int(block_size_inicial)),
        min=2,
        max=20,
        step=1,
        description="block_size",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=widgets.Layout(width="520px"),
    )
    selector_start = widgets.IntSlider(
        value=max(0, int(start_inicial)),
        min=0,
        max=1,
        step=1,
        description="start",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=widgets.Layout(width="520px"),
    )

    panel_resumen = widgets.HTML()
    panel_vocabulario = widgets.HTML()
    panel_alineacion = widgets.HTML()
    panel_visual = widgets.HTML()

    panel_resumen.layout = widgets.Layout(width="30%", min_width="280px")
    panel_vocabulario.layout = widgets.Layout(width="16%", min_width="180px")
    panel_alineacion.layout = widgets.Layout(width="50%", min_width="460px")

    paneles = widgets.HBox(
        [panel_resumen, panel_vocabulario, panel_alineacion],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="14px",
            width="100%",
        ),
    )

    def token_visible(token: str) -> str:
        return "<espacio>" if token == " " else token

    def actualizar_limites(*_) -> None:
        tokens = tokenizar_caracteres(textos[selector_texto.value])
        max_block = max(2, min(20, len(tokens) - 1))
        selector_block.max = max_block
        if selector_block.value > max_block:
            selector_block.value = max_block
        max_start = max(0, len(tokens) - int(selector_block.value) - 1)
        selector_start.max = max_start
        if selector_start.value > max_start:
            selector_start.value = max_start

    def actualizar_contenido(*_) -> None:
        actualizar_limites()
        texto = textos[selector_texto.value]
        tokens = tokenizar_caracteres(texto)
        token_a_id, id_a_token = construir_vocabulario(tokens)
        ids = codificar(tokens, token_a_id)
        block_size = int(selector_block.value)
        start = int(selector_start.value)
        x = ids[start : start + block_size]
        y = ids[start + 1 : start + block_size + 1]

        ventanas_posibles = max(0, len(ids) - block_size)
        resumen = pd.DataFrame(
            [
                {"objeto": "texto", "valor": selector_texto.value, "interpretación": "Corpus miniatura seleccionado."},
                {"objeto": "tokens", "valor": len(tokens), "interpretación": "Longitud de la secuencia tokenizada."},
                {"objeto": "vocabulario", "valor": len(token_a_id), "interpretación": "Símbolos distintos observados."},
                {"objeto": "block_size", "valor": block_size, "interpretación": "Longitud de la ventana X."},
                {"objeto": "start", "valor": start, "interpretación": "Posición inicial de la ventana en el corpus."},
                {"objeto": "ventanas posibles", "valor": ventanas_posibles, "interpretación": "Número de inicios válidos para X/Y."},
            ]
        )
        panel_resumen.value = _panel_value(
            "Resumen dinámico",
            resumen.style.hide(axis="index").to_html(),
        )

        vocabulario = pd.DataFrame(
            [
                {"token": token_visible(token), "id": token_id, "representación": repr(token)}
                for token, token_id in sorted(token_a_id.items(), key=lambda item: item[1])
            ]
        )
        panel_vocabulario.value = _panel_value(
            "Vocabulario",
            vocabulario.style.hide(axis="index").to_html(),
        )

        filas = []
        for col in range(block_size):
            pos_x = start + col
            pos_y = start + col + 1
            filas.append(
                {
                    "columna": col,
                    "posición X": pos_x,
                    "X token": token_visible(id_a_token[x[col]]),
                    "X id": x[col],
                    "posición Y": pos_y,
                    "Y token": token_visible(id_a_token[y[col]]),
                    "Y id": y[col],
                }
            )
        panel_alineacion.value = _panel_value(
            "Ventana alineada X/Y",
            _tabla_df(
                filas,
                [
                    ("col", "columna"),
                    ("pos X", "posición X"),
                    ("X token", "X token"),
                    ("X id", "X id"),
                    ("pos Y", "posición Y"),
                    ("Y token", "Y token"),
                    ("Y id", "Y id"),
                ],
            ).to_html(),
        )

        tokens_html = []
        for pos, token in enumerate(tokens):
            if start <= pos < start + block_size:
                bg, fg, borde = "#dbeafe", "#1f2a44", "#255c99"
            elif start + 1 <= pos < start + block_size + 1:
                bg, fg, borde = "#fee2e2", "#7f1d1d", "#b91c1c"
            else:
                bg, fg, borde = "#f6f8fa", "#57606a", "#d0d7de"
            tokens_html.append(
                f"""
                <span style="display:inline-block; min-width:34px; margin:3px; padding:5px 6px;
                             border:1px solid {borde}; border-radius:6px; background:{bg}; color:{fg};
                             text-align:center; font-family:monospace;">
                    <span style="font-size:11px; color:#57606a;">{pos}</span><br>{token_visible(token)}
                </span>
                """
            )
        leyenda = """
        <p style="margin: 8px 0 0 0;">
            <span style="background:#dbeafe; border:1px solid #255c99; padding:2px 6px;">X</span>
            contexto de entrada &nbsp;
            <span style="background:#fee2e2; border:1px solid #b91c1c; padding:2px 6px;">Y</span>
            objetivo desplazado. La superposición visual indica que una posición puede ser objetivo de una columna
            y entrada de la siguiente.
        </p>
        """
        panel_visual.value = _panel_value(
            "Secuencia codificada por posición",
            "<div style='line-height:1.45;'>" + "".join(tokens_html) + "</div>" + leyenda,
        )

    selector_texto.observe(actualizar_contenido, names="value")
    selector_block.observe(actualizar_contenido, names="value")
    selector_start.observe(actualizar_contenido, names="value")
    actualizar_contenido()

    controles = widgets.HBox(
        [selector_texto, selector_block, selector_start],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="center",
            gap="12px",
            width="100%",
        ),
    )

    return widgets.VBox(
        [
            widgets.HTML("<h4>Explorador interactivo de ventanas X/Y</h4>"),
            controles,
            paneles,
            panel_visual,
        ],
        layout=widgets.Layout(gap="10px", width="100%"),
    )


def crear_explorador_embeddings(
    textos: dict[str, str] | None = None,
    texto_inicial: str | None = None,
    dimension_inicial: int = 2,
) -> widgets.VBox:
    """Construye un explorador de lookup, parámetros y gradientes en embeddings."""

    if textos is None:
        textos = {
            "atención": "atencion transforma tokens",
            "secuencia": "secuencia con tokens repetidos",
            "mínimo": "abcaba",
        }
    if not textos:
        raise ValueError("textos debe contener al menos un ejemplo")
    if texto_inicial is None or texto_inicial not in textos:
        texto_inicial = next(iter(textos))

    dimensiones = [2, 4, 8, 16]
    if dimension_inicial not in dimensiones:
        dimension_inicial = 2

    selector_texto = widgets.Dropdown(
        options=list(textos.keys()),
        value=texto_inicial,
        description="texto",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="330px"),
    )
    selector_c = widgets.Dropdown(
        options=dimensiones,
        value=dimension_inicial,
        description="C",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="210px"),
    )
    selector_token = widgets.Dropdown(
        options=[],
        description="token",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="260px"),
    )
    selector_batch = widgets.Dropdown(
        options=[],
        description="batch",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="360px"),
    )

    panel_resumen = widgets.HTML()
    panel_vocabulario = widgets.HTML()
    panel_lookup = widgets.HTML()
    panel_gradientes = widgets.HTML()
    panel_nota_figura = widgets.HTML()
    panel_figura = widgets.Image(format="png")
    panel_figura.layout = widgets.Layout(width="620px", max_width="100%", align_self="center")

    panel_resumen.layout = widgets.Layout(width="23%", min_width="250px")
    panel_vocabulario.layout = widgets.Layout(width="15%", min_width="180px")
    panel_lookup.layout = widgets.Layout(width="29%", min_width="340px")
    panel_gradientes.layout = widgets.Layout(width="29%", min_width="360px")
    panel_nota_figura.layout = widgets.Layout(width="560px", max_width="100%")
    panel_figura.layout = widgets.Layout(width="560px", max_width="100%")

    paneles_tablas = widgets.HBox(
        [panel_resumen, panel_vocabulario, panel_lookup, panel_gradientes],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="14px",
            width="100%",
        ),
    )
    panel_visual = widgets.HBox(
        [panel_nota_figura, panel_figura],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            justify_content="center",
            align_items="flex-start",
            gap="14px",
            width="100%",
        ),
    )

    def token_visible(token: str) -> str:
        return "<espacio>" if token == " " else token

    def construir_estado():
        texto = textos[selector_texto.value]
        tokens = tokenizar_caracteres(texto)
        token_a_id, id_a_token = construir_vocabulario(tokens)
        ids = codificar(tokens, token_a_id)
        c = int(selector_c.value)
        generador = torch.Generator().manual_seed(37 + 11 * c + len(tokens))
        matriz = torch.randn(len(token_a_id), c, generator=generador)
        return texto, tokens, token_a_id, id_a_token, ids, c, matriz

    def actualizar_opciones(*_) -> None:
        _, tokens, token_a_id, id_a_token, ids, _, _ = construir_estado()
        opciones_token = [
            (f"{token_visible(token)} | id {idx}", idx)
            for token, idx in sorted(token_a_id.items(), key=lambda item: item[1])
        ]
        valor_anterior = selector_token.value
        selector_token.options = opciones_token
        ids_validos = set(token_a_id.values())
        selector_token.value = valor_anterior if valor_anterior in ids_validos else opciones_token[0][1]

        batch_opciones = []
        if len(ids) >= 4:
            batch_opciones.append(("primeros 4 tokens", tuple(ids[:4])))
            batch_opciones.append(("tokens 2 a 5", tuple(ids[2:6])))
        batch_opciones.append(("tokens repetidos frecuentes", tuple(ids[: min(6, len(ids))])))
        if len(ids) >= 8:
            batch_opciones.append(("ventana central", tuple(ids[len(ids)//2 - 2 : len(ids)//2 + 2])))
        selector_batch.options = batch_opciones
        if selector_batch.value not in [opcion[1] for opcion in batch_opciones]:
            selector_batch.value = batch_opciones[0][1]

    def crear_figura_embeddings(matriz: torch.Tensor, id_a_token: dict[int, str]) -> bytes:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        puntos = matriz[:, :2].numpy()
        ax.scatter(puntos[:, 0], puntos[:, 1], s=90, color="#0f766e", alpha=0.9)
        for idx, (x0, y0) in enumerate(puntos):
            ax.text(x0 + 0.03, y0 + 0.03, token_visible(id_a_token[idx]), fontsize=11)
        ax.axhline(0, color="#999", lw=0.8)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set(
            title="Visualización 2D de embeddings iniciales",
            xlabel="canal 1",
            ylabel="canal 2",
        )
        fig.subplots_adjust(wspace=0.35)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()

    def actualizar_contenido(*_) -> None:
        actualizar_opciones()
        texto, tokens, token_a_id, id_a_token, ids, c, matriz = construir_estado()
        token_id = int(selector_token.value)
        batch_ids = list(selector_batch.value)
        batch_tensor = torch.tensor([batch_ids], dtype=torch.long)
        salida = matriz[batch_tensor]
        parametros = len(token_a_id) * c

        resumen = pd.DataFrame(
            [
                {"objeto": "texto", "valor": selector_texto.value, "interpretación": "Corpus miniatura seleccionado."},
                {"objeto": "tokens", "valor": len(tokens), "interpretación": "Longitud de la secuencia tokenizada."},
                {"objeto": "vocabulario", "valor": len(token_a_id), "interpretación": "Filas disponibles en la tabla E."},
                {"objeto": "C", "valor": c, "interpretación": "Canales por vector de embedding."},
                {"objeto": "parámetros en E", "valor": parametros, "interpretación": "|V| x C."},
                {"objeto": "forma E[batch]", "valor": tuple(salida.shape), "interpretación": "Resultado del lookup sobre el batch."},
            ]
        )
        panel_resumen.value = _panel_value("Resumen dinámico", resumen.style.hide(axis="index").to_html())

        vocabulario = pd.DataFrame(
            [
                {"token": token_visible(token), "id": idx, "fila de E": idx}
                for token, idx in sorted(token_a_id.items(), key=lambda item: item[1])
            ]
        )
        panel_vocabulario.value = _panel_value("Vocabulario", vocabulario.style.hide(axis="index").to_html())

        fila = [round(float(valor), 4) for valor in matriz[token_id].numpy().tolist()]
        lookup = pd.DataFrame(
            [
                {
                    "token seleccionado": token_visible(id_a_token[token_id]),
                    "id": token_id,
                    "operación": f"E[{token_id}]",
                    "vector": fila,
                }
            ]
        )
        formas = pd.DataFrame(
            [
                {"objeto": "ids", "forma": (len(token_a_id),), "lectura": "índices del vocabulario"},
                {"objeto": "E", "forma": tuple(matriz.shape), "lectura": "tabla entrenable"},
                {"objeto": "batch", "forma": tuple(batch_tensor.shape), "lectura": "índices consultados"},
                {"objeto": "E[batch]", "forma": tuple(salida.shape), "lectura": "vectores resultantes"},
            ]
        )
        panel_lookup.value = _panel_value(
            "Lookup y formas",
            lookup.style.hide(axis="index").to_html() + formas.style.hide(axis="index").to_html(),
        )

        consultados = set(batch_ids)
        gradientes = pd.DataFrame(
            [
                {
                    "fila": idx,
                    "token": token_visible(id_a_token[idx]),
                    "aparece en batch": idx in consultados,
                    "recibiría gradiente": "sí" if idx in consultados else "no",
                    "veces consultada": batch_ids.count(idx),
                }
                for idx in range(len(token_a_id))
            ]
        )
        panel_gradientes.value = _panel_value(
            "Filas consultadas y actualización potencial",
            gradientes.style.hide(axis="index").to_html()
            + "<p>Si la pérdida depende de todos los vectores del batch, las filas consultadas son las candidatas a recibir señal de actualización.</p>",
        )

        if c == 2:
            panel_figura.value = crear_figura_embeddings(matriz, id_a_token)
            panel_figura.layout.display = None
            panel_nota_figura.value = ""
            panel_nota_figura.layout.display = "none"
        else:
            panel_figura.value = b""
            panel_figura.layout.display = "none"
            panel_nota_figura.value = _panel_value(
                "Visualización 2D",
                (
                    "<p>La gráfica de puntos se muestra solo cuando <strong>C=2</strong>. "
                    "Con dimensiones mayores, cada embedding vive en más canales y no puede "
                    "dibujarse directamente en un plano sin aplicar una proyección adicional.</p>"
                ),
            )
            panel_nota_figura.layout.display = None

    def manejar_texto_o_cambio_c(change=None) -> None:
        actualizar_opciones()
        actualizar_contenido()

    selector_texto.observe(manejar_texto_o_cambio_c, names="value")
    selector_c.observe(manejar_texto_o_cambio_c, names="value")
    selector_token.observe(actualizar_contenido, names="value")
    selector_batch.observe(actualizar_contenido, names="value")
    actualizar_opciones()
    actualizar_contenido()

    controles = widgets.HBox(
        [selector_texto, selector_c, selector_token, selector_batch],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="center",
            gap="12px",
            width="100%",
        ),
    )

    return widgets.VBox(
        [
            widgets.HTML("<h4>Explorador interactivo de embeddings</h4>"),
            controles,
            paneles_tablas,
            panel_visual,
        ],
        layout=widgets.Layout(gap="10px", width="100%"),
    )


def crear_explorador_posicion_orden(
    secuencias: dict[str, list[str]] | None = None,
    secuencia_inicial: str | None = None,
    dimension_inicial: int = 4,
) -> widgets.VBox:
    """Construye un explorador de embeddings de token, posición y suma."""

    if secuencias is None:
        secuencias = {
            "a b c a": ["a", "b", "c", "a"],
            "a c b a": ["a", "c", "b", "a"],
            "t o k e n": ["t", "o", "k", "e", "n"],
        }
    if not secuencias:
        raise ValueError("secuencias debe contener al menos un ejemplo")
    if secuencia_inicial is None or secuencia_inicial not in secuencias:
        secuencia_inicial = next(iter(secuencias))

    dimensiones = [2, 4, 8]
    if dimension_inicial not in dimensiones:
        dimension_inicial = 4

    selector_secuencia = widgets.Dropdown(
        options=list(secuencias.keys()),
        value=secuencia_inicial,
        description="secuencia",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="330px"),
    )
    selector_c = widgets.Dropdown(
        options=dimensiones,
        value=dimension_inicial,
        description="C",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="210px"),
    )
    selector_posicion = widgets.IntSlider(
        value=0,
        min=0,
        max=len(secuencias[secuencia_inicial]) - 1,
        step=1,
        description="posición t",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=widgets.Layout(width="480px"),
    )

    panel_resumen = widgets.HTML()
    panel_tabla = widgets.HTML()
    panel_posicion = widgets.HTML()
    panel_visual = widgets.HTML()

    panel_resumen.layout = widgets.Layout(width="30%", min_width="280px")
    panel_posicion.layout = widgets.Layout(width="28%", min_width="280px")
    panel_tabla.layout = widgets.Layout(width="38%", min_width="420px")

    paneles = widgets.HBox(
        [panel_resumen, panel_posicion, panel_tabla],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="14px",
            width="100%",
        ),
    )

    def actualizar_limites(*_) -> None:
        seq = secuencias[selector_secuencia.value]
        selector_posicion.max = len(seq) - 1
        if selector_posicion.value > selector_posicion.max:
            selector_posicion.value = selector_posicion.max

    def construir_estado():
        seq = secuencias[selector_secuencia.value]
        vocab = sorted(set(seq))
        token_a_id = {token: idx for idx, token in enumerate(vocab)}
        ids = [token_a_id[token] for token in seq]
        c = int(selector_c.value)
        generador = torch.Generator().manual_seed(101 + 13 * len(seq) + c)
        token_embedding = torch.randn(len(vocab), c, generator=generador)
        position_embedding = torch.randn(len(seq), c, generator=generador)
        token_vectors = token_embedding[torch.tensor(ids)]
        pos_vectors = position_embedding[torch.arange(len(seq))]
        h = token_vectors + pos_vectors
        return seq, token_a_id, ids, c, token_embedding, position_embedding, token_vectors, pos_vectors, h

    def actualizar_contenido(*_) -> None:
        actualizar_limites()
        seq, token_a_id, ids, c, token_embedding, position_embedding, token_vectors, pos_vectors, h = construir_estado()
        t = int(selector_posicion.value)

        def vector_redondeado(vector: torch.Tensor) -> list[float]:
            return [round(float(valor), 3) for valor in vector.numpy().tolist()]

        resumen = pd.DataFrame(
            [
                {"objeto": "secuencia", "valor": selector_secuencia.value, "interpretación": "Tokens en orden."},
                {"objeto": "longitud T", "valor": len(seq), "interpretación": "Número de posiciones."},
                {"objeto": "vocabulario", "valor": len(token_a_id), "interpretación": "Filas en E_tok."},
                {"objeto": "C", "valor": c, "interpretación": "Canales por vector."},
                {"objeto": "E_tok", "valor": tuple(token_embedding.shape), "interpretación": "Tabla de contenido."},
                {"objeto": "E_pos", "valor": tuple(position_embedding.shape), "interpretación": "Tabla de posiciones."},
            ]
        )
        panel_resumen.value = _panel_value("Resumen dinámico", resumen.style.hide(axis="index").to_html())

        filas = []
        for pos, token in enumerate(seq):
            filas.append(
                {
                    "t": pos,
                    "token": token,
                    "id token": ids[pos],
                    "misma fila de E_tok": f"E_tok[{ids[pos]}]",
                    "fila posicional": f"E_pos[{pos}]",
                    "norma token": f"{float(token_vectors[pos].norm()):.3f}",
                    "norma posición": f"{float(pos_vectors[pos].norm()):.3f}",
                    "norma suma": f"{float(h[pos].norm()):.3f}",
                }
            )
        panel_tabla.value = _panel_value(
            "Token, posición y suma por fila",
            _tabla_df(
                filas,
                [
                    ("t", "t"),
                    ("token", "token"),
                    ("id", "id token"),
                    ("lookup token", "misma fila de E_tok"),
                    ("lookup pos", "fila posicional"),
                    ("||tok||", "norma token"),
                    ("||pos||", "norma posición"),
                    ("||h||", "norma suma"),
                ],
            ).to_html(),
        )

        detalle = pd.DataFrame(
            [
                {"componente": "token", "objeto": f"E_tok[{ids[t]}]", "vector": vector_redondeado(token_vectors[t])},
                {"componente": "posición", "objeto": f"E_pos[{t}]", "vector": vector_redondeado(pos_vectors[t])},
                {"componente": "suma", "objeto": f"h_{t}^(0)", "vector": vector_redondeado(h[t])},
            ]
        )
        panel_posicion.value = _panel_value(
            f"Lectura de la posición t={t}",
            detalle.style.hide(axis="index").to_html()
            + "<p>El token indica contenido; la fila posicional indica ubicación. La suma conserva la forma C.</p>",
        )

        chips = []
        for pos, token in enumerate(seq):
            activo = pos == t
            bg = "#dbeafe" if activo else "#f6f8fa"
            borde = "#255c99" if activo else "#d0d7de"
            chips.append(
                f"""
                <span style="display:inline-block; min-width:64px; margin:4px; padding:7px 8px;
                             border:1px solid {borde}; border-radius:7px; background:{bg};
                             text-align:center; font-family:monospace;">
                    <span style="font-size:11px; color:#57606a;">t={pos}</span><br>
                    <strong>{token}</strong><br>
                    <span style="font-size:11px;">id={ids[pos]}</span>
                </span>
                """
            )
        panel_visual.value = _panel_value(
            "Secuencia con posiciones explícitas",
            "<div style='line-height:1.45;'>" + "".join(chips) + "</div>"
            + "<p><strong>Lectura:</strong> si el mismo token aparece en dos posiciones, consulta la misma fila de "
            + "<code>E_tok</code>, pero recibe una fila distinta de <code>E_pos</code>. Si cambia el token en la "
            + "misma posición, cambia la fila de contenido y se conserva la fila posicional.</p>",
        )

    def manejar_cambio_secuencia(change=None) -> None:
        actualizar_limites()
        actualizar_contenido()

    selector_secuencia.observe(manejar_cambio_secuencia, names="value")
    selector_c.observe(actualizar_contenido, names="value")
    selector_posicion.observe(actualizar_contenido, names="value")
    actualizar_contenido()

    controles = widgets.HBox(
        [selector_secuencia, selector_c, selector_posicion],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="center",
            gap="12px",
            width="100%",
        ),
    )

    return widgets.VBox(
        [
            widgets.HTML("<h4>Explorador interactivo de contenido y posición</h4>"),
            controles,
            paneles,
            panel_visual,
        ],
        layout=widgets.Layout(gap="10px", width="100%"),
    )


def crear_explorador_tensores_transformer(
    b_inicial: int = 2,
    t_inicial: int = 5,
    c_inicial: int = 12,
    h_inicial: int = 3,
) -> widgets.VBox:
    """Construye un explorador de ejes y formas para QKV y cabezas."""

    selector_b = widgets.IntSlider(
        value=int(b_inicial),
        min=1,
        max=8,
        step=1,
        description="B",
        continuous_update=False,
        style={"description_width": "60px"},
        layout=widgets.Layout(width="260px"),
    )
    selector_t = widgets.IntSlider(
        value=int(t_inicial),
        min=2,
        max=16,
        step=1,
        description="T",
        continuous_update=False,
        style={"description_width": "60px"},
        layout=widgets.Layout(width="260px"),
    )
    selector_c = widgets.IntSlider(
        value=int(c_inicial),
        min=4,
        max=32,
        step=1,
        description="C",
        continuous_update=False,
        style={"description_width": "60px"},
        layout=widgets.Layout(width="260px"),
    )
    selector_h = widgets.IntSlider(
        value=int(h_inicial),
        min=1,
        max=8,
        step=1,
        description="H",
        continuous_update=False,
        style={"description_width": "60px"},
        layout=widgets.Layout(width="260px"),
    )

    panel_resumen = widgets.HTML()
    panel_formas = widgets.HTML()
    panel_operaciones = widgets.HTML()
    panel_visual = widgets.HTML()

    panel_resumen.layout = widgets.Layout(width="28%", min_width="280px")
    panel_formas.layout = widgets.Layout(width="34%", min_width="360px")
    panel_operaciones.layout = widgets.Layout(width="34%", min_width="360px")

    paneles = widgets.HBox(
        [panel_resumen, panel_formas, panel_operaciones],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="14px",
            width="100%",
        ),
    )

    def actualizar_contenido(*_) -> None:
        b = int(selector_b.value)
        t = int(selector_t.value)
        c = int(selector_c.value)
        h = int(selector_h.value)
        divisible = c % h == 0
        d = c // h if divisible else None

        resumen = pd.DataFrame(
            [
                {"eje": "B", "valor": b, "significado": "ejemplos del batch"},
                {"eje": "T", "valor": t, "significado": "posiciones por secuencia"},
                {"eje": "C", "valor": c, "significado": "canales por posición"},
                {"eje": "H", "valor": h, "significado": "número de cabezas"},
                {"eje": "D", "valor": d if d is not None else "no definido", "significado": "canales por cabeza"},
            ]
        )
        panel_resumen.value = _panel_value(
            "Lectura de ejes",
            resumen.style.hide(axis="index").to_html(),
        )

        if divisible:
            formas = [
                {"objeto": "X", "forma": (b, t, c), "lectura": "entrada"},
                {"objeto": "QKV", "forma": (b, t, 3 * c), "lectura": "proyección conjunta"},
                {"objeto": "Q, K, V", "forma": (b, t, c), "lectura": "tres tensores separados"},
                {"objeto": "Q separado", "forma": (b, t, h, d), "lectura": "canales divididos por cabeza"},
                {"objeto": "Q por cabezas", "forma": (b, h, t, d), "lectura": "cabeza antes de tiempo"},
            ]
        else:
            formas = [
                {"objeto": "X", "forma": (b, t, c), "lectura": "entrada"},
                {"objeto": "QKV", "forma": (b, t, 3 * c), "lectura": "proyección conjunta"},
                {"objeto": "Q, K, V", "forma": (b, t, c), "lectura": "tres tensores separados"},
                {"objeto": "Q separado", "forma": "no definido", "lectura": "C no se divide en H partes iguales"},
                {"objeto": "Q por cabezas", "forma": "no definido", "lectura": "la arquitectura debe corregir C o H"},
            ]
        panel_formas.value = _panel_value(
            "Auditoría de formas",
            pd.DataFrame(formas).style.hide(axis="index").to_html(),
        )

        operaciones = pd.DataFrame(
            [
                {"paso": "proyección", "operación": "Linear(C, 3C)", "qué cambia": "canales C pasan a 3C"},
                {"paso": "separación", "operación": "chunk(3, dim=-1)", "qué cambia": "Q, K y V quedan con C canales"},
                {
                    "paso": "separar cabezas",
                    "operación": "view(B,T,H,D)",
                    "qué cambia": "C se separa en H · D" if divisible else "bloqueado porque C no es H · D",
                },
                {
                    "paso": "reordenamiento",
                    "operación": "transpose(1,2)",
                    "qué cambia": "pone H antes de T para operar por cabeza" if divisible else "no se ejecuta sin separación válida",
                },
            ]
        )
        panel_operaciones.value = _panel_value(
            "Operaciones",
            operaciones.style.hide(axis="index").to_html(),
        )

        estado = (
            "<span style='background:#dcfce7; border:1px solid #16a34a; padding:3px 8px; border-radius:6px;'>"
            "C = H · D válido</span>"
            if divisible
            else
            "<span style='background:#fee2e2; border:1px solid #b91c1c; padding:3px 8px; border-radius:6px;'>"
            "C no es divisible por H: cambie C o H</span>"
        )
        chips = [
            ("X", f"({b},{t},{c})", "#255c99"),
            ("QKV", f"({b},{t},{3*c})", "#0f766e"),
        ]
        if divisible:
            chips.extend(
                [
                    ("Q", f"({b},{t},{c})", "#ca8a04"),
                    ("separar", f"({b},{t},{h},{d})", "#d97706"),
                    ("cabezas", f"({b},{h},{t},{d})", "#7c3aed"),
                ]
            )
        else:
            chips.extend(
                [
                    ("Q", f"({b},{t},{c})", "#ca8a04"),
                    ("separar", "no válido", "#b91c1c"),
                    ("cabezas", "no válido", "#b91c1c"),
                ]
            )
        html_chips = []
        for nombre, forma, color in chips:
            html_chips.append(
                f"""
                <span style="display:inline-block; min-width:112px; margin:5px; padding:9px 10px;
                             border:1px solid {color}; border-radius:8px; background:#f8fafc;
                             text-align:center; font-family:monospace;">
                    <strong style="color:{color};">{nombre}</strong><br>{forma}
                </span>
                """
            )
        panel_visual.value = _panel_value(
            "Cadena de formas",
            "<p>" + estado + "</p><div>" + " → ".join(html_chips) + "</div>"
            + "<p>La validez de multi-head attention depende de poder leer C como H grupos de D canales. "
            + "Cuando la partición es válida, <code>transpose(1,2)</code> reordena "
            + "<code>(B,T,H,D)</code> como <code>(B,H,T,D)</code> para trabajar por cabeza.</p>",
        )

    for selector in [selector_b, selector_t, selector_c, selector_h]:
        selector.observe(actualizar_contenido, names="value")
    actualizar_contenido()

    controles = widgets.HBox(
        [selector_b, selector_t, selector_c, selector_h],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="center",
            gap="10px",
            width="100%",
        ),
    )

    return widgets.VBox(
        [
            widgets.HTML("<h4>Explorador interactivo de formas tensoriales</h4>"),
            controles,
            paneles,
            panel_visual,
        ],
        layout=widgets.Layout(gap="10px", width="100%"),
    )


def crear_explorador_mezcla_ponderada() -> widgets.VBox:
    """Construye un explorador de atención como mezcla ponderada de valores."""

    valores = np.array(
        [
            [0.05, 0.10],
            [1.00, 0.25],
            [0.25, 1.05],
            [1.25, 0.92],
        ],
        dtype=float,
    )
    etiquetas = ["v0", "v1", "v2", "v3"]

    sliders = [
        widgets.FloatSlider(
            value=valor,
            min=-2.0,
            max=2.0,
            step=0.1,
            description=f"logit {i}",
            continuous_update=False,
            readout_format=".1f",
            style={"description_width": "70px"},
            layout=widgets.Layout(width="280px"),
        )
        for i, valor in enumerate([0.2, 1.2, 0.4, -0.1])
    ]
    selector_modo = widgets.ToggleButtons(
        options=[
            ("softmax", "softmax"),
            ("uniforme", "uniforme"),
            ("un punto", "one_hot"),
        ],
        value="softmax",
        description="pesos",
        style={"description_width": "70px"},
    )

    panel_resumen = widgets.HTML()
    panel_valores = widgets.HTML()
    panel_pesos = widgets.HTML()
    panel_lectura = widgets.HTML()
    panel_figura = widgets.Image(format="png")
    panel_figura.layout = widgets.Layout(width="620px", max_width="100%", align_self="center")

    panel_resumen.layout = widgets.Layout(width="28%", min_width="280px")
    panel_valores.layout = widgets.Layout(width="24%", min_width="240px")
    panel_pesos.layout = widgets.Layout(width="42%", min_width="380px")
    panel_lectura.layout = widgets.Layout(width="100%")

    paneles = widgets.HBox(
        [panel_resumen, panel_valores, panel_pesos],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="14px",
            width="100%",
        ),
    )

    def calcular_pesos(logits: np.ndarray, modo: str) -> np.ndarray:
        if modo == "uniforme":
            return np.ones_like(logits) / len(logits)
        if modo == "one_hot":
            pesos = np.zeros_like(logits)
            pesos[int(np.argmax(logits))] = 1.0
            return pesos
        logits_estables = logits - logits.max()
        exp = np.exp(logits_estables)
        return exp / exp.sum()

    def crear_figura(pesos: np.ndarray, salida: np.ndarray) -> bytes:
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        ax.scatter(
            valores[:, 0],
            valores[:, 1],
            s=(pesos * 900) + 90,
            color="#255c99",
            alpha=0.78,
            edgecolor="#1f2328",
            linewidth=0.8,
            label="valores",
        )
        for i, (x0, y0) in enumerate(valores):
            ax.text(x0 + 0.035, y0 + 0.035, etiquetas[i], fontsize=11, color="#1f2a44")
            ax.plot([x0, salida[0]], [y0, salida[1]], color="#c9d1d9", lw=1.1, zorder=0)
        ax.scatter(
            [salida[0]],
            [salida[1]],
            marker="*",
            s=320,
            color="#b91c1c",
            edgecolor="#7f1d1d",
            linewidth=0.8,
            label="mezcla",
            zorder=4,
        )
        ax.set(
            title="Salida como promedio ponderado de valores",
            xlabel="canal 1",
            ylabel="canal 2",
            xlim=(-0.12, 1.48),
            ylim=(-0.08, 1.26),
        )
        ax.grid(True, alpha=0.22)
        ax.legend(loc="lower right")
        fig.subplots_adjust(wspace=0.35)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()

    def actualizar(*_) -> None:
        logits = np.array([slider.value for slider in sliders], dtype=float)
        pesos = calcular_pesos(logits, selector_modo.value)
        salida = pesos @ valores

        resumen = pd.DataFrame(
            [
                {"objeto": "valores", "forma": "(4, 2)", "lectura": "cuatro vectores con dos canales"},
                {"objeto": "logits", "forma": "(4,)", "lectura": "puntajes antes de normalizar"},
                {"objeto": "pesos", "forma": "(4,)", "lectura": f"distribución sobre valores; suma={pesos.sum():.3f}"},
                {"objeto": "salida", "forma": "(2,)", "lectura": "vector mezclado"},
            ]
        )
        panel_resumen.value = _panel_value("Auditoría tensorial", resumen.style.hide(axis="index").to_html())

        filas_valores = [
            {"valor": etiqueta, "canal 1": f"{x:.2f}", "canal 2": f"{y:.2f}"}
            for etiqueta, (x, y) in zip(etiquetas, valores)
        ]
        panel_valores.value = _panel_value(
            "Vectores disponibles",
            _tabla_df(
                filas_valores,
                [("valor", "valor"), ("canal 1", "canal 1"), ("canal 2", "canal 2")],
            ).to_html(),
        )

        filas_pesos = []
        for etiqueta, logit, peso in zip(etiquetas, logits, pesos):
            filas_pesos.append(
                {
                    "valor": etiqueta,
                    "logit": f"{logit:.2f}",
                    "peso": f"{peso:.3f}",
                    "aporte canal 1": f"{peso * valores[int(etiqueta[-1]), 0]:.3f}",
                    "aporte canal 2": f"{peso * valores[int(etiqueta[-1]), 1]:.3f}",
                }
            )
        filas_pesos.append(
            {
                "valor": "salida",
                "logit": "",
                "peso": f"{pesos.sum():.3f}",
                "aporte canal 1": f"{salida[0]:.3f}",
                "aporte canal 2": f"{salida[1]:.3f}",
            }
        )
        panel_pesos.value = _panel_value(
            "Pesos y aportes",
            _tabla_df(
                filas_pesos,
                [
                    ("valor", "valor"),
                    ("logit", "logit"),
                    ("peso", "peso"),
                    ("aporte c1", "aporte canal 1"),
                    ("aporte c2", "aporte canal 2"),
                ],
            ).to_html(),
        )
        lectura_modo = {
            "softmax": "Softmax convierte logits en una mezcla diferenciable: todos los valores con peso mayor que cero siguen aportando.",
            "uniforme": "Con pesos uniformes, la salida es el promedio simple de los valores disponibles.",
            "one_hot": "Un punto es un contraste didáctico: aproxima selección dura, pero no representa el caso normal de atención diferenciable.",
        }[selector_modo.value]
        panel_lectura.value = (
            "<div style='border-left:4px solid #1f2a44; padding:8px 12px; "
            "background:#f6f8fa; color:#1f2a44;'>"
            f"{lectura_modo} La estrella roja se mueve porque cambian los pesos, no porque cambien los valores."
            "</div>"
        )
        panel_figura.value = crear_figura(pesos, salida)

    for slider in sliders:
        slider.observe(actualizar, names="value")
    selector_modo.observe(actualizar, names="value")
    actualizar()

    controles = widgets.HBox(
        [selector_modo, *sliders],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="center",
            gap="10px",
            width="100%",
        ),
    )

    return widgets.VBox(
        [
            widgets.HTML("<h4>Explorador interactivo de mezcla ponderada</h4>"),
            controles,
            paneles,
            panel_lectura,
            panel_figura,
        ],
        layout=widgets.Layout(gap="10px", width="100%"),
    )


def crear_explorador_qkv_atencion() -> widgets.VBox:
    """Construye un explorador de una fila de atención con consultas, llaves y valores."""

    tokens = ["el", "modelo", "atiende", "contexto"]
    x = torch.tensor(
        [
            [1.0, 0.0, 0.3, 0.2],
            [0.2, 1.0, 0.0, 0.4],
            [0.3, 0.1, 1.0, 0.5],
            [0.5, 0.3, 0.2, 1.0],
        ],
        dtype=torch.float32,
    )
    torch.manual_seed(13)
    wq = torch.nn.Linear(4, 4, bias=False)
    wk = torch.nn.Linear(4, 4, bias=False)
    wv = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        q = wq(x)
        k = wk(x)
        v = wv(x)

    selector_t = widgets.IntSlider(
        value=1,
        min=0,
        max=len(tokens) - 1,
        step=1,
        description="posición t",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=widgets.Layout(width="420px"),
    )

    panel_roles = widgets.HTML()
    panel_scores = widgets.HTML()
    panel_mezcla = widgets.HTML()
    panel_lectura = widgets.HTML()
    panel_figura = widgets.Image(format="png")

    panel_roles.layout = widgets.Layout(width="32%", min_width="320px")
    panel_scores.layout = widgets.Layout(width="32%", min_width="320px")
    panel_mezcla.layout = widgets.Layout(width="32%", min_width="320px")
    panel_lectura.layout = widgets.Layout(width="100%")
    panel_figura.layout = widgets.Layout(width="620px", max_width="100%", align_self="center")

    paneles = widgets.HBox(
        [panel_roles, panel_scores, panel_mezcla],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="14px",
            width="100%",
        ),
    )

    def _vector_corto(vector: torch.Tensor) -> str:
        return "[" + ", ".join(f"{float(x):.2f}" for x in vector[:4]) + "]"

    def crear_figura(pesos: torch.Tensor, t: int) -> bytes:
        fig, ax = plt.subplots(figsize=(5.6, 2.9))
        matriz = pesos.unsqueeze(0).numpy()
        im = ax.imshow(matriz, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels([f"{j}: {tok}" for j, tok in enumerate(tokens)], rotation=0)
        ax.set_yticks([0])
        ax.set_yticklabels([f"consulta t={t}"])
        ax.set_title("Distribución de atención para una consulta")
        for j, peso in enumerate(pesos):
            ax.text(j, 0, f"{float(peso):.2f}", ha="center", va="center", color="#1f2a44")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()

    def actualizar(*_) -> None:
        t = int(selector_t.value)
        scores = (q[t] @ k.T) / math.sqrt(q.shape[-1])
        pesos = torch.softmax(scores, dim=-1)
        salida_v = pesos @ v
        salida_k = pesos @ k

        roles = pd.DataFrame(
            [
                {"objeto": "x_t", "forma": "(4,)", "lectura": "representación de entrada de la posición seleccionada"},
                {"objeto": "q_t", "forma": "(4,)", "lectura": "lo que la posición usa para comparar"},
                {"objeto": "K", "forma": "(4, 4)", "lectura": "llaves de todas las posiciones"},
                {"objeto": "V", "forma": "(4, 4)", "lectura": "contenidos; entran después de softmax"},
            ]
        )
        panel_roles.value = _panel_value(
            f"Roles para t={t} ({tokens[t]})",
            roles.style.hide(axis="index").to_html()
            + f"<p><strong>q_t:</strong> <code>{_vector_corto(q[t])}</code></p>",
        )

        filas_scores = []
        for j, token in enumerate(tokens):
            filas_scores.append(
                {
                    "posición j": j,
                    "token": token,
                    "puntaje": f"{float(scores[j]):.3f}",
                    "peso": f"{float(pesos[j]):.3f}",
                    "V en score": "no",
                    "k_j": _vector_corto(k[j]),
                }
            )
        panel_scores.value = _panel_value(
            "Comparación contra llaves",
            _tabla_df(
                filas_scores,
                [
                    ("j", "posición j"),
                    ("token", "token"),
                    ("puntaje", "puntaje"),
                    ("peso", "peso"),
                    ("V en score", "V en score"),
                    ("k_j", "k_j"),
                ],
            ).to_html(),
        )

        filas_mezcla = []
        for j, token in enumerate(tokens):
            filas_mezcla.append(
                {
                    "posición j": j,
                    "token": token,
                    "peso": f"{float(pesos[j]):.3f}",
                    "v_j": _vector_corto(v[j]),
                    "aporte primer canal": f"{float(pesos[j] * v[j, 0]):.3f}",
                }
            )
        filas_mezcla.append(
            {
                "posición j": "salida",
                "token": "",
                "peso": f"{float(pesos.sum()):.3f}",
                "v_j": _vector_corto(salida_v),
                "aporte primer canal": f"{float(salida_v[0]):.3f}",
            }
        )
        panel_mezcla.value = _panel_value(
            "Mezcla de valores",
            _tabla_df(
                filas_mezcla,
                [
                    ("j", "posición j"),
                    ("token", "token"),
                    ("peso", "peso"),
                    ("v_j / salida", "v_j"),
                    ("aporte c1", "aporte primer canal"),
                ],
            ).to_html()
            + "<p><strong>Control conceptual:</strong> mezclar llaves produciría "
            + f"<code>{_vector_corto(salida_k)}</code>, pero esa no es la salida correcta de atención.</p>",
        )
        panel_lectura.value = (
            "<div style='border-left:4px solid #1f2a44; padding:8px 12px; "
            "background:#f6f8fa; color:#1f2a44;'>"
            f"Para t={t}, Q y K producen los pesos. La fila suma "
            f"<strong>{float(pesos.sum()):.3f}</strong>. Después, esos pesos se aplican sobre V para transportar contenido."
            "</div>"
        )
        panel_figura.value = crear_figura(pesos, t)

    selector_t.observe(actualizar, names="value")
    actualizar()

    return widgets.VBox(
        [
            widgets.HTML("<h4>Explorador interactivo de Q, K y V</h4>"),
            selector_t,
            paneles,
            panel_lectura,
            panel_figura,
        ],
        layout=widgets.Layout(gap="10px", width="100%"),
    )


def crear_explorador_self_attention_matricial() -> widgets.VBox:
    """Construye un explorador para leer filas de una matriz de self-attention."""

    tokens = ["el", "modelo", "usa", "contexto", "local"]
    torch.manual_seed(23)
    x = torch.randn(1, len(tokens), 6)
    q_proj = torch.nn.Linear(6, 6, bias=False)
    k_proj = torch.nn.Linear(6, 6, bias=False)
    v_proj = torch.nn.Linear(6, 6, bias=False)
    with torch.no_grad():
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)
        puntajes = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        pesos = torch.softmax(puntajes, dim=-1)
        salida = pesos @ v

    selector_t = widgets.IntSlider(
        value=2,
        min=0,
        max=len(tokens) - 1,
        step=1,
        description="fila t",
        continuous_update=False,
        style={"description_width": "80px"},
        layout=widgets.Layout(width="380px"),
    )

    panel_resumen = widgets.HTML()
    panel_fila = widgets.HTML()
    panel_salida = widgets.HTML()
    panel_lectura = widgets.HTML()
    panel_figura = widgets.Image(format="png")

    panel_resumen.layout = widgets.Layout(width="28%", min_width="280px")
    panel_fila.layout = widgets.Layout(width="36%", min_width="360px")
    panel_salida.layout = widgets.Layout(width="30%", min_width="300px")
    panel_lectura.layout = widgets.Layout(width="100%")
    panel_figura.layout = widgets.Layout(width="620px", max_width="100%", align_self="center")

    paneles = widgets.HBox(
        [panel_resumen, panel_fila, panel_salida],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="14px",
            width="100%",
        ),
    )

    def _vector_corto(vector: torch.Tensor) -> str:
        return "[" + ", ".join(f"{float(x):.2f}" for x in vector[:4]) + ("..." if vector.numel() > 4 else "") + "]"

    def crear_figura(t: int) -> bytes:
        fig, ax = plt.subplots(figsize=(5.7, 4.2))
        matriz = pesos[0].numpy()
        im = ax.imshow(matriz, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels([f"{j}\\n{tok}" for j, tok in enumerate(tokens)])
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels([f"{i}\\n{tok}" for i, tok in enumerate(tokens)])
        ax.set_xlabel("posición consultada j")
        ax.set_ylabel("consulta t")
        ax.set_title("Matriz de pesos de autoatención")
        ax.axhline(t - 0.5, color="#b91c1c", lw=1.2)
        ax.axhline(t + 0.5, color="#b91c1c", lw=1.2)
        for row in range(len(tokens)):
            for col in range(len(tokens)):
                ax.text(col, row, f"{matriz[row, col]:.2f}", ha="center", va="center", fontsize=9, color="#1f2a44")
            ax.text(
                len(tokens) - 0.12,
                row + 0.36,
                f"suma={float(pesos[0, row].sum()):.1f}",
                ha="right",
                va="center",
                fontsize=8,
                color="#57606a",
            )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()

    def actualizar(*_) -> None:
        t = int(selector_t.value)
        resumen = pd.DataFrame(
            [
                {"objeto": "Q", "forma": tuple(q.shape), "lectura": "consultas de todas las posiciones"},
                {"objeto": "K", "forma": tuple(k.shape), "lectura": "llaves de todas las posiciones"},
                {"objeto": "V", "forma": tuple(v.shape), "lectura": "valores de todas las posiciones"},
                {"objeto": "S", "forma": tuple(puntajes.shape), "lectura": "puntajes t por j"},
                {"objeto": "A", "forma": tuple(pesos.shape), "lectura": "pesos normalizados por fila"},
                {"objeto": "Y", "forma": tuple(salida.shape), "lectura": "salidas contextuales"},
            ]
        )
        panel_resumen.value = _panel_value("Auditoría de formas", resumen.style.hide(axis="index").to_html())

        filas = []
        for j, token in enumerate(tokens):
            filas.append(
                {
                    "j": j,
                    "token": token,
                    "puntaje S[t,j]": f"{float(puntajes[0, t, j]):.3f}",
                    "peso A[t,j]": f"{float(pesos[0, t, j]):.3f}",
                    "primer canal V[j]": f"{float(v[0, j, 0]):.3f}",
                    "aporte c1": f"{float(pesos[0, t, j] * v[0, j, 0]):.3f}",
                }
            )
        panel_fila.value = _panel_value(
            f"Lectura de la fila t={t}",
            _tabla_df(
                filas,
                [
                    ("j", "j"),
                    ("token", "token"),
                    ("S[t,j]", "puntaje S[t,j]"),
                    ("A[t,j]", "peso A[t,j]"),
                    ("V[j] c1", "primer canal V[j]"),
                    ("aporte c1", "aporte c1"),
                ],
            ).to_html(),
        )

        salida_manual = pesos[0, t] @ v[0]
        salida_df = pd.DataFrame(
            [
                {"objeto": "suma fila A[t,:]", "valor": f"{float(pesos[0, t].sum()):.3f}", "lectura": "debe sumar uno"},
                {"objeto": "Y[t] manual", "valor": _vector_corto(salida_manual), "lectura": "A[t,:] por V"},
                {"objeto": "Y[t] tensor", "valor": _vector_corto(salida[0, t]), "lectura": "fila ya calculada por AV"},
                {
                    "objeto": "coincide",
                    "valor": str(bool(torch.allclose(salida_manual, salida[0, t], atol=1e-6))),
                    "lectura": "verificación de la forma matricial",
                },
            ]
        )
        panel_salida.value = _panel_value("Salida de la fila", salida_df.style.hide(axis="index").to_html())
        token_mayor = int(torch.argmax(pesos[0, t]))
        panel_lectura.value = (
            "<div style='border-left:4px solid #1f2a44; padding:8px 12px; "
            "background:#f6f8fa; color:#1f2a44;'>"
            f"La fila A[{t},:] es una distribución sobre posiciones j. "
            f"Su mayor peso está en j={token_mayor} ({tokens[token_mayor]}), "
            f"y la salida se reconstruye como Y[{t}] = A[{t},:] @ V."
            "</div>"
        )
        panel_figura.value = crear_figura(t)

    selector_t.observe(actualizar, names="value")
    actualizar()

    return widgets.VBox(
        [
            widgets.HTML("<h4>Explorador interactivo de autoatención matricial</h4>"),
            selector_t,
            paneles,
            panel_lectura,
            panel_figura,
        ],
        layout=widgets.Layout(gap="10px", width="100%"),
    )


def crear_explorador_atencion_causal() -> widgets.VBox:
    """Construye un explorador de máscara causal y bloqueo de posiciones futuras."""

    tokens = ["el", "modelo", "predice", "el", "siguiente", "token"]
    t_max = len(tokens)
    torch.manual_seed(31)
    x = torch.randn(1, t_max, 6)
    puntajes = x @ x.transpose(-2, -1) / math.sqrt(x.size(-1))
    mascara = torch.tril(torch.ones(t_max, t_max, dtype=torch.bool))
    pesos_libres = torch.softmax(puntajes, dim=-1)
    puntajes_causales = puntajes.masked_fill(~mascara.unsqueeze(0), float("-inf"))
    pesos_causales = torch.softmax(puntajes_causales, dim=-1)

    selector_t = widgets.IntSlider(
        value=3,
        min=0,
        max=t_max - 1,
        step=1,
        description="fila t",
        continuous_update=False,
        style={"description_width": "80px"},
        layout=widgets.Layout(width="380px"),
    )
    selector_modo = widgets.ToggleButtons(
        options=[("causal", "causal"), ("sin máscara", "libre")],
        value="causal",
        description="modo",
        style={"description_width": "70px"},
    )

    panel_resumen = widgets.HTML()
    panel_fila = widgets.HTML()
    panel_comparacion = widgets.HTML()
    panel_lectura = widgets.HTML()
    panel_figura = widgets.Image(format="png")

    panel_resumen.layout = widgets.Layout(width="24%", min_width="270px")
    panel_fila.layout = widgets.Layout(width="34%", min_width="390px")
    panel_comparacion.layout = widgets.Layout(width="22%", min_width="280px")
    panel_lectura.layout = widgets.Layout(width="16%", min_width="230px")
    panel_figura.layout = widgets.Layout(width="760px", max_width="100%", align_self="center")

    paneles = widgets.HBox(
        [panel_resumen, panel_fila, panel_comparacion, panel_lectura],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="14px",
            width="100%",
        ),
    )

    def crear_figura(t: int, modo: str) -> bytes:
        matriz = pesos_causales[0].numpy() if modo == "causal" else pesos_libres[0].numpy()
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
        axes[0].imshow(mascara.numpy(), cmap="Greys", vmin=0, vmax=1)
        axes[0].set_title("Máscara causal")
        axes[0].set_xlabel("posición consultada j")
        axes[0].set_ylabel("posición que consulta t")
        axes[0].axhline(t - 0.5, color="#b91c1c", lw=1.2)
        axes[0].axhline(t + 0.5, color="#b91c1c", lw=1.2)
        cmap = "Blues" if modo == "causal" else "YlOrRd"
        im = axes[1].imshow(matriz, cmap=cmap, vmin=0, vmax=1)
        axes[1].set_title("Pesos de atención")
        axes[1].set_xlabel("posición consultada j")
        axes[1].set_ylabel("posición que consulta t")
        axes[1].axhline(t - 0.5, color="#b91c1c", lw=1.2)
        axes[1].axhline(t + 0.5, color="#b91c1c", lw=1.2)
        for ax in axes:
            ax.set_xticks(range(t_max))
            ax.set_yticks(range(t_max))
        for row in range(t_max):
            for col in range(t_max):
                axes[1].text(col, row, f"{matriz[row, col]:.2f}", ha="center", va="center", fontsize=8, color="#1f2a44")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()

    def actualizar(*_) -> None:
        t = int(selector_t.value)
        modo = selector_modo.value
        pesos = pesos_causales if modo == "causal" else pesos_libres
        resumen = pd.DataFrame(
            [
                {"objeto": "máscara", "forma": tuple(mascara.shape), "lectura": "permitida si j <= t"},
                {"objeto": "puntajes", "forma": tuple(puntajes.shape), "lectura": "comparaciones antes de enmascarar"},
                {"objeto": "pesos sin máscara", "forma": tuple(pesos_libres.shape), "lectura": "puede mirar futuro"},
                {"objeto": "pesos causales", "forma": tuple(pesos_causales.shape), "lectura": "futuro bloqueado"},
            ]
        )
        panel_resumen.value = _panel_value("Auditoría", resumen.style.hide(axis="index").to_html())

        filas = []
        for j, token in enumerate(tokens):
            permitido = j <= t
            filas.append(
                {
                    "j": j,
                    "token": token,
                    "relación": "diagonal" if j == t else ("pasado" if j < t else "futuro"),
                    "permitido": "sí" if permitido else "no",
                    "peso causal": f"{float(pesos_causales[0, t, j]):.3f}",
                    "peso sin máscara": f"{float(pesos_libres[0, t, j]):.3f}",
                }
            )
        panel_fila.value = _panel_value(
            f"Lectura de la fila t={t} ({tokens[t]})",
            _tabla_df(
                filas,
                [
                    ("j", "j"),
                    ("token", "token"),
                    ("relación", "relación"),
                    ("permitido", "permitido"),
                    ("A causal", "peso causal"),
                    ("A libre", "peso sin máscara"),
                ],
            ).to_html(),
        )

        futuro_causal = float(pesos_causales[0, t, t + 1 :].sum()) if t + 1 < t_max else 0.0
        futuro_libre = float(pesos_libres[0, t, t + 1 :].sum()) if t + 1 < t_max else 0.0
        comparacion = pd.DataFrame(
            [
                {"medida": "masa futura causal", "valor": f"{futuro_causal:.3f}", "lectura": "debe ser cero"},
                {"medida": "masa futura sin máscara", "valor": f"{futuro_libre:.3f}", "lectura": "fuga de información"},
                {"medida": "suma fila causal", "valor": f"{float(pesos_causales[0, t].sum()):.3f}", "lectura": "softmax sobre posiciones permitidas"},
                {"medida": "suma fila libre", "valor": f"{float(pesos_libres[0, t].sum()):.3f}", "lectura": "softmax sobre toda la fila"},
            ]
        )
        panel_comparacion.value = _panel_value("Comparación", comparacion.style.hide(axis="index").to_html())
        bloqueadas = ", ".join(str(j) for j in range(t + 1, t_max)) or "ninguna"
        panel_lectura.value = _panel_value(
            "Lectura",
            f"""
            <p>En la fila <strong>t={t}</strong>, las columnas <strong>j &gt; t</strong>
            son futuro y quedan bloqueadas: {bloqueadas}.</p>
            <p>La diagonal <strong>j=t</strong> queda permitida porque la posición usa
            su propia representación para construir la salida de esa posición.</p>
            """,
        )
        panel_figura.value = crear_figura(t, modo)

    selector_t.observe(actualizar, names="value")
    selector_modo.observe(actualizar, names="value")
    actualizar()

    return widgets.VBox(
        [
            widgets.HTML("<h4>Explorador interactivo de atención causal</h4>"),
            widgets.HBox([selector_t, selector_modo], layout=widgets.Layout(gap="12px", flex_flow="row wrap")),
            paneles,
            panel_figura,
        ],
        layout=widgets.Layout(gap="10px", width="100%"),
    )


def crear_explorador_multi_cabeza() -> widgets.VBox:
    """Construye un explorador de división de canales y mapas por cabeza."""

    opciones_c = [8, 12, 16, 24]
    opciones_h = [1, 2, 3, 4, 6]
    t_max = 6
    b = 1

    selector_c = widgets.Dropdown(
        options=opciones_c,
        value=12,
        description="canales C",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="210px"),
    )
    selector_h = widgets.Dropdown(
        options=opciones_h,
        value=3,
        description="cabezas H",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="210px"),
    )
    selector_t = widgets.IntSlider(
        value=t_max,
        min=3,
        max=t_max,
        step=1,
        description="tiempo T",
        continuous_update=False,
        style={"description_width": "80px"},
        layout=widgets.Layout(width="260px"),
    )
    selector_cabeza = widgets.IntSlider(
        value=0,
        min=0,
        max=2,
        step=1,
        description="cabeza",
        continuous_update=False,
        style={"description_width": "70px"},
        layout=widgets.Layout(width="220px"),
    )
    selector_consulta = widgets.IntSlider(
        value=4,
        min=0,
        max=t_max - 1,
        step=1,
        description="consulta t",
        continuous_update=False,
        style={"description_width": "80px"},
        layout=widgets.Layout(width="240px"),
    )

    panel_resumen = widgets.HTML()
    panel_formas = widgets.HTML()
    panel_fila = widgets.HTML()
    panel_lectura = widgets.HTML()
    panel_figura = widgets.Image(format="png")

    for panel in (panel_resumen, panel_formas, panel_fila, panel_lectura):
        panel.layout = widgets.Layout(width="24%", min_width="260px")
    panel_figura.layout = widgets.Layout(width="780px", max_width="100%", align_self="center")

    paneles = widgets.HBox(
        [panel_resumen, panel_formas, panel_fila, panel_lectura],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="12px",
            width="100%",
        ),
    )

    def crear_figura(pesos: torch.Tensor, valido: bool, cabeza_activa: int, consulta_activa: int) -> bytes:
        h_actual = pesos.shape[1] if valido else 1
        columnas = max(1, h_actual)
        fig, axes = plt.subplots(1, columnas, figsize=(2.85 * columnas, 2.85), squeeze=False)
        if not valido:
            ax = axes[0, 0]
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "C no es divisible por H.\nNo se puede formar D=C/H.",
                ha="center",
                va="center",
                fontsize=11,
                color="#9b2226",
            )
        else:
            for i in range(columnas):
                ax = axes[0, i]
                matriz = pesos[0, i].detach().numpy()
                im = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=1)
                ax.set_title(f"cabeza {i}")
                ax.set_xlabel("posición consultada j")
                if i == 0:
                    ax.set_ylabel("posición que consulta t")
                ax.set_xticks(range(matriz.shape[1]))
                ax.set_yticks(range(matriz.shape[0]))
                if i == cabeza_activa:
                    ax.axhline(consulta_activa - 0.5, color="#0f766e", linewidth=1.6)
                    ax.axhline(consulta_activa + 0.5, color="#0f766e", linewidth=1.6)
                for row in range(matriz.shape[0]):
                    for col in range(matriz.shape[1]):
                        color = "white" if matriz[row, col] > 0.55 else "#1f2a44"
                        ax.text(col, row, f"{matriz[row, col]:.2f}", ha="center", va="center", fontsize=7, color=color)
            fig.suptitle("Cada cabeza produce su propio mapa T×T", y=1.04)
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        fig.subplots_adjust(wspace=0.35)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()

    def actualizar(*_) -> None:
        c = int(selector_c.value)
        h = int(selector_h.value)
        t = int(selector_t.value)
        valido = c % h == 0
        d = c // h if valido else None
        selector_consulta.max = t - 1
        if selector_consulta.value > selector_consulta.max:
            selector_consulta.value = selector_consulta.max
        selector_cabeza.max = max(h - 1, 0)
        if selector_cabeza.value > selector_cabeza.max:
            selector_cabeza.value = selector_cabeza.max
        cabeza_activa = int(selector_cabeza.value)
        consulta_activa = int(selector_consulta.value)

        resumen = pd.DataFrame(
            [
                {"símbolo": "B", "valor": b, "lectura": "secuencias por lote"},
                {"símbolo": "T", "valor": t, "lectura": "posiciones por secuencia"},
                {"símbolo": "C", "valor": c, "lectura": "canales totales"},
                {"símbolo": "H", "valor": h, "lectura": "número de cabezas"},
                {"símbolo": "D", "valor": d if d is not None else "no definido", "lectura": "canales por cabeza"},
            ]
        )
        panel_resumen.value = _panel_value("Parámetros", resumen.style.hide(axis="index").to_html())

        if not valido:
            formas = pd.DataFrame(
                [
                    {"objeto": "entrada X", "forma": (b, t, c), "lectura": "representación inicial"},
                    {"objeto": "división en cabezas", "forma": "no válida", "lectura": "C debe ser divisible por H"},
                ]
            )
            fila = pd.DataFrame(
                [
                    {"j": "-", "peso": "-", "lectura": "sin D entero no existe fila de atención por cabeza"},
                ]
            )
            lectura = pd.DataFrame(
                [
                    {"criterio": "divisibilidad", "resultado": "no", "interpretación": "no existe D entero para repartir canales"},
                    {"criterio": "acción", "resultado": "ajustar C o H", "interpretación": "la arquitectura debe conservar C = H·D"},
                ]
            )
            pesos = torch.zeros(1, max(1, h), t, t)
        else:
            torch.manual_seed(41 + c + h + t)
            x = torch.randn(b, t, c)
            qkv = torch.nn.Linear(c, 3 * c, bias=False)(x)
            q, k, v = qkv.chunk(3, dim=-1)
            qh = q.view(b, t, h, d).transpose(1, 2)
            kh = k.view(b, t, h, d).transpose(1, 2)
            vh = v.view(b, t, h, d).transpose(1, 2)
            puntajes = qh @ kh.transpose(-2, -1) / math.sqrt(d)
            mascara = torch.tril(torch.ones(t, t, dtype=torch.bool))
            puntajes = puntajes.masked_fill(~mascara.unsqueeze(0).unsqueeze(0), float("-inf"))
            pesos = torch.softmax(puntajes, dim=-1)
            salida_cabezas = pesos @ vh
            salida = salida_cabezas.transpose(1, 2).contiguous().view(b, t, c)
            pesos_fila = pesos[0, cabeza_activa, consulta_activa].detach()
            mascara_futuro = torch.arange(t) > consulta_activa
            permitidos = pesos_fila.masked_fill(mascara_futuro, -1)
            posicion_top = int(torch.argmax(permitidos))

            formas = pd.DataFrame(
                [
                    {"objeto": "entrada X", "forma": tuple(x.shape), "lectura": "B,T,C"},
                    {"objeto": "Q,K,V", "forma": tuple(q.shape), "lectura": "proyecciones con C canales"},
                    {"objeto": "Q por cabezas", "forma": tuple(qh.shape), "lectura": "B,H,T,D"},
                    {"objeto": "pesos A", "forma": tuple(pesos.shape), "lectura": "un mapa T×T por cabeza"},
                    {"objeto": "salida por cabezas", "forma": tuple(salida_cabezas.shape), "lectura": "mezcla en cada subespacio"},
                    {"objeto": "salida concatenada", "forma": tuple(salida.shape), "lectura": "regresa a B,T,C"},
                ]
            )
            fila = pd.DataFrame(
                [
                    {
                        "j": j,
                        "peso A[h,t,j]": f"{float(pesos_fila[j]):.3f}",
                        "lectura": "permitido" if j <= consulta_activa else "futuro bloqueado",
                    }
                    for j in range(t)
                ]
            )
            masa_futura = float(pesos.masked_fill(mascara.unsqueeze(0).unsqueeze(0), 0).sum())
            masa_futura_fila = float(pesos_fila[mascara_futuro].sum())
            lectura = pd.DataFrame(
                [
                    {"criterio": "divisibilidad", "resultado": "sí", "interpretación": f"C=H·D={h}·{d}"},
                    {"criterio": "posiciones por cabeza", "resultado": str(t), "interpretación": "cada cabeza ve todas las posiciones permitidas"},
                    {"criterio": "fila seleccionada", "resultado": f"h={cabeza_activa}, t={consulta_activa}", "interpretación": "una distribución sobre posiciones j"},
                    {"criterio": "suma de fila", "resultado": f"{float(pesos_fila.sum()):.3f}", "interpretación": "la fila se normaliza por softmax"},
                    {"criterio": "masa futura", "resultado": f"{masa_futura_fila:.3f}", "interpretación": "j>t queda bloqueado"},
                    {"criterio": "mayor peso permitido", "resultado": f"j={posicion_top}", "interpretación": "no implica rol semántico garantizado"},
                    {"criterio": "interfaz final", "resultado": str(tuple(salida.shape)), "interpretación": "la capa siguiente recibe B,T,C"},
                ]
            )

        panel_formas.value = _panel_value("Auditoría tensorial", formas.style.hide(axis="index").to_html())
        panel_fila.value = _panel_value("Fila de atención", fila.style.hide(axis="index").to_html())
        panel_lectura.value = _panel_value("Lectura", lectura.style.hide(axis="index").to_html())
        panel_figura.value = crear_figura(pesos, valido, cabeza_activa, consulta_activa)

    selector_c.observe(actualizar, names="value")
    selector_h.observe(actualizar, names="value")
    selector_t.observe(actualizar, names="value")
    selector_cabeza.observe(actualizar, names="value")
    selector_consulta.observe(actualizar, names="value")
    actualizar()

    return widgets.VBox(
        [
            widgets.HTML("<h4>Explorador interactivo de atención multi-cabeza</h4>"),
            widgets.HBox([selector_c, selector_h, selector_t], layout=widgets.Layout(gap="12px", flex_flow="row wrap")),
            widgets.HBox([selector_cabeza, selector_consulta], layout=widgets.Layout(gap="12px", flex_flow="row wrap")),
            paneles,
            panel_figura,
        ],
        layout=widgets.Layout(gap="10px", width="100%"),
    )


def crear_explorador_residuales_layernorm_mlp() -> widgets.VBox:
    """Construye un explorador de pre-norm, residual y MLP por posición."""

    selector_c = widgets.Dropdown(
        options=[4, 8, 16],
        value=8,
        description="canales C",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="210px"),
    )
    selector_escala = widgets.FloatSlider(
        value=4.0,
        min=0.5,
        max=8.0,
        step=0.5,
        description="escala X",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=widgets.Layout(width="300px"),
    )
    selector_residual = widgets.ToggleButtons(
        options=[("con residual", "con"), ("sin residual", "sin")],
        value="con",
        description="salida",
        style={"description_width": "70px"},
    )

    panel_formas = widgets.HTML()
    panel_estadisticas = widgets.HTML()
    panel_ejes = widgets.HTML()
    panel_parametros = widgets.HTML()
    panel_figura = widgets.Image(format="png")

    for panel in (panel_formas, panel_estadisticas, panel_ejes, panel_parametros):
        panel.layout = widgets.Layout(width="24%", min_width="260px")
    panel_figura.layout = widgets.Layout(width="760px", max_width="100%", align_self="center")

    paneles = widgets.HBox(
        [panel_formas, panel_estadisticas, panel_ejes, panel_parametros],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="12px",
            width="100%",
        ),
    )

    def crear_figura(x: torch.Tensor, x_norm: torch.Tensor, y: torch.Tensor) -> bytes:
        fig, axes = plt.subplots(1, 3, figsize=(8.7, 2.8))
        datos = [
            (x.flatten().detach().numpy(), "X antes de LN", "#4f7fd5"),
            (x_norm.flatten().detach().numpy(), "LN(X)", "#0f766e"),
            (y.flatten().detach().numpy(), "salida", "#5c6f82"),
        ]
        for ax, (valores, titulo, color) in zip(axes, datos):
            ax.hist(valores, bins=20, color=color, alpha=0.78, edgecolor="white")
            ax.set_title(titulo)
            ax.set_xlabel("valor")
            ax.set_ylabel("frecuencia")
        fig.suptitle("Estabilización de escala por posición", y=1.04)
        fig.subplots_adjust(wspace=0.35)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()

    def actualizar(*_) -> None:
        c = int(selector_c.value)
        escala = float(selector_escala.value)
        usar_residual = selector_residual.value == "con"
        b, t = 2, 4
        torch.manual_seed(73 + c)
        x = torch.randn(b, t, c) * escala + 3.0
        ln = torch.nn.LayerNorm(c)
        mlp = torch.nn.Sequential(
            torch.nn.Linear(c, 4 * c),
            torch.nn.GELU(),
            torch.nn.Linear(4 * c, c),
        )
        x_norm = ln(x)
        cambio = mlp(x_norm)
        y = x + cambio if usar_residual else cambio

        formas = pd.DataFrame(
            [
                {"objeto": "X", "forma": tuple(x.shape), "lectura": "entrada B,T,C"},
                {"objeto": "LN(X)", "forma": tuple(x_norm.shape), "lectura": "normaliza canales por posición"},
                {"objeto": "MLP(LN(X))", "forma": tuple(cambio.shape), "lectura": "transforma canales sin mezclar posiciones"},
                {
                    "objeto": "Y",
                    "forma": tuple(y.shape),
                    "lectura": "X + cambio" if usar_residual else "solo cambio MLP",
                },
            ]
        )
        panel_formas.value = _panel_value("Auditoría tensorial", formas.style.hide(axis="index").to_html())

        stats = pd.DataFrame(
            [
                {"medida": "media global X", "valor": f"{float(x.mean()):.3f}", "lectura": "escala original"},
                {"medida": "std global X", "valor": f"{float(x.std()):.3f}", "lectura": "dispersión original"},
                {"medida": "media posición LN", "valor": f"{float(x_norm.mean(dim=-1).abs().mean()):.3f}", "lectura": "cerca de cero por posición"},
                {"medida": "std posición LN", "valor": f"{float(x_norm.std(dim=-1, unbiased=False).mean()):.3f}", "lectura": "cerca de uno por posición"},
                {"medida": "norma cambio MLP", "valor": f"{float(cambio.norm()):.3f}", "lectura": "señal transformada"},
                {"medida": "norma X", "valor": f"{float(x.norm()):.3f}", "lectura": "señal preservada por residual"},
                {
                    "medida": "norma salida Y",
                    "valor": f"{float(y.norm()):.3f}",
                    "lectura": "con residual conserva ruta; sin residual reemplaza",
                },
            ]
        )
        panel_estadisticas.value = _panel_value("Estadísticas", stats.style.hide(axis="index").to_html())

        ejes = pd.DataFrame(
            [
                {"pieza": "LayerNorm", "opera sobre": "C", "lectura": "media y varianza por cada (b,t)"},
                {"pieza": "MLP", "opera sobre": "C", "lectura": "misma red aplicada a cada posición"},
                {"pieza": "Residual", "opera sobre": "B,T,C", "lectura": "exige formas idénticas para sumar"},
                {"pieza": "Atención", "opera sobre": "T", "lectura": "mezcla posiciones; vista en notebooks previos"},
            ]
        )
        panel_ejes.value = _panel_value("Qué eje opera", ejes.style.hide(axis="index").to_html())

        parametros = pd.DataFrame(
            [
                {"componente": "LayerNorm", "parámetros": 2 * c, "rol": "escala y sesgo por canal"},
                {"componente": "MLP C->4C", "parámetros": c * (4 * c) + 4 * c, "rol": "expansión por posición"},
                {"componente": "MLP 4C->C", "parámetros": (4 * c) * c + c, "rol": "retorno a C canales"},
                {"componente": "residual", "parámetros": 0, "rol": "suma la señal de entrada"},
            ]
        )
        panel_parametros.value = _panel_value("Parámetros", parametros.style.hide(axis="index").to_html())
        panel_figura.value = crear_figura(x, x_norm, y)

    selector_c.observe(actualizar, names="value")
    selector_escala.observe(actualizar, names="value")
    selector_residual.observe(actualizar, names="value")
    actualizar()

    return widgets.VBox(
        [
            widgets.HTML("<h4>Explorador interactivo de residuales, LayerNorm y MLP</h4>"),
            widgets.HBox(
                [selector_c, selector_escala, selector_residual],
                layout=widgets.Layout(gap="12px", flex_flow="row wrap"),
            ),
            paneles,
            panel_figura,
        ],
        layout=widgets.Layout(gap="10px", width="100%"),
    )


def crear_explorador_bloque_decoder() -> widgets.VBox:
    """Construye un explorador de formas, parámetros y atención de un bloque decoder."""

    selector_c = widgets.Dropdown(
        options=[8, 16, 32],
        value=16,
        description="canales C",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="210px"),
    )
    selector_h = widgets.Dropdown(
        options=[1, 2, 4, 8],
        value=4,
        description="cabezas H",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="210px"),
    )
    selector_t = widgets.IntSlider(
        value=6,
        min=3,
        max=8,
        step=1,
        description="tiempo T",
        continuous_update=False,
        style={"description_width": "80px"},
        layout=widgets.Layout(width="260px"),
    )

    panel_formas = widgets.HTML()
    panel_parametros = widgets.HTML()
    panel_ejes = widgets.HTML()
    panel_lectura = widgets.HTML()
    panel_figura = widgets.Image(format="png")

    for panel in (panel_formas, panel_parametros, panel_ejes, panel_lectura):
        panel.layout = widgets.Layout(width="24%", min_width="260px")
    panel_figura.layout = widgets.Layout(width="620px", max_width="100%", align_self="center")

    paneles = widgets.HBox(
        [panel_formas, panel_parametros, panel_ejes, panel_lectura],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="12px",
            width="100%",
        ),
    )

    def crear_figura(pesos: torch.Tensor | None, valido: bool) -> bytes:
        fig, ax = plt.subplots(figsize=(4.7, 3.8))
        if not valido or pesos is None:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "C no es divisible por H.\nEl bloque no puede separar cabezas.",
                ha="center",
                va="center",
                fontsize=11,
                color="#1f2a44",
            )
        else:
            matriz = pesos[0, 0].detach().numpy()
            im = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=1)
            ax.set_title("Atención causal interna\ndropout=0 para lectura")
            ax.set_xlabel("posición consultada j")
            ax.set_ylabel("posición que consulta t")
            ax.set_xticks(range(matriz.shape[1]))
            ax.set_yticks(range(matriz.shape[0]))
            for row in range(matriz.shape[0]):
                for col in range(matriz.shape[1]):
                    color = "white" if matriz[row, col] > 0.55 else "#1f2a44"
                    ax.text(col, row, f"{matriz[row, col]:.2f}", ha="center", va="center", fontsize=7, color=color)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()

    def actualizar(*_) -> None:
        from .modelo import TransformerBlock

        c = int(selector_c.value)
        h = int(selector_h.value)
        t = int(selector_t.value)
        b = 2
        valido = c % h == 0

        if not valido:
            formas = pd.DataFrame(
                [
                    {"paso": "X", "forma": (b, t, c), "lectura": "entrada B,T,C"},
                    {"paso": "bloque decoder", "forma": "no válido", "lectura": "C debe ser divisible por H"},
                ]
            )
            parametros = pd.DataFrame(
                [{"componente": "bloque", "parámetros": "no definido", "lectura": "ajustar C o H"}]
            )
            ejes = pd.DataFrame(
                [
                    {"pieza": "multi-head", "opera sobre": "C=H·D", "lectura": "bloqueado sin D entero"},
                    {"pieza": "acción", "opera sobre": "arquitectura", "lectura": "elegir C múltiplo de H"},
                ]
            )
            lectura = pd.DataFrame(
                [
                    {"criterio": "divisibilidad", "resultado": "no", "interpretación": "no existe D=C/H entero"},
                    {"criterio": "acción", "resultado": "corregir arquitectura", "interpretación": "elegir C múltiplo de H"},
                ]
            )
            pesos = None
        else:
            torch.manual_seed(97 + c + h + t)
            block = TransformerBlock(n_embd=c, n_head=h, block_size=8, dropout=0.0)
            x = torch.randn(b, t, c)
            x_ln1 = block.ln1(x)
            attn_out, pesos = block.attn(x_ln1, return_attention=True)
            u = x + attn_out
            u_ln2 = block.ln2(u)
            mlp_out = block.ffwd(u_ln2)
            y = u + mlp_out
            formas = pd.DataFrame(
                [
                    {"paso": "X", "forma": tuple(x.shape), "lectura": "entrada del bloque"},
                    {"paso": "LN1(X)", "forma": tuple(x_ln1.shape), "lectura": "normaliza antes de atención"},
                    {"paso": "MHA causal", "forma": tuple(attn_out.shape), "lectura": "mezcla posiciones permitidas"},
                    {"paso": "residual 1: U", "forma": tuple(u.shape), "lectura": "X + MHA(LN1(X))"},
                    {"paso": "LN2(U)", "forma": tuple(u_ln2.shape), "lectura": "normaliza antes de MLP"},
                    {"paso": "MLP", "forma": tuple(mlp_out.shape), "lectura": "transforma canales por posición"},
                    {"paso": "residual 2: Y", "forma": tuple(y.shape), "lectura": "U + MLP(LN2(U))"},
                    {"paso": "pesos A", "forma": tuple(pesos.shape), "lectura": "B,H,T,T"},
                ]
            )
            parametros = pd.DataFrame(
                [
                    {"componente": "ln1", "parámetros": sum(p.numel() for p in block.ln1.parameters()), "lectura": "pre-norm de atención"},
                    {"componente": "attn", "parámetros": sum(p.numel() for p in block.attn.parameters()), "lectura": "QKV, proyección y dropout"},
                    {"componente": "ln2", "parámetros": sum(p.numel() for p in block.ln2.parameters()), "lectura": "pre-norm de MLP"},
                    {"componente": "ffwd", "parámetros": sum(p.numel() for p in block.ffwd.parameters()), "lectura": "MLP por posición"},
                ]
            )
            ejes = pd.DataFrame(
                [
                    {"pieza": "atención", "opera sobre": "T", "lectura": "mezcla posiciones permitidas"},
                    {"pieza": "MLP", "opera sobre": "C", "lectura": "transforma canales por posición"},
                    {"pieza": "LayerNorm", "opera sobre": "C", "lectura": "normaliza cada posición"},
                    {"pieza": "residual", "opera sobre": "B,T,C", "lectura": "exige formas compatibles"},
                ]
            )
            mascara = torch.tril(torch.ones(t, t, dtype=torch.bool))
            masa_futura = float(pesos.masked_fill(mascara.unsqueeze(0).unsqueeze(0), 0).sum())
            lectura = pd.DataFrame(
                [
                    {"criterio": "D=C/H", "resultado": c // h, "interpretación": "canales por cabeza"},
                    {"criterio": "interfaz", "resultado": str(tuple(y.shape)), "interpretación": "se conserva B,T,C"},
                    {"criterio": "masa futura", "resultado": f"{masa_futura:.3f}", "interpretación": "debe ser cero por máscara causal"},
                    {"criterio": "filas de A", "resultado": f"{float(pesos.sum(dim=-1).mean()):.3f}", "interpretación": "suman 1 con dropout=0"},
                ]
            )

        panel_formas.value = _panel_value("Traza interna", formas.style.hide(axis="index").to_html())
        panel_parametros.value = _panel_value("Parámetros", parametros.style.hide(axis="index").to_html())
        panel_ejes.value = _panel_value("Qué eje opera", ejes.style.hide(axis="index").to_html())
        panel_lectura.value = _panel_value("Lectura", lectura.style.hide(axis="index").to_html())
        panel_figura.value = crear_figura(pesos, valido)

    selector_c.observe(actualizar, names="value")
    selector_h.observe(actualizar, names="value")
    selector_t.observe(actualizar, names="value")
    actualizar()

    return widgets.VBox(
        [
            widgets.HTML("<h4>Explorador interactivo del bloque Transformer decoder</h4>"),
            widgets.HBox([selector_c, selector_h, selector_t], layout=widgets.Layout(gap="12px", flex_flow="row wrap")),
            paneles,
            panel_figura,
        ],
        layout=widgets.Layout(gap="10px", width="100%"),
    )


def crear_explorador_mini_gpt() -> widgets.VBox:
    """Construye un explorador de formas y parámetros para un Mini-GPT."""

    selector_vocab = widgets.Dropdown(
        options=[16, 20, 32, 48],
        value=20,
        description="Vocab",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="170px"),
    )
    selector_t = widgets.Dropdown(
        options=[4, 8, 12, 16],
        value=8,
        description="block T",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="170px"),
    )
    selector_c = widgets.Dropdown(
        options=[12, 16, 24, 32],
        value=16,
        description="canales C",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="190px"),
    )
    selector_h = widgets.Dropdown(
        options=[1, 2, 3, 4, 8],
        value=4,
        description="heads H",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="170px"),
    )
    selector_l = widgets.Dropdown(
        options=[1, 2, 3, 4],
        value=2,
        description="capas L",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="170px"),
    )

    panel_formas = widgets.HTML()
    panel_parametros = widgets.HTML()
    panel_decisiones = widgets.HTML()
    panel_lectura = widgets.HTML()

    for panel in (panel_formas, panel_parametros, panel_decisiones, panel_lectura):
        panel.layout = widgets.Layout(width="24%", min_width="260px")

    paneles = widgets.HBox(
        [panel_formas, panel_parametros, panel_decisiones, panel_lectura],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="flex-start",
            gap="12px",
            width="100%",
        ),
    )

    def contar(modulo) -> int:
        return sum(p.numel() for p in modulo.parameters() if p.requires_grad)

    def actualizar(*_) -> None:
        from .modelo import MiniTransformerLM

        vocab = int(selector_vocab.value)
        block_size = int(selector_t.value)
        c = int(selector_c.value)
        h = int(selector_h.value)
        l = int(selector_l.value)
        b = 2
        t = min(6, block_size)
        valido = c % h == 0

        if not valido:
            formas = pd.DataFrame(
                [
                    {"objeto": "idx", "forma": (b, t), "lectura": "índices de entrada"},
                    {"objeto": "Mini-GPT", "forma": "no válida", "lectura": "C debe dividirse en H cabezas"},
                ]
            )
            parametros = pd.DataFrame(
                [{"componente": "modelo", "parámetros": "no definido", "lectura": "ajustar C o H"}]
            )
            decisiones = pd.DataFrame(
                [
                    {"decisión": "C = H·D", "valor": f"{c} = {h}·D", "efecto": "no existe D entero"},
                    {"decisión": "acción", "valor": "cambiar C o H", "efecto": "recuperar arquitectura válida"},
                ]
            )
            lectura = pd.DataFrame(
                [
                    {
                        "criterio": "divisibilidad",
                        "resultado": "no",
                        "interpretación": "las cabezas dividen canales, no tokens ni posiciones",
                    }
                ]
            )
        else:
            modelo = MiniTransformerLM(
                vocab_size=vocab,
                block_size=block_size,
                n_embd=c,
                n_head=h,
                n_layer=l,
                dropout=0.0,
            )
            total = contar(modelo)
            formas = pd.DataFrame(
                [
                    {"objeto": "idx", "forma": (b, t), "lectura": "ids de tokens"},
                    {"objeto": "E_tok(idx)", "forma": (b, t, c), "lectura": "contenido tokenizado"},
                    {"objeto": "E_pos[0:T]", "forma": (1, t, c), "lectura": "posición compartida sobre batch"},
                    {"objeto": "X inicial", "forma": (b, t, c), "lectura": "contenido más posición"},
                    {"objeto": f"{l} bloques", "forma": (b, t, c), "lectura": "contextualizan y conservan interfaz"},
                    {"objeto": "LN final", "forma": (b, t, c), "lectura": "normaliza antes de vocabulario"},
                    {"objeto": "lm_head", "forma": (b, t, vocab), "lectura": "logits por token candidato"},
                    {"objeto": "targets", "forma": (b, t), "lectura": "índices objetivo para la pérdida"},
                ]
            )
            parametros = pd.DataFrame(
                [
                    {"componente": "token_embedding", "parámetros": contar(modelo.token_embedding), "lectura": "Vocab·C"},
                    {"componente": "position_embedding", "parámetros": contar(modelo.position_embedding), "lectura": "block_size·C"},
                    {"componente": "blocks", "parámetros": contar(modelo.blocks), "lectura": "crece con L y C"},
                    {"componente": "ln_f", "parámetros": contar(modelo.ln_f), "lectura": "2C"},
                    {"componente": "lm_head", "parámetros": contar(modelo.lm_head), "lectura": "C -> Vocab"},
                ]
            )
            parametros["%"] = [f"{valor:.1f}" for valor in 100 * parametros["parámetros"] / total]
            decisiones = pd.DataFrame(
                [
                    {"decisión": "Vocab", "valor": vocab, "efecto": "tamaño de embeddings de token y lm_head"},
                    {"decisión": "block_size", "valor": block_size, "efecto": "tamaño de tabla posicional y contexto máximo"},
                    {"decisión": "C", "valor": c, "efecto": "anchura de casi todas las representaciones"},
                    {"decisión": "H", "valor": h, "efecto": f"D={c // h} canales por cabeza"},
                    {"decisión": "L", "valor": l, "efecto": "cantidad de bloques decoder apilados"},
                ]
            )
            lectura = pd.DataFrame(
                [
                    {"criterio": "interfaz interna", "resultado": "(B,T,C)", "interpretación": "los bloques conservan forma"},
                    {"criterio": "salida", "resultado": "(B,T,Vocab)", "interpretación": "lm_head cambia canales por vocabulario"},
                    {"criterio": "pérdida", "resultado": "(B·T,Vocab) vs (B·T,)", "interpretación": "cross-entropy compara logits con targets"},
                    {"criterio": "generación", "resultado": "posterior", "interpretación": "softmax y selección no entrenan por sí solas"},
                ]
            )

        panel_formas.value = _panel_value("Formas esperadas", formas.style.hide(axis="index").to_html())
        panel_parametros.value = _panel_value("Parámetros", parametros.style.hide(axis="index").to_html())
        panel_decisiones.value = _panel_value("Decisiones", decisiones.style.hide(axis="index").to_html())
        panel_lectura.value = _panel_value("Lectura", lectura.style.hide(axis="index").to_html())

    for selector in (selector_vocab, selector_t, selector_c, selector_h, selector_l):
        selector.observe(actualizar, names="value")
    actualizar()

    return widgets.VBox(
        [
            widgets.HTML("<h4>Explorador interactivo de arquitectura Mini-GPT</h4>"),
            widgets.HBox(
                [selector_vocab, selector_t, selector_c, selector_h, selector_l],
                layout=widgets.Layout(gap="12px", flex_flow="row wrap"),
            ),
            paneles,
        ],
        layout=widgets.Layout(gap="10px", width="100%"),
    )
