"""Configuración visual compartida."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Any

import matplotlib.pyplot as plt
from IPython.display import HTML, display


def configurar_matplotlib() -> None:
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.22
    plt.rcParams["image.cmap"] = "viridis"


def _tabla_a_html(tabla: Any) -> str:
    if hasattr(tabla, "hide") and hasattr(tabla, "to_html"):
        return tabla.to_html()
    if hasattr(tabla, "to_html"):
        try:
            return tabla.to_html(index=False, border=0)
        except TypeError:
            return tabla.to_html()
    return f"<pre>{escape(str(tabla))}</pre>"


def mostrar_tablas_en_columnas(
    tablas: Sequence[tuple[str, Any]],
    *,
    min_width: str = "280px",
) -> None:
    """Muestra tablas pequeñas en columnas para ahorrar espacio vertical."""

    paneles = []
    for titulo, tabla in tablas:
        paneles.append(
            f"""
            <section class="ft-table-panel">
                <h4>{escape(titulo)}</h4>
                <div class="ft-table-scroll">{_tabla_a_html(tabla)}</div>
            </section>
            """
        )

    html = f"""
    <style>
    .ft-table-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax({min_width}, 1fr));
        gap: 16px;
        align-items: start;
        width: 100%;
    }}
    .ft-table-panel h4 {{
        margin: 0 0 8px 0;
        color: #1f2a44;
        font-size: 15px;
    }}
    .ft-table-scroll {{
        overflow-x: auto;
    }}
    .ft-table-panel table {{
        border-collapse: collapse;
        width: 100%;
        font-size: 14px;
    }}
    .ft-table-panel th {{
        background-color: #1f2a44;
        color: white;
        text-align: left;
        padding: 8px;
        white-space: nowrap;
    }}
    .ft-table-panel td {{
        border-bottom: 1px solid #d0d7de;
        padding: 8px;
        text-align: left;
        vertical-align: top;
    }}
    </style>
    <div class="ft-table-grid">
        {''.join(paneles)}
    </div>
    """
    display(HTML(html))
