"""Tablas auxiliares para el proyecto final."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class ResultadoProyectoFinal:
    evidencias: pd.DataFrame
    rubrica: pd.DataFrame
    figura_rubrica: plt.Figure
    checklist: pd.DataFrame
    preguntas_defensa: pd.DataFrame
    errores: pd.DataFrame
    autoevaluacion: pd.DataFrame


def preparar_material_proyecto_final() -> ResultadoProyectoFinal:
    evidencias = pd.DataFrame(
        [
            {"componente": "Datos y tokenizacion", "evidencia minima": "corpus, vocabulario, codificacion y ejemplos x/y decodificados", "pregunta que responde": "¿Que datos alimentan realmente al modelo?", "artefacto esperado": "tabla de vocabulario y batch"},
            {"componente": "Arquitectura", "evidencia minima": "block_size, C, heads, layers, parametros y formas internas", "pregunta que responde": "¿Que modelo se construyo?", "artefacto esperado": "tabla de configuracion y formas"},
            {"componente": "Atencion y mascara causal", "evidencia minima": "mapa de atencion y verificacion de no mirar al futuro", "pregunta que responde": "¿Como se restringe el contexto autoregresivo?", "artefacto esperado": "heatmap e interpretacion"},
            {"componente": "Entrenamiento", "evidencia minima": "curvas train/valid, loss y perplexity aproximada", "pregunta que responde": "¿El modelo se ajusta al corpus y como se valida?", "artefacto esperado": "curvas y tabla de metricas"},
            {"componente": "Generacion", "evidencia minima": "comparacion de temperature, top-k y top-p con prompt fijo", "pregunta que responde": "¿Como cambia la salida al cambiar el muestreo?", "artefacto esperado": "tabla de generaciones"},
            {"componente": "Limites", "evidencia minima": "errores observables, causa probable y precaucion interpretativa", "pregunta que responde": "¿Que no demuestra el modelo?", "artefacto esperado": "tabla de diagnostico"},
        ]
    )
    rubrica = pd.DataFrame(
        [
            {"criterio": "Reproducibilidad", "insuficiente": "faltan datos, semillas o configuracion", "aceptable": "ejecuta con configuracion documentada", "sobresaliente": "incluye rutas, semillas, versiones y decisiones justificadas"},
            {"criterio": "Lectura tensorial", "insuficiente": "menciona formas sin interpretarlas", "aceptable": "explica formas principales", "sobresaliente": "conecta cada forma con operacion, eje y posible error conceptual"},
            {"criterio": "Entrenamiento", "insuficiente": "solo reporta perdida final", "aceptable": "incluye curvas train/valid", "sobresaliente": "interpreta ajuste, sobreajuste, perplexity y limites del corpus"},
            {"criterio": "Generacion", "insuficiente": "muestra una salida sin configuracion", "aceptable": "compara estrategias con prompt fijo", "sobresaliente": "relaciona distribucion, muestreo, errores y limites"},
            {"criterio": "Analisis critico", "insuficiente": "sobreinterpreta resultados", "aceptable": "reconoce limites principales", "sobresaliente": "separa sintoma, evidencia, causa probable y precaucion"},
        ]
    )
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    criterios = rubrica["criterio"].tolist()
    ax.barh(criterios, [3] * len(criterios), color="#1f77b4")
    ax.set_xlim(0, 3.4)
    ax.set_xticks([1, 2, 3], ["insuf.", "aceptable", "sobresaliente"])
    ax.set_title("Rubrica: meta de referencia por criterio")
    ax.invert_yaxis()
    fig.tight_layout()

    checklist = pd.DataFrame(
        [
            {"bloque": "Ejecucion", "verificacion": "el notebook corre de arriba abajo", "evidencia": "sin errores ni celdas omitidas"},
            {"bloque": "Datos", "verificacion": "corpus y tokenizacion documentados", "evidencia": "tabla de vocabulario y ejemplos x/y"},
            {"bloque": "Modelo", "verificacion": "configuracion completa", "evidencia": "block_size, C, heads, layers y parametros"},
            {"bloque": "Metricas", "verificacion": "train y valid separadas", "evidencia": "curvas y tabla de loss/perplexity"},
            {"bloque": "Generacion", "verificacion": "prompt fijo y configuraciones claras", "evidencia": "tabla de salidas comparadas"},
            {"bloque": "Limites", "verificacion": "diagnostico de errores", "evidencia": "sintoma, evidencia, causa probable y cautela"},
        ]
    )
    preguntas_defensa = pd.DataFrame(
        [
            {"categoria": "Arquitectura", "pregunta": "¿Que parte del modelo mezcla posiciones y que parte refina canales?", "evidencia aceptable": "diagrama o tabla del bloque decoder con atencion y MLP"},
            {"categoria": "Q/K/V", "pregunta": "¿Donde aparecen Q, K y V en la implementacion y que forma tienen?", "evidencia aceptable": "auditoria tensorial de proyeccion y cabezas"},
            {"categoria": "Causalidad", "pregunta": "¿Por que la mascara causal evita fuga de futuro?", "evidencia aceptable": "matriz triangular o mapa de atencion interpretado"},
            {"categoria": "Logits", "pregunta": "¿Que significan los logits $(B,T,V)$ antes de muestrear?", "evidencia aceptable": "tabla de top tokens y softmax para una posicion"},
            {"categoria": "Limites", "pregunta": "¿Que limitacion concreta observaste en una muestra generada?", "evidencia aceptable": "ejemplo generado con diagnostico de error"},
        ]
    )
    errores = pd.DataFrame(
        [
            {"error frecuente": "Decir que el modelo comprende porque genera texto legible", "correccion esperada": "separar fluidez local de comprension y reportar limites"},
            {"error frecuente": "Comparar estrategias de muestreo cambiando tambien el prompt", "correccion esperada": "mantener prompt fijo para aislar el efecto del muestreo"},
            {"error frecuente": "Mostrar atencion como explicacion completa", "correccion esperada": "presentarla como evidencia parcial y contextual"},
            {"error frecuente": "Reportar solo perdida de entrenamiento", "correccion esperada": "incluir validacion y discutir sobreajuste"},
            {"error frecuente": "No documentar formas tensoriales", "correccion esperada": "auditar $(B,T)$, $(B,T,C)$, $(B,H,T,T)$ y $(B,T,V)$"},
        ]
    )
    autoevaluacion = pd.DataFrame(
        [
            {"pregunta": "¿Puedo reconstruir el experimento desde cero?", "criterio de evidencia": "si, con rutas, semillas y configuracion"},
            {"pregunta": "¿Cada afirmacion tiene evidencia?", "criterio de evidencia": "si, tabla, curva, mapa o ejemplo"},
            {"pregunta": "¿Distingo resultado de interpretacion?", "criterio de evidencia": "si, separo observacion y conclusion"},
            {"pregunta": "¿Declaro limites?", "criterio de evidencia": "si, no sobreinterpreto el mini modelo"},
        ]
    )
    return ResultadoProyectoFinal(evidencias, rubrica, fig, checklist, preguntas_defensa, errores, autoevaluacion)
