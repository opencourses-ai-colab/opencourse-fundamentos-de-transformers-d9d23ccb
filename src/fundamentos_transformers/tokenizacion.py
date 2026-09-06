"""Tokenización mínima para notebooks pedagógicos."""

from __future__ import annotations


def tokenizar_caracteres(texto: str) -> list[str]:
    return list(texto)


def tokenizar_palabras_basico(texto: str) -> list[str]:
    return texto.lower().replace(".", " .").replace(",", " ,").split()


def construir_vocabulario(tokens: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    vocab = sorted(set(tokens))
    token_a_id = {token: i for i, token in enumerate(vocab)}
    id_a_token = {i: token for token, i in token_a_id.items()}
    return token_a_id, id_a_token


def codificar(tokens: list[str], token_a_id: dict[str, int]) -> list[int]:
    return [token_a_id[token] for token in tokens]


def decodificar(ids: list[int], id_a_token: dict[int, str]) -> str:
    return "".join(id_a_token[int(i)] for i in ids)
