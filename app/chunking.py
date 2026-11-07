"""Segmentación en tokens y oraciones con offsets del texto original."""
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import hashlib
import re
from typing import Any


@dataclass(frozen=True)
class Span:
    start: int
    end: int


def _encode(text, tok):
    return tok(text, add_special_tokens=False, return_offsets_mapping=True,
               return_attention_mask=False, truncation=False, verbose=False)


def _windows(text, offsets, tok, size, overlap):
    if not offsets:
        return []
    spans = []
    first = 0
    while first < len(offsets):
        last = min(first + size, len(offsets))
        start = offsets[first][0] if first else len(text) - len(text.lstrip())
        while last > first:
            end = offsets[last][0] if last < len(offsets) else len(text.rstrip())
            while end > start and text[end - 1].isspace():
                end -= 1
            # El prefijo de subpalabra puede cambiar al tokenizar un recorte.
            if len(_encode(text[start:end], tok)["input_ids"]) <= size:
                break
            last -= 1
        if last == first:
            raise ValueError("El tamaño no permite representar un carácter completo")
        if end > start:
            spans.append(Span(start, end))
        if last == len(offsets):
            break
        first = max(first + 1, last - overlap)
    return spans


def chunk_fixed_tokens(text, tok, size=256, overlap=32) -> list[Span]:
    """Crea ventanas solapadas sin reconstruir ni normalizar el texto."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Se requiere size > 0 y 0 <= overlap < size")
    if not text.strip():
        return []
    return _windows(text, _encode(text, tok)["offset_mapping"], tok, size, overlap)


_URL = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_BOUNDARY = re.compile(r'''[.!?…]+["'»”’)\]]*(?=\s|$)|\n[ \t\r]*\n+''')


def _sentences(text):
    urls = [(match.start(), match.end()) for match in _URL.finditer(text)]
    url_starts = [start for start, _ in urls]
    start = 0
    for match in _BOUNDARY.finditer(text):
        index = bisect_right(url_starts, match.start()) - 1
        if index >= 0 and match.start() < urls[index][1] and match.end() < urls[index][1]:
            continue
        end = match.end()
        fragment = text[start:end]
        left = start + len(fragment) - len(fragment.lstrip())
        right = start + len(fragment.rstrip())
        if left < right:
            yield Span(left, right)
        start = end
    fragment = text[start:]
    left = start + len(fragment) - len(fragment.lstrip())
    right = start + len(fragment.rstrip())
    if left < right:
        yield Span(left, right)


def chunk_sentences(text, tok, max_tokens=256) -> list[Span]:
    """Empaqueta oraciones; las demasiado largas usan ventanas de tokens."""
    if max_tokens <= 0:
        raise ValueError("max_tokens debe ser positivo")
    if not text.strip():
        return []
    result = []
    pending = None
    for sentence in _sentences(text):
        fragment = text[sentence.start:sentence.end]
        offsets = _encode(fragment, tok)["offset_mapping"]
        if len(offsets) > max_tokens:
            if pending is not None:
                result.append(pending)
                pending = None
            for span in _windows(fragment, offsets, tok, max_tokens, min(32, max_tokens - 1)):
                result.append(Span(sentence.start + span.start, sentence.start + span.end))
        elif pending is None:
            pending = sentence
        else:
            candidate = text[pending.start:sentence.end]
            if len(_encode(candidate, tok)["input_ids"]) <= max_tokens:
                pending = Span(pending.start, sentence.end)
            else:
                result.append(pending)
                pending = sentence
    if pending is not None:
        result.append(pending)
    return result


CHUNKERS: dict[str, Callable[[str, Any], list[Span]]] = {
    "tok256": partial(chunk_fixed_tokens, size=256, overlap=32),
    "sent256": chunk_sentences,
    "tok128": partial(chunk_fixed_tokens, size=128, overlap=16),
}


def chunk_id(document_id, chunker, start, end) -> str:
    """Identifica de forma estable un fragmento y su segmentación."""
    return hashlib.sha256(f"{document_id}|{chunker}|{start}|{end}".encode("utf-8")).hexdigest()[:32]
