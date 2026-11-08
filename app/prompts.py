"""Plantillas versionadas y contexto limitado a fragmentos completos."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re


ABSTENTION = "No hay evidencia suficiente en los documentos recuperados."
_ROOT = Path(__file__).resolve().parents[1]
_HANDLE = re.compile(r"(?<![\w@])@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+")


def load_prompt(version: str = "rag_v1") -> tuple[str, str]:
    """Lee la plantilla desde la raíz y calcula su huella sobre los bytes."""
    if not re.fullmatch(r"[\w.-]+", version) or version in (".", ".."):
        raise ValueError("Versión de plantilla no válida.")
    raw = (_ROOT / "prompts" / f"{version}.txt").read_bytes()
    return raw.decode("utf-8"), f"{version}@{hashlib.sha256(raw).hexdigest()[:12]}"


def _date(created_ts) -> str:
    if created_ts is None or created_ts == -1:
        return "s/f"
    try:
        return datetime.fromtimestamp(created_ts, timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OverflowError, OSError):
        return "s/f"


def build_context(chunks: list[dict], budget_chars: int) -> tuple[str, list[dict]]:
    """Omite fragmentos que no caben y numera solo la evidencia incluida."""
    blocks, evidence = [], []
    size = 0
    ranked = sorted(enumerate(chunks), key=lambda item: item[1].get("rank") or item[0] + 1)
    for _, chunk in ranked:
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        text = re.sub(r"</documentos>", "&lt;/documentos&gt;", text, flags=re.IGNORECASE)
        n = len(evidence) + 1
        block = (
            f"[{n}] (fuente: {chunk.get('source', '')} | "
            f"colección: {chunk.get('collection_name', '')} | "
            f"fecha: {_date(chunk.get('created_ts'))} | "
            f"título: {chunk.get('title') or 's/t'})\n{text}"
        )
        block = _HANDLE.sub("@usuario", block)
        added = len(block) + (2 if blocks else 0)
        if size + added > budget_chars:
            continue
        blocks.append(block)
        evidence.append(dict(chunk, n=n))
        size += added
    return "\n\n".join(blocks), evidence


def build_prompt(question: str, context: str, template: str) -> str:
    """Sustituye marcadores sin reinterpretar el contenido insertado."""
    values = {"question": question, "context": context}
    return re.sub(r"\{(question|context)\}", lambda match: values[match[1]], template)
