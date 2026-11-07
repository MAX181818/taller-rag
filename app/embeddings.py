"""Vectores normalizados y huellas de los codificadores en CPU."""
from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

from app import config


ENCODERS = {
    "e5small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
}


class EncoderMismatch(Exception):
    """La colección pertenece a un codificador con otra huella."""


def _encoder(encoder):
    try:
        return ENCODERS[encoder]
    except KeyError:
        raise ValueError(f"Codificador no soportado: {encoder}") from None


@lru_cache(maxsize=None)
def _model(encoder):
    name, _, _ = _encoder(encoder)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cpu")


def embed(texts: list[str], kind: Literal["query", "passage"],
          encoder: str = config.ENCODER) -> list[list[float]]:
    if kind not in ("query", "passage"):
        raise ValueError(f"Tipo de texto no soportado: {kind}")
    _, query_prefix, passage_prefix = _encoder(encoder)
    if not texts:
        return []
    prefix = query_prefix if kind == "query" else passage_prefix
    vectors = _model(encoder).encode([prefix + text for text in texts], normalize_embeddings=True)
    return [[float(value) for value in vector] for vector in vectors]


def dim(encoder) -> int:
    return int(_model(encoder).get_sentence_embedding_dimension())


def fingerprint(encoder) -> dict:
    model, query_prefix, passage_prefix = _encoder(encoder)
    dimension = dim(encoder)
    home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache/huggingface")))
    ref = home / "hub" / ("models--" + model.replace("/", "--")) / "refs/main"
    revision = ref.read_text(encoding="utf-8").strip() if ref.is_file() else "desconocida"
    return dict(model=model, revision=revision, query_prefix=query_prefix,
                passage_prefix=passage_prefix, normalize=True, dim=dimension)
