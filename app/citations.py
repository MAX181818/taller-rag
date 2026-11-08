"""Extracción de referencias y reconocimiento de abstenciones."""
import re
import unicodedata

from app.prompts import ABSTENTION


def parse_citations(answer: str) -> list[int]:
    groups = re.findall(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]", answer)
    return sorted({int(number) for group in groups for number in group.split(",")})


def _normalize(text: str) -> str:
    plain = "".join(char for char in unicodedata.normalize("NFD", text.casefold())
                    if not unicodedata.combining(char))
    return " ".join(re.sub(r"[\W_]+", " ", plain).split())


def is_abstention(answer: str) -> bool:
    variants = (ABSTENTION, "no hay evidencia suficiente", "not enough evidence",
                "no hay información suficiente", "there is not enough evidence")
    # Los modelos alargan la frase o le pegan una cita; la cita espuria se mide aparte (calidad de citas).
    normalized = _normalize(answer)
    return any(normalized.startswith(_normalize(variant)) for variant in variants)


def invalid_citations(cited: list[int], n_evidence: int) -> list[int]:
    return [number for number in cited if not 1 <= number <= n_evidence]
