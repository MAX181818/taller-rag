"""Reconocimiento de citas y abstenciones sin inferir sustento semántico."""
import pytest

from app.citations import invalid_citations, is_abstention, parse_citations


def test_parse_citations_variants():
    assert parse_citations("a [1] b [1, 3] c [2][4]") == [1, 2, 3, 4]
    assert parse_citations("[1,2,3] [3] [0] [12]") == [0, 1, 2, 3, 12]
    assert parse_citations("sin citas [abc] [1-3] [1, x]") == []


@pytest.mark.parametrize("answer", [
    "No hay evidencia suficiente en los documentos recuperados.",
    "NO HAY EVIDENCIA SUFICIENTE!", " No hay información suficiente. ",
    "Not enough evidence.", "THERE IS NOT ENOUGH EVIDENCE",
    "No hay evidencia suficiente en los documentos recuperados para proporcionar la receta.",
    "No hay evidencia suficiente en los documentos recuperados. [4]",
])
def test_is_abstention(answer):
    assert is_abstention(answer)


@pytest.mark.parametrize("answer", ["", "Hay evidencia suficiente [1].",
                                         "El evento es el 15 de mayo [1]. No hay evidencia suficiente sobre el lugar."])
def test_substantive_answer_is_not_abstention(answer):
    assert not is_abstention(answer)


def test_invalid_citations():
    assert invalid_citations([1, 7], 5) == [7]
    assert invalid_citations([0, 1, 2], 0) == [0, 1, 2]
    assert invalid_citations([-1, 1, 2], 2) == [-1]
