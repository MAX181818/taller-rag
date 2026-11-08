"""Pruebas del análisis con modelos y búsquedas locales falsos."""
import csv
import importlib
import json

import pytest


@pytest.fixture
def probe():
    return importlib.import_module("scripts.embeddings_probe")


@pytest.mark.parametrize("left,right,want", [([1, 2], [1, 2], 1), ([1], [2], 0), ([1, 1, 2], [2, 3], 1 / 3), ([], [], 1)])
def test_jaccard_usa_conjuntos(probe, left, right, want):
    assert probe.jaccard(left, right) == pytest.approx(want)


def test_coseno_identico_y_ortogonal(probe):
    assert probe.cosine([3, 4], [3, 4]) == pytest.approx(1)
    assert probe.cosine([1, 0], [0, 1]) == pytest.approx(0)


def test_resumen_brecha_con_vectores_de_juguete(probe):
    rows = [dict(encoder="e5small", prefijo="query: ", tipo=kind,
                 similitud=probe.cosine([1, 0], vector))
            for kind, vector in [("parafrasis", [1, 0]), ("traduccion", [0, 1]),
                                 ("no_relacionados", [-1, 0])]]
    summary = probe.summarize(rows)
    assert len(summary) == 3
    assert all(row["brecha"] == pytest.approx(1.5) for row in summary)
    assert summary[0]["media"] == summary[0]["min"] == summary[0]["max"] == 1


def test_sin_prefijo_codifica_texto_crudo(probe, monkeypatch):
    class Model:
        def encode(self, texts, normalize_embeddings):
            assert texts == ["hola", "query: literal"]
            assert normalize_embeddings is True
            return [[1, 0], [0, 1]]
    monkeypatch.setattr(probe.embeddings, "_model", lambda encoder: Model())
    assert probe._embed_raw(["hola", "query: literal"], "e5small") == [[1, 0], [0, 1]]


def test_cli_genera_informes_con_fakes(probe, monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(probe.embeddings, "embed", lambda texts, kind, encoder: [[1, 0] for _ in texts])
    monkeypatch.setattr(probe, "_embed_raw", lambda texts, encoder: [[1, 0] for _ in texts])
    monkeypatch.setattr(probe.embeddings, "fingerprint", lambda encoder: {"model": encoder, "dim": 2})
    monkeypatch.setattr(probe, "get_client", lambda: object())

    def search(client, name, qvec, k, filters=None, fingerprint=None):
        calls.append((name, k, fingerprint))
        return [dict(document_id=f"doc{i // 2}", score=1 - i / 100,
                     collection_name="corpus", language="es", title="" if i else "Título @secreto | noticia",
                     text="Texto @oculto " + "x" * 100) for i in range(k)]

    monkeypatch.setattr(probe, "search", search)
    probe.main(["--out", str(tmp_path), "--k", "3"])
    md = (tmp_path / "vecinos.md").read_text(encoding="utf-8")
    for query, language in probe.QUERIES:
        assert query in md
    assert len(probe.QUERIES) == 8
    assert [language for _, language in probe.QUERIES].count("es") == 4
    assert md.count("| 3 |") == 16
    assert "| 4 |" not in md
    assert "e5small" in md and "minilm" in md
    assert "@secreto" not in md and "@oculto" not in md and "@usuario" in md
    assert "\\| noticia" in md
    assert all(k == 10 and fp["model"] in name for name, k, fp in calls)
    data = json.loads((tmp_path / "sensibilidad.json").read_text(encoding="utf-8"))
    assert set(data["fingerprints"]) == {"e5small", "minilm"}
    assert len(data["pares"]) == 18
    assert len(data["similitudes"]) == 54
    assert len(data["resumen"]) == 9
    assert all(row["similitud"] == pytest.approx(1) for row in data["similitudes"])
    assert all(row["brecha"] == pytest.approx(0) for row in data["resumen"])
    with (tmp_path / "solapamiento.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert all(float(row["jaccard_top10"]) == 1 and row["n_e5"] == row["n_minilm"] == "5" for row in rows)
    assert json.loads(capsys.readouterr().out)["consultas"] == 8


def test_k_invalido_no_abre_servicios(probe, monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "get_client", lambda: pytest.fail("No debe abrir servicios"))
    with pytest.raises(SystemExit):
        probe.main(["--out", str(tmp_path), "--k", "0"])
