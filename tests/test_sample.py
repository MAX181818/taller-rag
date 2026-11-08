"""Contratos de muestreo y fechas para las tareas siguientes."""
import pandas as pd
import pytest
from scripts.make_sample import make_sample, normalize_created_at


def corpus():
    return pd.DataFrame([{"document_id": f"{i:05d}", "text_sha256": f"h{i}",
                          "text": f"Texto {i}", "collection_name": "A" if i < 9000 else "B"}
                         for i in range(12000)])


def test_dedup_keeps_min_id_and_all_provenance():
    df = pd.DataFrame({"document_id": ["b", "a", "c"], "text_sha256": ["h"] * 3,
                       "text": ["Texto"] * 3, "collection_name": ["A"] * 3})
    assert make_sample(df, 1).to_dict("records") == [
        {"document_id": "a", "text_sha256": "h", "collection_name": "A", "provenance_ids": "a|b|c"}]


def test_sample_is_deterministic_and_stratified():
    df = corpus()
    a = make_sample(df, 700)
    pd.testing.assert_frame_equal(a, make_sample(df.sample(frac=1, random_state=8), 700))
    assert len(a) == 700 and a.document_id.is_unique
    assert abs((a.collection_name == "A").mean() - 0.75) <= 0.01
    assert list(a.columns) == ["document_id", "text_sha256", "collection_name", "provenance_ids"]


def test_samples_are_nested():
    df = corpus()
    small, large = make_sample(df, 700), make_sample(df, 10000)
    assert len(large) == 10000
    assert set(small.document_id) <= set(large.document_id)
    for sample in (small, large):
        assert abs((sample.collection_name == "A").mean() - 0.75) <= 0.01


def test_excludes_empty_text():
    df = pd.DataFrame({"document_id": list("abcde"), "text_sha256": list("abcde"),
                       "text": ["", "  \t", None, "Texto", "\n"], "collection_name": ["A"] * 5})
    assert make_sample(df, 1).document_id.tolist() == ["d"]


@pytest.mark.parametrize("value,expected", [
    ("2026-08-16T21:20:04-03:00", 1786926004), ("2024-01-02", 1704153600),
    (None, -1), ("basura", -1), ("", -1)])
def test_normalize_created_at(value, expected):
    assert normalize_created_at(value) == expected


def test_invalid_sample_size():
    with pytest.raises(ValueError):
        make_sample(corpus(), 12001)
    with pytest.raises(ValueError):
        make_sample(corpus(), -1)
    assert make_sample(corpus(), 0).empty


def test_manifest_comparison(tmp_path):
    """Contrasta cifras y excluye conteos personales del documento publicado."""
    import json
    from scripts.profile_corpus import manifest_comparison

    archivo = tmp_path / "corpus.parquet"
    archivo.write_bytes(b"datos")
    manifiesto = tmp_path / "manifiesto.md"
    manifiesto.write_text(json.dumps({
        "rows": 3, "columns": 4, "output_bytes": 5, "output_sha256": "abc",
        "nonempty_counts": {"text": 1, "author_id": 2},
        "distributions": {"collection_name": {"A": 3}},
        "unique_texts": 2, "duplicate_text_rows": 1,
        "validation": {"ids_unique": True, "all_columns_round_trip": True},
    }), encoding="utf-8")
    df = pd.DataFrame({"document_id": ["a", "b", "c"], "text": ["Texto", "  ", None],
                       "collection_name": ["A"] * 3, "author_id": ["x", "y", None]})
    perfil = {"rows": 3, "fields": 4, "sha256_before": "abc", "unique_text_hashes": 2,
              "dup_text_rows": 1, "document_id_unique": True}
    resultado = manifest_comparison(manifiesto, df, perfil, archivo)
    cifras = {fila["figure"]: fila for fila in resultado["comparison"]}
    assert cifras["nonempty_counts.text"]["measured"] == 1
    assert cifras["output_bytes"]["matches"] is True
    assert cifras["distributions.collection_name.A"]["matches"] is True
    assert cifras["validation.all_columns_round_trip"]["matches"] is None
    assert "author_id" not in json.dumps(resultado)
    for path in (None, tmp_path / "ausente"):
        missing = manifest_comparison(path, df, perfil, archivo)
        assert missing["status"] == "Manifiesto no proporcionado: se omite la comparaci\u00f3n."
        assert missing["comparison"] == []
