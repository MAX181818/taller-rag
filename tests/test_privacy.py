"""Privacidad compartida y salidas de texto sin servicios."""
import csv
import pytest


@pytest.mark.parametrize("text,want", [
    ("Hola @ana.bsky.social y @x.com, @local", "Hola @usuario y @usuario, @local"),
    ("https://bsky.app/profile/ana.bsky.social/post/3", "https://bsky.app/profile/usuario/post/3"),
    ("bsky.app/profile/did:plc:abc/post/3", "bsky.app/profile/usuario/post/3"),
    ("@usuario https://example.com/x", "@usuario https://example.com/x"),
    ("correo a@correo.org.", "correo a@correo.org."),
])
def test_mask_handles(text, want):
    from app.privacy import mask_handles
    assert mask_handles(text) == want
    assert mask_handles(want) == want


def test_reexport_y_consumidores():
    from app.privacy import mask_handles
    from scripts import validate_gold, embeddings_probe, eval_retrieval
    assert validate_gold.mask_handles is mask_handles
    text = "@ana.bsky.social https://bsky.app/profile/ana.bsky.social/post/1"
    assert embeddings_probe._cell(text) == mask_handles(text)
    assert eval_retrieval.mask_text(text) == mask_handles(text)


def test_contexto_enmascara_perfiles():
    from app.prompts import build_context
    text = "https://bsky.app/profile/ana.bsky.social/post/1"
    context, _ = build_context([dict(text=text)], 1000)
    assert "ana.bsky.social" not in context


def test_human_eval_enmascara_antes_del_recorte(tmp_path):
    from scripts import human_eval
    prefix = "x" * 389 + " "
    text = prefix + "@ana.bsky.social final"
    record = dict(qid="Q01", run_id="r", run_idx=1, model="m", backend="solr",
                  answer="respuesta", evidence=[dict(n=1, text=text)])
    human_eval.sample([record], {"Q01": dict(type="directa")}, 1, 42, tmp_path)
    with (tmp_path / "hoja_evaluador_A.csv").open(encoding="utf-8-sig") as source:
        item = next(csv.DictReader(source))
    assert item["evidencia"] == "[1] " + (prefix + "@usuario final")[:400]


def test_validate_gold_como_script():
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "scripts/validate_gold.py", "--help"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--gold" in result.stdout
