"""Contratos del almacén y del indexador sin servicios externos."""
import json

import pytest
import requests

from app import solr_store as store
from scripts import index_solr as indexer


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No se permite acceso de red")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


def response(data=None, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(data or {}).encode()
    result.url = "http://solr/solr/rag"
    return result


def record(identifier="a", chunker="tok256"):
    return dict(chunk_id=identifier, document_id="doc", provenance_ids=["doc", "other"],
                chunker=chunker, start_char=0, end_char=5, n_tokens=1,
                text="Texto", title="Título", url="https://example.org", collection_name="A",
                source="test", language="es", created_ts=1, occurrences=[{"document_id": "doc"}])


@pytest.mark.parametrize("special", list('+-!(){}[]^"~*?:\\/') + ["&&", "||"])
def test_escape_query(special):
    assert store.escape_query(special) == "".join("\\" + char for char in special)


def test_escape_query_sentence_and_operators():
    assert store.escape_query('¿Qué dijo "Petro": (hoy)?') == '¿Qué dijo \\"Petro\\"\\: \\(hoy\\)\\?'
    assert store.escape_query("AND OR NOT ANDES CANDOR anD") == "and or not ANDES CANDOR anD"


def test_search_builds_edismax_params(monkeypatch):
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response({"response": {"numFound": 0, "docs": []}})
    monkeypatch.setattr(requests, "get", get)
    assert store.search("a:b", 7, "tok256", {"collection_name": "A B", "language": "es"}) == []
    url, kwargs = calls[0]
    params = kwargs["params"]
    assert url.endswith("/select")
    assert params == dict(defType="edismax", q="a\\:b", qf="text text_es^0.8 text_en^0.8 title^0.5",
                          fq=["chunker:tok256", 'collection_name:"A B"', "language:es"],
                          rows=7, fl="*,score", wt="json")


@pytest.mark.parametrize("provenance", [["doc", "other"], "doc"])
def test_provenance_roundtrip(monkeypatch, provenance):
    sent = []
    monkeypatch.setattr(requests, "post", lambda url, **kw: sent.extend(kw["json"]) or response())
    item = record()
    assert store.upsert([item]) == 1
    assert sent[0]["provenance_ids"] == ["doc", "other"]
    returned = dict(sent[0], provenance_ids=provenance, score="2.5", _version_=42, text_es="Texto")
    monkeypatch.setattr(requests, "get", lambda *a, **kw: response({"response": {"docs": [returned]}}))
    result = store.search("Texto", 1, "tok256")[0]
    assert result == dict(sent[0], provenance_ids=provenance if isinstance(provenance, list) else [provenance], score=2.5)


def test_upsert_drops_occurrences(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "post", lambda url, **kw: calls.append((url, kw)) or response())
    item = dict(record(), unknown="x", vector=[1])
    store.upsert([item], commit=False)
    assert calls[0][0].endswith("/update")
    assert "commit" not in calls[0][1]["params"]
    assert set(calls[0][1]["json"][0]) == set(record()) - {"occurrences"}
    assert "occurrences" in item
    store.upsert([item])
    assert calls[1][1]["params"]["commit"] == "true"


def test_count_delete_and_ping(monkeypatch):
    calls = []
    def get(url, **kw):
        calls.append((url, kw))
        return response({"response": {"numFound": 12}, "status": "OK"})
    monkeypatch.setattr(requests, "get", get)
    assert store.count("tok256") == 12
    assert calls[0][1]["params"] == dict(q="*:*", fq="chunker:tok256", rows=0, wt="json")
    assert store.ping() is True
    assert calls[1][0].endswith("/admin/ping")
    monkeypatch.setattr(requests, "post", lambda url, **kw: calls.append((url, kw)) or response())
    store.delete_chunker("tok:256")
    assert calls[-1][1]["json"] == {"delete": {"query": "chunker:tok\\:256"}}
    assert calls[-1][1]["params"]["commit"] == "true"


def test_errors_propagate_except_ping(monkeypatch):
    error = requests.ConnectionError("sin conexión")
    def fail(*a, **kw):
        raise error
    monkeypatch.setattr(requests, "get", fail)
    assert store.ping() is False
    with pytest.raises(requests.ConnectionError) as caught:
        store.count("tok256")
    assert caught.value is error


def test_resume_large_batch_uses_post(monkeypatch):
    identifiers = [f"{number:032x}" for number in range(500)]
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response({"response": {"numFound": 500}})
    monkeypatch.setattr(requests, "post", post)
    assert store.contains_all(identifiers, "tok256") is True
    assert calls[0][0].endswith("/select")
    assert calls[0][1]["data"] == dict(q="{!terms f=chunk_id}" + ",".join(identifiers),
                                             fq="chunker:tok256", rows=0, wt="json")


def chunks_file(tmp_path, records):
    path = tmp_path / "chunks.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    return str(path)


def test_index_exit_code_on_count_mismatch(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(store, "upsert", lambda records, commit=True: len(records))
    monkeypatch.setattr(store, "commit", lambda: None)
    monkeypatch.setattr(store, "count", lambda chunker: 0)
    assert indexer.main(["--chunks", chunks_file(tmp_path, [record()])]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary == dict(collection="solr:rag", chunker="tok256", expected=1, count=0, seconds=summary["seconds"])


def check_http_retries(monkeypatch, tmp_path, caplog, status, attempts, delays):
    calls, sleeps = [], []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return response({"error": {"msg": "campo desconocido"}}, status)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(indexer.time, "sleep", sleeps.append)
    assert indexer.main(["--chunks", chunks_file(tmp_path, [record()])]) == 1
    assert len(calls) == attempts
    assert sleeps == delays
    assert "campo desconocido" in caplog.text


def test_400_not_retried(monkeypatch, tmp_path, caplog):
    check_http_retries(monkeypatch, tmp_path, caplog, 400, 1, [])


def test_5xx_retried_three_times(monkeypatch, tmp_path, caplog):
    check_http_retries(monkeypatch, tmp_path, caplog, 503, 3, [1, 2])


def test_connection_retry_then_success(monkeypatch, tmp_path):
    calls, sleeps = [], []
    def post(url, **kw):
        calls.append(kw)
        if len(calls) < 3:
            raise requests.ConnectionError("sin conexión")
        return response()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *a, **kw: response({"response": {"numFound": 1}}))
    monkeypatch.setattr(indexer.time, "sleep", sleeps.append)
    assert indexer.main(["--chunks", chunks_file(tmp_path, [record()])]) == 0
    assert sleeps == [1, 2]
    assert calls[-1]["json"] == {"commit": {}}


def test_resume_recreate_and_commits(monkeypatch, tmp_path):
    posts, gets = [], []
    ids = set()
    def post(url, **kw):
        if "data" in kw:
            return get(url, params=kw["data"])
        posts.append(kw)
        if isinstance(kw["json"], list):
            ids.update(item["chunk_id"] for item in kw["json"])
        return response()
    def get(url, **kw):
        gets.append(kw["params"])
        if kw["params"]["q"].startswith("{!terms"):
            wanted = kw["params"]["q"].split("}", 1)[1].split(",")
            return response({"response": {"numFound": sum(item in ids for item in wanted)}})
        return response({"response": {"numFound": len(ids)}})
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", get)
    path = chunks_file(tmp_path, [record(str(i)) for i in range(11)])
    assert indexer.main(["--chunks", path, "--batch", "1", "--recreate"]) == 0
    assert posts[0]["json"] == {"delete": {"query": "chunker:tok256"}}
    assert sum(item["json"] == {"commit": {}} for item in posts) == 2
    posts.clear()
    assert indexer.main(["--chunks", path, "--batch", "2", "--resume"]) == 0
    assert not any(isinstance(item["json"], list) for item in posts)
    assert sum(item["q"].startswith("{!terms f=chunk_id}") for item in gets) == 6
    ids.remove("0")
    posts.clear()
    assert indexer.main(["--chunks", path, "--resume"]) == 0
    assert len(ids) == 11
    assert sum(isinstance(item["json"], list) for item in posts) == 1


@pytest.mark.parametrize("records", [[], [record(), record("b", "sent256")]])
def test_invalid_input_has_no_side_effects(tmp_path, records):
    assert indexer.main(["--chunks", chunks_file(tmp_path, records), "--recreate"]) == 1


def test_index_skips_blank_lines(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n  \n" + json.dumps(record()) + "\n\t\n", encoding="utf-8")
    assert list(indexer._records(path)) == [record()]


def test_solr_endpoint_uses_shared_config(monkeypatch):
    import importlib
    from app import config

    monkeypatch.setattr(config, "SOLR_URL", "http://configured:8983/solr/core/")
    try:
        importlib.reload(store)
        urls = []
        monkeypatch.setattr(requests, "get", lambda url, **kw: urls.append(url) or response({"status": "OK"}))
        assert store.ping() is True
        assert urls == ["http://configured:8983/solr/core/admin/ping"]
    finally:
        monkeypatch.undo()
        importlib.reload(store)
