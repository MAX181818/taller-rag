import importlib
import json
import pytest
import grpc
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException
import requests
from app import solr_store as solr_mod
from scripts import index_solr as solr_indexer
from copy import deepcopy
from app import retrieval
from app.embeddings import EncoderMismatch


# --- stores ---

class FakeClient:
    create_schema = staticmethod(MilvusClient.create_schema)
    prepare_index_params = staticmethod(MilvusClient.prepare_index_params)

    def __init__(self):
        self.collections = {}
        self.rows = {}
        self.calls = []
        self.failures = 0

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, schema, index_params, **kwargs):
        self.collections[collection_name] = dict(schema=schema.to_dict(), index=[entry.to_dict() for entry in index_params], **kwargs)
        self.rows[collection_name] = {}

    def describe_collection(self, collection_name):
        return self.collections[collection_name]["schema"]

    def load_collection(self, collection_name):
        self.calls.append(("load", collection_name))

    def upsert(self, collection_name, data):
        self.calls.append(("upsert", data))
        if self.failures:
            self.failures -= 1
            raise ConnectionError("Servidor no disponible")
        self.rows[collection_name].update({row["chunk_id"]: row for row in data})
        return {"upsert_count": len(data)}

    def query(self, collection_name, output_fields, filter="", ids=None, **kwargs):
        self.calls.append(("query", kwargs))
        rows = self.rows[collection_name]
        if output_fields == ["count(*)"]:
            return [{"count(*)": len(rows)}]
        return [{"chunk_id": key} for key in ids if key in rows]

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        # Misma forma que devuelve pymilvus 2.6.6: clave primaria por nombre de campo.
        return [[dict(chunk_id=key, distance=0.9, entity={k: row[k] for k in kwargs["output_fields"]})
                 for key, row in self.rows[kwargs["collection_name"]].items()][:kwargs["limit"]]]

    def drop_collection(self, collection_name):
        self.calls.append(("drop", collection_name))
        del self.collections[collection_name]
        del self.rows[collection_name]


def record(chunk_id="uno", **changes):
    return dict(dict(chunk_id=chunk_id, document_id="doc", provenance_ids=["doc", "otro"],
                     chunker="tok256", start_char=0, end_char=5, n_tokens=2, text="Hola.",
                     title=None, url=None, collection_name="bluesky_posts", source="fixture",
                     language="es", created_ts=-1, occurrences=[dict(document_id="doc")]), **changes)


@pytest.fixture
def store():
    return importlib.import_module("app.milvus_store")


def test_filter_expression(store):
    assert store.filter_expression(None) == store.filter_expression({}) == ""
    assert store.filter_expression(dict(collection_name="bluesky_posts", language="es")) == (
        'collection_name == "bluesky_posts" and language == "es"')
    assert store.filter_expression(dict(language='e"s\\x')) == 'language == "e\\"s\\\\x"'
    with pytest.raises(ValueError):
        store.filter_expression(dict(text="x"))


def test_schema_and_round_trip(store):
    client = FakeClient()
    fp = {"model": "modelo", "dim": 2}
    store.ensure_collection(client, "c", 2, fp)
    description = client.describe_collection("c")
    assert json.loads(description["description"]) == fp
    assert description["auto_id"] is False
    assert description["enable_dynamic_field"] is False
    fields = {f["name"]: f for f in description["fields"]}
    assert fields["chunk_id"]["is_primary"] is True
    assert fields["vector"]["params"]["dim"] == 2
    assert fields["provenance_ids"]["params"]["max_length"] == 65535
    index = client.collections["c"]["index"][0]
    assert index["index_type"] == "HNSW" and index["metric_type"] == "COSINE"
    assert index["M"] == 16 and index["efConstruction"] == 200
    assert store.upsert(client, "c", [record()], [[0.6, 0.8]]) == 1
    assert "occurrences" not in client.rows["c"]["uno"]
    assert store.count(client, "c") == 1
    assert client.calls[-1] == ("query", {"consistency_level": "Strong"})
    assert store.existing_ids(client, "c", ["uno", "dos"]) == {"uno"}
    assert store.existing_ids(client, "c", []) == set()
    result = store.search(client, "c", [0.6, 0.8], 5, dict(language="es"), fp)
    assert result[0]["provenance_ids"] == ["doc", "otro"] and result[0]["score"] == 0.9
    params = client.calls[-1][1]
    assert params["search_params"] == dict(metric_type="COSINE", params=dict(ef=64))
    store.drop(client, "c")
    assert not client.has_collection("c")


def test_fingerprint_mismatch_raises(store):
    from app.embeddings import EncoderMismatch
    client = FakeClient()
    store.ensure_collection(client, "c", 2, {"model": "anterior"})
    with pytest.raises(EncoderMismatch):
        store.search(client, "c", [0, 1], 5, fingerprint={"model": "nuevo"})
    with pytest.raises(EncoderMismatch):
        store.ensure_collection(client, "c", 2, {"model": "nuevo"})
    assert not any(call[0] == "search" for call in client.calls)


@pytest.mark.parametrize("field,value,size", [
    ("text", "á" * 8193, 16386), ("provenance_ids", ["á" * 32768], 65540),
    ("document_id", "á" * 129, 258),
])
def test_upsert_validates_bytes(store, field, value, size):
    client = FakeClient()
    with pytest.raises(ValueError) as error:
        store.upsert(client, "c", [record(**{field: value})], [[0, 1]])
    assert field in str(error.value) and "uno" in str(error.value) and str(size) in str(error.value)
    assert client.calls == []


def test_metadata_truncation_and_large_provenance(store):
    client = FakeClient()
    store.ensure_collection(client, "c", 2, {})
    provenance = [f"{i:032d}" for i in range(674)]
    row = record(title="a" + "😀" * 700, url="á" * 3000, provenance_ids=provenance)
    store.upsert(client, "c", [row], [[0, 1]])
    stored = client.rows["c"]["uno"]
    assert len(stored["title"].encode()) == 2045
    assert len(stored["url"].encode()) == 4096
    assert json.loads(stored["provenance_ids"]) == provenance
    assert row["title"] == "a" + "😀" * 700


@pytest.fixture
def indexer(monkeypatch, tmp_path):
    module = importlib.import_module("scripts.index_milvus")
    client = FakeClient()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.milvus_store, "get_client", lambda: client)
    monkeypatch.setattr(module.embeddings, "dim", lambda encoder: 2)
    monkeypatch.setattr(module.embeddings, "fingerprint", lambda encoder: {"model": encoder, "dim": 2})
    monkeypatch.setattr(module.embeddings, "embed", lambda texts, kind, encoder: [[0, 1] for _ in texts])
    return module, client, tmp_path


def write_chunks(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def test_index_cli_resume_recreate_describe(indexer, capsys):
    module, client, path = indexer
    chunks = write_chunks(path / "chunks.jsonl", [record(), record("dos")])
    args = ["--chunks", chunks, "--encoder", "e5small", "--batch", "1", "--describe"]
    assert module.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["collection"] == "c_e5small_tok256" and summary["expected"] == summary["count"] == 2
    assert summary["seconds"] >= 0
    description = json.loads((path / "reports/milvus/schema_c_e5small_tok256.json").read_text(encoding="utf-8"))
    assert description["count"] == 2 and all(key in description for key in ("schema", "index", "description"))
    client.calls.clear()
    assert module.main(args + ["--resume"]) == 0
    assert not any(c[0] == "upsert" for c in client.calls)
    assert module.main(args + ["--recreate"]) == 0
    assert ("drop", "c_e5small_tok256") in client.calls


def test_index_cli_retries_and_count_mismatch(indexer, monkeypatch, capsys):
    module, client, path = indexer
    delays = []
    monkeypatch.setattr(module.time, "sleep", delays.append)
    client.failures = 2
    chunks = write_chunks(path / "chunks.jsonl", [record()])
    args = ["--chunks", chunks]
    assert module.main(args) == 0 and delays == [1, 2]
    client.rows["c_e5small_tok256"]["extra"] = record("extra")
    assert module.main(args) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["count"] == 2


@pytest.mark.parametrize("rows", [[], [record(), record("dos", chunker="tok128")],
                                  [record(), record()], [record(text="á" * 8193)]])
def test_index_cli_rejects_invalid_input_before_connection(indexer, rows, monkeypatch):
    module, _, path = indexer
    def forbidden():
        pytest.fail("No debe conectarse con datos inválidos")
    monkeypatch.setattr(module.milvus_store, "get_client", forbidden)
    assert module.main(["--chunks", write_chunks(path / "chunks.jsonl", rows)]) == 1


@pytest.mark.parametrize("error", [MilvusException(code=grpc.StatusCode.UNAVAILABLE, message="Sin conexión"),
                                  MilvusException(code=65535, message="Servidor no disponible")])
def test_retry_handles_real_sdk_connection_codes(indexer, monkeypatch, error):
    module, _, _ = indexer
    calls, delays = [], []
    monkeypatch.setattr(module.time, "sleep", delays.append)

    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise error
        return 7

    assert module._retry_batch(operation) == 7
    assert len(calls) == 3 and delays == [1, 2]


@pytest.mark.parametrize("error", [ValueError("Valor inválido"), MilvusException(code=1100, message="Esquema inválido")])
def test_retry_does_not_repeat_invalid_data(indexer, monkeypatch, error):
    module, _, _ = indexer
    calls, delays = [], []
    monkeypatch.setattr(module.time, "sleep", delays.append)

    def operation():
        calls.append(1)
        raise error

    with pytest.raises(type(error)):
        module._retry_batch(operation)
    assert len(calls) == 1 and delays == []


def test_retry_stops_after_three_attempts(indexer, monkeypatch):
    module, client, path = indexer
    delays = []
    monkeypatch.setattr(module.time, "sleep", delays.append)
    client.failures = 8
    chunks = write_chunks(path / "chunks.jsonl", [record()])
    assert module.main(["--chunks", chunks]) == 1
    assert sum(c[0] == "upsert" for c in client.calls) == 3
    assert delays == [1, 2]


@pytest.mark.parametrize("code,retry", [(grpc.StatusCode.DEADLINE_EXCEEDED, True),
                                      (grpc.StatusCode.INVALID_ARGUMENT, False)])
def test_retry_handles_direct_grpc_errors(indexer, monkeypatch, code, retry):
    module, client, path = indexer
    calls, delays = [], []
    monkeypatch.setattr(module.time, "sleep", delays.append)

    class RpcFailure(grpc.RpcError):
        def code(self):
            return code

    def operation():
        calls.append(1)
        if len(calls) < 2:
            raise RpcFailure()
        return 7

    if retry:
        assert module._retry_batch(operation) == 7
        assert delays == [1]
    else:
        with pytest.raises(RpcFailure):
            module._retry_batch(operation)
        assert delays == [] and len(calls) == 1

    def fail_upsert(**kwargs):
        raise RpcFailure()

    monkeypatch.setattr(client, "upsert", fail_upsert)
    chunks = write_chunks(path / "chunks.jsonl", [record()])
    assert module.main(["--chunks", chunks]) == 1



# --- solr_store ---

@pytest.fixture
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


def record_solr_store(identifier="a", chunker="tok256"):
    return dict(chunk_id=identifier, document_id="doc", provenance_ids=["doc", "other"],
                chunker=chunker, start_char=0, end_char=5, n_tokens=1,
                text="Texto", title="Título", url="https://example.org", collection_name="A",
                source="test", language="es", created_ts=1, occurrences=[{"document_id": "doc"}])


@pytest.mark.usefixtures('no_network')
@pytest.mark.parametrize("special", list('+-!(){}[]^"~*?:\\/') + ["&&", "||"])
def test_escape_query(special):
    assert solr_mod.escape_query(special) == "".join("\\" + char for char in special)


@pytest.mark.usefixtures('no_network')
def test_escape_query_sentence_and_operators():
    assert solr_mod.escape_query('¿Qué dijo "Petro": (hoy)?') == '¿Qué dijo \\"Petro\\"\\: \\(hoy\\)\\?'
    assert solr_mod.escape_query("AND OR NOT ANDES CANDOR anD") == "and or not ANDES CANDOR anD"


@pytest.mark.usefixtures('no_network')
def test_search_builds_edismax_params(monkeypatch):
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response({"response": {"numFound": 0, "docs": []}})
    monkeypatch.setattr(requests, "get", get)
    assert solr_mod.search("a:b", 7, "tok256", {"collection_name": "A B", "language": "es"}) == []
    url, kwargs = calls[0]
    params = kwargs["params"]
    assert url.endswith("/select")
    assert params == dict(defType="edismax", q="a\\:b", qf="text text_es^0.8 text_en^0.8 title^0.5",
                          fq=["chunker:tok256", 'collection_name:"A B"', "language:es"],
                          rows=7, fl="*,score", wt="json")


@pytest.mark.usefixtures('no_network')
@pytest.mark.parametrize("provenance", [["doc", "other"], "doc"])
def test_provenance_roundtrip(monkeypatch, provenance):
    sent = []
    monkeypatch.setattr(requests, "post", lambda url, **kw: sent.extend(kw["json"]) or response())
    item = record_solr_store()
    assert solr_mod.upsert([item]) == 1
    assert sent[0]["provenance_ids"] == ["doc", "other"]
    returned = dict(sent[0], provenance_ids=provenance, score="2.5", _version_=42, text_es="Texto")
    monkeypatch.setattr(requests, "get", lambda *a, **kw: response({"response": {"docs": [returned]}}))
    result = solr_mod.search("Texto", 1, "tok256")[0]
    assert result == dict(sent[0], provenance_ids=provenance if isinstance(provenance, list) else [provenance], score=2.5)


@pytest.mark.usefixtures('no_network')
def test_upsert_drops_occurrences(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "post", lambda url, **kw: calls.append((url, kw)) or response())
    item = dict(record_solr_store(), unknown="x", vector=[1])
    solr_mod.upsert([item], commit=False)
    assert calls[0][0].endswith("/update")
    assert "commit" not in calls[0][1]["params"]
    assert set(calls[0][1]["json"][0]) == set(record_solr_store()) - {"occurrences"}
    assert "occurrences" in item
    solr_mod.upsert([item])
    assert calls[1][1]["params"]["commit"] == "true"


@pytest.mark.usefixtures('no_network')
def test_count_delete_and_ping(monkeypatch):
    calls = []
    def get(url, **kw):
        calls.append((url, kw))
        return response({"response": {"numFound": 12}, "status": "OK"})
    monkeypatch.setattr(requests, "get", get)
    assert solr_mod.count("tok256") == 12
    assert calls[0][1]["params"] == dict(q="*:*", fq="chunker:tok256", rows=0, wt="json")
    assert solr_mod.ping() is True
    assert calls[1][0].endswith("/admin/ping")
    monkeypatch.setattr(requests, "post", lambda url, **kw: calls.append((url, kw)) or response())
    solr_mod.delete_chunker("tok:256")
    assert calls[-1][1]["json"] == {"delete": {"query": "chunker:tok\\:256"}}
    assert calls[-1][1]["params"]["commit"] == "true"


@pytest.mark.usefixtures('no_network')
def test_errors_propagate_except_ping(monkeypatch):
    error = requests.ConnectionError("sin conexión")
    def fail(*a, **kw):
        raise error
    monkeypatch.setattr(requests, "get", fail)
    assert solr_mod.ping() is False
    with pytest.raises(requests.ConnectionError) as caught:
        solr_mod.count("tok256")
    assert caught.value is error


@pytest.mark.usefixtures('no_network')
def test_resume_large_batch_uses_post(monkeypatch):
    identifiers = [f"{number:032x}" for number in range(500)]
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response({"response": {"numFound": 500}})
    monkeypatch.setattr(requests, "post", post)
    assert solr_mod.contains_all(identifiers, "tok256") is True
    assert calls[0][0].endswith("/select")
    assert calls[0][1]["data"] == dict(q="{!terms f=chunk_id}" + ",".join(identifiers),
                                             fq="chunker:tok256", rows=0, wt="json")


def chunks_file(tmp_path, records):
    path = tmp_path / "chunks.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    return str(path)


@pytest.mark.usefixtures('no_network')
def test_index_exit_code_on_count_mismatch(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(solr_mod, "upsert", lambda records, commit=True: len(records))
    monkeypatch.setattr(solr_mod, "commit", lambda: None)
    monkeypatch.setattr(solr_mod, "count", lambda chunker: 0)
    assert solr_indexer.main(["--chunks", chunks_file(tmp_path, [record_solr_store()])]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary == dict(collection="solr:rag", chunker="tok256", expected=1, count=0, seconds=summary["seconds"])


def check_http_retries(monkeypatch, tmp_path, caplog, status, attempts, delays):
    calls, sleeps = [], []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return response({"error": {"msg": "campo desconocido"}}, status)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(solr_indexer.time, "sleep", sleeps.append)
    assert solr_indexer.main(["--chunks", chunks_file(tmp_path, [record_solr_store()])]) == 1
    assert len(calls) == attempts
    assert sleeps == delays
    assert "campo desconocido" in caplog.text


@pytest.mark.usefixtures('no_network')
def test_400_not_retried(monkeypatch, tmp_path, caplog):
    check_http_retries(monkeypatch, tmp_path, caplog, 400, 1, [])


@pytest.mark.usefixtures('no_network')
def test_5xx_retried_three_times(monkeypatch, tmp_path, caplog):
    check_http_retries(monkeypatch, tmp_path, caplog, 503, 3, [1, 2])


@pytest.mark.usefixtures('no_network')
def test_connection_retry_then_success(monkeypatch, tmp_path):
    calls, sleeps = [], []
    def post(url, **kw):
        calls.append(kw)
        if len(calls) < 3:
            raise requests.ConnectionError("sin conexión")
        return response()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *a, **kw: response({"response": {"numFound": 1}}))
    monkeypatch.setattr(solr_indexer.time, "sleep", sleeps.append)
    assert solr_indexer.main(["--chunks", chunks_file(tmp_path, [record_solr_store()])]) == 0
    assert sleeps == [1, 2]
    assert calls[-1]["json"] == {"commit": {}}


@pytest.mark.usefixtures('no_network')
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
    path = chunks_file(tmp_path, [record_solr_store(str(i)) for i in range(11)])
    assert solr_indexer.main(["--chunks", path, "--batch", "1", "--recreate"]) == 0
    assert posts[0]["json"] == {"delete": {"query": "chunker:tok256"}}
    assert sum(item["json"] == {"commit": {}} for item in posts) == 2
    posts.clear()
    assert solr_indexer.main(["--chunks", path, "--batch", "2", "--resume"]) == 0
    assert not any(isinstance(item["json"], list) for item in posts)
    assert sum(item["q"].startswith("{!terms f=chunk_id}") for item in gets) == 6
    ids.remove("0")
    posts.clear()
    assert solr_indexer.main(["--chunks", path, "--resume"]) == 0
    assert len(ids) == 11
    assert sum(isinstance(item["json"], list) for item in posts) == 1


@pytest.mark.usefixtures('no_network')
@pytest.mark.parametrize("records", [[], [record_solr_store(), record_solr_store("b", "sent256")]])
def test_invalid_input_has_no_side_effects(tmp_path, records):
    assert solr_indexer.main(["--chunks", chunks_file(tmp_path, records), "--recreate"]) == 1


@pytest.mark.usefixtures('no_network')
def test_index_skips_blank_lines(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n  \n" + json.dumps(record_solr_store()) + "\n\t\n", encoding="utf-8")
    assert list(solr_indexer._records(path)) == [record_solr_store()]


@pytest.mark.usefixtures('no_network')
def test_solr_endpoint_uses_shared_config(monkeypatch):
    import importlib
    from app import config

    monkeypatch.setattr(config, "SOLR_URL", "http://configured:8983/solr/core/")
    try:
        importlib.reload(solr_mod)
        urls = []
        monkeypatch.setattr(requests, "get", lambda url, **kw: urls.append(url) or response({"status": "OK"}))
        assert solr_mod.ping() is True
        assert urls == ["http://configured:8983/solr/core/admin/ping"]
    finally:
        monkeypatch.undo()
        importlib.reload(solr_mod)



# --- retrieval ---

def record_retrieval(chunk_id, document_id=None, score=1.0):
    return dict(chunk_id=chunk_id, document_id=document_id or chunk_id,
                provenance_ids=[document_id or chunk_id, "origen"], score=score,
                text="Texto", title="Título", url="https://example.org",
                collection_name="archivo", source="prueba", language="es",
                created_ts=1, start_char=0, end_char=5, n_tokens=1, chunker="tok256")


@pytest.fixture
def fake_dependencies(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No se permite acceso de red")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(retrieval.milvus_store, "get_client", lambda: "cliente")
    monkeypatch.setattr(retrieval.milvus_store, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval.solr_store, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval, "embed", lambda *args, **kwargs: [[0.1, 0.2]])
    monkeypatch.setattr(retrieval, "fingerprint", lambda encoder: {"model": encoder})


@pytest.mark.usefixtures('fake_dependencies')
def test_rrf_merges_by_chunk_id():
    lists = [[record_retrieval("a"), record_retrieval("b")], [record_retrieval("b"), record_retrieval("c")]]
    original = deepcopy(lists)
    rows = retrieval.rrf(lists)
    assert [row["chunk_id"] for row in rows] == ["b", "a", "c"]
    assert [row["score"] for row in rows] == pytest.approx(
        [1 / 61 + 1 / 62, 1 / 61, 1 / 62])
    assert lists == original


@pytest.mark.usefixtures('fake_dependencies')
def test_rrf_ties_use_best_rank_then_chunk_id():
    rows = retrieval.rrf([[record_retrieval("z"), record_retrieval("a")],
                          [record_retrieval("a"), record_retrieval("z")]], k_rrf=10)
    assert [row["chunk_id"] for row in rows] == ["a", "z"]
    assert rows[0]["score"] == pytest.approx(1 / 11 + 1 / 12)
    assert retrieval.rrf([[], []]) == []
    rows = retrieval.rrf([[record_retrieval("z"), record_retrieval("a"), record_retrieval("b")],
                          [record_retrieval("b"), record_retrieval("a"), record_retrieval("z")]], k_rrf=0)
    assert [row["chunk_id"] for row in rows] == ["b", "z", "a"]


@pytest.mark.usefixtures('fake_dependencies')
@pytest.mark.parametrize("backend", ["milvus", "solr", "hybrid"])
@pytest.mark.parametrize("k", [2, 30])
def test_one_per_doc_keeps_best_chunk(monkeypatch, backend, k):
    rows = [record_retrieval("a", "doc1", 9), record_retrieval("b", "doc1", 8),
            record_retrieval("c", "doc2", 7), record_retrieval("d", "doc3", 6)]
    calls = []

    def solr(query, limit, chunker, filters=None):
        calls.append(limit)
        return rows[:limit]

    def milvus(client, name, qvec, limit, **kwargs):
        calls.append(limit)
        return rows[:limit]

    monkeypatch.setattr(retrieval.solr_store, "search", solr)
    monkeypatch.setattr(retrieval.milvus_store, "search", milvus)
    result = retrieval.retrieve("consulta", backend, k=k, one_per_doc=True)
    assert [row["chunk_id"] for row in result] == ["a", "c", "d"][:k]
    expected = min(k * 4, 100)
    if backend == "hybrid":
        expected = max(20, expected)
    assert calls == [expected] * (2 if backend == "hybrid" else 1)
    assert [row["rank"] for row in result] == list(range(1, len(result) + 1))


@pytest.mark.usefixtures('fake_dependencies')
@pytest.mark.parametrize("backend", ["solr", "milvus"])
@pytest.mark.parametrize("error", [ConnectionError("caído"),
                                   requests.RequestException("servidor"),
                                   MilvusException(message="servidor")])
@pytest.mark.parametrize("hybrid", [False, True])
def test_backend_error_raises_retriever_unavailable(monkeypatch, backend, error, hybrid):
    def fail(*args, **kwargs):
        raise error

    store = retrieval.solr_store if backend == "solr" else retrieval.milvus_store
    monkeypatch.setattr(store, "search", fail)
    with pytest.raises(retrieval.RetrieverUnavailable) as caught:
        retrieval.retrieve("consulta", "hybrid" if hybrid else backend)
    assert caught.value.backend == backend
    assert backend in str(caught.value)
    assert "no está disponible" in str(caught.value)
    assert caught.value.__cause__ is error


@pytest.mark.usefixtures('fake_dependencies')
def test_milvus_client_connection_error_is_wrapped(monkeypatch):
    def fail():
        raise ConnectionError("caído")

    monkeypatch.setattr(retrieval.milvus_store, "get_client", fail)
    with pytest.raises(retrieval.RetrieverUnavailable) as caught:
        retrieval.retrieve("consulta")
    assert caught.value.backend == "milvus"


@pytest.mark.usefixtures('fake_dependencies')
@pytest.mark.parametrize("backend", ["milvus", "solr", "hybrid"])
def test_ranks_are_1_based_and_contiguous(monkeypatch, backend):
    rows = [record_retrieval("a", score=5), record_retrieval("b", score=3), record_retrieval("c", score=2)]
    original = deepcopy(rows)
    calls = []
    filters = {"language": "es"}

    def solr(query, k, chunker, filters=None):
        calls.append((query, k, chunker, filters))
        return rows

    def milvus(client, name, qvec, k, filters=None, fingerprint=None):
        calls.append(("consulta", k, "tok512", filters))
        return rows

    monkeypatch.setattr(retrieval.solr_store, "search", solr)
    monkeypatch.setattr(retrieval.milvus_store, "search", milvus)
    result = retrieval.retrieve("consulta", backend, k=2, chunker="tok512", filters=filters)
    assert [row["rank"] for row in result] == [1, 2]
    assert [row["backend"] for row in result] == [backend, backend]
    assert [row["chunk_id"] for row in result] == ["a", "b"]
    for row, source in zip(result, rows):
        assert {key: row[key] for key in source if key != "score"} == {
            key: value for key, value in source.items() if key != "score"}
    assert result[0]["score"] == pytest.approx(2 / 61 if backend == "hybrid" else 5)
    assert calls == [("consulta", 20 if backend == "hybrid" else 2, "tok512", filters)] * (
        2 if backend == "hybrid" else 1)
    assert rows == original


@pytest.mark.usefixtures('fake_dependencies')
def test_milvus_uses_query_prefix_kind_and_fingerprint(monkeypatch):
    calls = []
    stamp = {"model": "minilm", "revision": "prueba"}

    def embed(texts, kind, encoder):
        calls.append((texts, kind, encoder))
        return [[0.5, 0.7]]

    def fingerprint(encoder):
        assert encoder == "minilm"
        return stamp

    def search(client, name, qvec, k, filters=None, fingerprint=None):
        assert (client, name, qvec, k, filters, fingerprint) == (
            "cliente", "c_minilm_tok512", [0.5, 0.7], 5, {"language": "es"}, stamp)
        return []

    monkeypatch.setattr(retrieval, "embed", embed)
    monkeypatch.setattr(retrieval, "fingerprint", fingerprint)
    monkeypatch.setattr(retrieval.milvus_store, "search", search)
    assert retrieval.retrieve("consulta", chunker="tok512", encoder="minilm",
                              filters={"language": "es"}) == []
    assert calls == [(["consulta"], "query", "minilm")]


@pytest.mark.usefixtures('fake_dependencies')
@pytest.mark.parametrize("backend", ["milvus", "hybrid"])
def test_encoder_mismatch_propagates(monkeypatch, backend):
    error = EncoderMismatch("Huella distinta")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(retrieval.milvus_store, "search", fail)
    with pytest.raises(EncoderMismatch) as caught:
        retrieval.retrieve("consulta", backend)
    assert caught.value is error


@pytest.mark.usefixtures('fake_dependencies')
def test_solr_does_not_encode(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Solr no requiere vectores")

    monkeypatch.setattr(retrieval, "embed", forbidden)
    monkeypatch.setattr(retrieval, "fingerprint", forbidden)
    monkeypatch.setattr(retrieval.milvus_store, "get_client", forbidden)
    assert retrieval.retrieve("consulta", "solr") == []


@pytest.mark.usefixtures('fake_dependencies')
def test_invalid_backend_and_nonpositive_k():
    with pytest.raises(ValueError, match="Backend"):
        retrieval.retrieve("consulta", "otro")
    for backend in ("milvus", "solr", "hybrid"):
        assert retrieval.retrieve("consulta", backend, k=0) == []
        assert retrieval.retrieve("consulta", backend, k=-1) == []
