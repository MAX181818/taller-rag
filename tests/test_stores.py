"""Contratos Milvus e indexación con cliente en memoria."""
import importlib
import json

import pytest
import grpc
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException


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
        return [[dict(id=key, distance=0.9, entity={k: row[k] for k in kwargs["output_fields"]})
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
