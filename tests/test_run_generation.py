"""Experimentos y monitor con dependencias simuladas, sin servicios."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import requests

from scripts import run_generation as runner
from scripts import sysmon


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dependencia externa sin simular")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def experiment(monkeypatch, tmp_path):
    gold = tmp_path / "gold.jsonl"
    questions = [
        dict(qid="Q01", split="test", question="Primera", type="directa"),
        dict(qid="Q02", split="test", question="Segunda", type="sintesis"),
        dict(qid="Q03", split="dev", question="Desarrollo", type="directa"),
    ]
    gold.write_text("".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8")
    state = SimpleNamespace(retrieved=[], answered=[], events=[], hardware_calls=0,
                            version_calls=0, fail_once=False, mutate=False)

    def retrieve(query, **kwargs):
        state.retrieved.append((query, deepcopy(kwargs)))
        state.events.append(("retrieve", query, kwargs["backend"]))
        return [dict(chunk_id=query + kwargs["backend"], text=query, rank=1)]

    def answer(query, **kwargs):
        state.answered.append((query, deepcopy(kwargs)))
        state.events.append(("answer", kwargs["model"], kwargs["backend"], query))
        if state.fail_once:
            state.fail_once = False
            raise RuntimeError("generacion fallida")
        digest = hashlib.sha256(json.dumps(kwargs["evidence"], sort_keys=True).encode()).hexdigest()
        if state.mutate:
            kwargs["evidence"][0]["text"] = "alterado"
        return dict(answer="Respuesta [1]", model=kwargs["model"], backend=kwargs["backend"],
                    context_sha256=digest, timings={"retrieval_ms": 0}, tokens={"prompt": 3})

    hardware = dict(cpu="CPU de prueba", ram_gb=16, gpu=None, vram_gb=None)
    versions = dict(ollama="0.0", milvus=None, solr=None, python="3.11",
                    packages={name: "1" for name in runner.PACKAGES})

    def collect_hardware():
        state.hardware_calls += 1
        return hardware

    def collect_versions():
        state.version_calls += 1
        return versions

    monkeypatch.setattr(runner, "retrieve", retrieve)
    monkeypatch.setattr(runner, "answer", answer)
    monkeypatch.setattr(runner, "unload", lambda model: state.events.append(("unload", model)))
    monkeypatch.setattr(runner, "list_models", lambda: ["ajeno", "m1", "m2"])
    monkeypatch.setattr(runner, "model_info", lambda model: {"digest": model})
    monkeypatch.setattr(runner, "collect_hardware", collect_hardware)
    monkeypatch.setattr(runner, "collect_versions", collect_versions)
    state.argv = ["--gold", str(gold), "--out-dir", str(tmp_path / "raw"),
                  "--models", "m1", "m2", "--backends", "milvus", "solr",
                  "--runs", "2", "--tag", "ensayo", "--one-per-doc"]
    state.out = tmp_path / "raw"
    state.execute = lambda *extra: runner.run(runner.build_parser().parse_args(state.argv + list(extra)))
    return state


def test_frozen_evidence_order_and_cold(experiment):
    result = experiment.execute()
    records = read_jsonl(Path(result["file"]))
    frozen = read_jsonl(experiment.out / "ensayo_evidence.jsonl")
    assert result["calls"] == 16 and result["errors"] == 0
    assert len(frozen) == 4
    assert len(experiment.retrieved) == 6
    assert [q for q, _ in experiment.retrieved] == ["Primera", "Primera", "Segunda"] * 2
    assert all(kw == dict(backend=kw["backend"], k=5, chunker="tok256", encoder="e5small",
                          filters=None, one_per_doc=True) for _, kw in experiment.retrieved)
    assert [event for event in experiment.events if event[0] == "answer"] == [
        ("answer", model, backend, query) for model in ("m1", "m2") for _ in range(2)
        for backend in ("milvus", "solr") for query in ("Primera", "Segunda")]
    for model in ("m1", "m2"):
        rows = [r for r in records if r["model"] == model]
        assert [r["cold"] for r in rows] == [True] + [False] * 7
        index = experiment.events.index(("answer", model, "milvus", "Primera"))
        other = "m2" if model == "m1" else "m1"
        assert experiment.events[index - 3:index] == [
            ("unload", "ajeno"), ("unload", other), ("unload", model)]
    for evidence in frozen:
        rows = [r for r in records if (r["qid"], r["backend"]) ==
                (evidence["qid"], evidence["backend"])]
        assert len({r["context_sha256"] for r in rows}) == 1
        assert all(r["retrieval_ms_original"] == evidence["retrieval_ms"] for r in rows)
    assert all(kw["evidence"] == next(row["chunks"] for row in frozen
               if row["backend"] == kw["backend"] and row["chunks"][0]["text"] == query)
               for query, kw in experiment.answered)
    assert all(kw["prompt_version"] == "rag_v3" and kw["temperature"] == 0
               for _, kw in experiment.answered)


def test_error_is_flushed_and_does_not_stop(experiment, monkeypatch):
    experiment.fail_once = True
    fake_answer = runner.answer

    def inspect_previous(*args, **kwargs):
        files = [p for p in experiment.out.glob("*.jsonl") if "evidence" not in p.name]
        if experiment.answered:
            assert len(read_jsonl(files[0])) == len(experiment.answered)
        return fake_answer(*args, **kwargs)

    monkeypatch.setattr(runner, "answer", inspect_previous)
    result = experiment.execute()
    records = read_jsonl(Path(result["file"]))
    assert result["calls"] == 16 and result["errors"] == 1
    assert records[0]["error"] == "generacion fallida"
    assert records[0]["cold"] is True and records[1]["cold"] is False
    assert all("error" not in row for row in records[1:])


def test_resume_retries_errors_and_reuses_evidence(experiment):
    experiment.fail_once = True
    first = experiment.execute()
    frozen_bytes = (experiment.out / "ensayo_evidence.jsonl").read_bytes()
    experiment.retrieved.clear()
    experiment.answered.clear()
    second = experiment.execute("--resume")
    assert second["calls"] == 1 and second["errors"] == 0
    assert len(experiment.answered) == 1 and not experiment.retrieved
    rows = read_jsonl(Path(first["file"]))
    assert len(rows) == 17 and rows[-1]["cold"] is True
    assert frozen_bytes == (experiment.out / "ensayo_evidence.jsonl").read_bytes()
    assert experiment.execute("--resume")["calls"] == 0


def test_resume_reads_all_dates_and_only_exact_tag(experiment):
    experiment.execute()
    logs = [p for p in experiment.out.glob("*.jsonl") if "evidence" not in p.name]
    rows = read_jsonl(logs[0])
    logs[0].unlink()
    for name, records in [("ensayo_20260101_0101.jsonl", rows[:8]),
                          ("ensayo_20260102_0101.jsonl", rows[8:15]),
                          ("ensayo_otro_20260101_0101.jsonl", rows[15:])]:
        (experiment.out / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    experiment.answered.clear()
    assert experiment.execute("--resume")["calls"] == 1
    assert experiment.answered[0][1]["model"] == "m2"


def test_resume_retries_exception_with_empty_message(experiment, monkeypatch):
    real_answer = runner.answer

    def fail(*args, **kwargs):
        raise RuntimeError()

    monkeypatch.setattr(runner, "answer", fail)
    assert experiment.execute()["errors"] == 16
    monkeypatch.setattr(runner, "answer", real_answer)
    assert experiment.execute("--resume")["calls"] == 16


def test_resume_after_interrupted_record(experiment):
    result = experiment.execute()
    path = Path(result["file"])
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + '\n{"model":', encoding="utf-8")
    assert experiment.execute("--resume")["calls"] == 1
    assert len(read_jsonl(path)) == 16
    assert experiment.execute("--resume")["calls"] == 0


def test_partial_evidence_is_reused_and_completed(experiment):
    experiment.execute()
    path = experiment.out / "ensayo_evidence.jsonl"
    rows = read_jsonl(path)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]) + '{"qid":', encoding="utf-8")
    experiment.retrieved.clear()
    assert experiment.execute("--resume")["calls"] == 0
    assert len(experiment.retrieved) == 2
    assert len(read_jsonl(path)) == 4


@pytest.mark.parametrize("kind", ["calls", "evidence"])
def test_resume_after_interrupted_utf8_character(experiment, kind):
    result = experiment.execute()
    path = (experiment.out / "ensayo_evidence.jsonl") if kind == "evidence" else Path(result["file"])
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]) + b'{"text":"\xc3')
    assert experiment.execute("--resume")["calls"] == (0 if kind == "evidence" else 1)
    assert len(read_jsonl(path)) == (4 if kind == "evidence" else 16)
    assert experiment.execute("--resume")["calls"] == 0


def test_evidence_remains_frozen_if_consumer_mutates(experiment):
    experiment.mutate = True
    result = experiment.execute()
    records = read_jsonl(Path(result["file"]))
    for qid in ("Q01", "Q02"):
        for backend in ("milvus", "solr"):
            assert len({r["context_sha256"] for r in records
                        if r["qid"] == qid and r["backend"] == backend}) == 1


def test_qids_config_and_call_metadata(experiment):
    result = experiment.execute("--qids", "Q02", "--k", "3", "--temperature", "0.2")
    rows = read_jsonl(Path(result["file"]))
    assert result["calls"] == 8 and {r["qid"] for r in rows} == {"Q02"}
    effective = json.loads((experiment.out / "ensayo_config.json").read_text(encoding="utf-8"))
    assert {"gold", "split", "backends", "models", "runs", "k", "chunker", "encoder",
            "prompt", "tag", "one_per_doc", "qids", "out_dir", "resume", "temperature"} <= effective.keys()
    assert effective["prompt_version"].startswith("rag_v3@")
    assert effective["MAX_CONTEXT_CHARS"] == runner.config.MAX_CONTEXT_CHARS
    assert effective["inference_params"] == dict(temperature=0.2, seed=42, num_ctx=4096,
                                                  num_predict=384, keep_alive="10m")
    assert effective["model_info"] == {"m1": {"digest": "m1"}, "m2": {"digest": "m2"}}
    assert effective["qids_used"] == ["Q02"] and effective["seed"] == 42
    assert datetime.fromisoformat(effective["date"]).utcoffset().total_seconds() == 0
    assert experiment.hardware_calls == experiment.version_calls == 1
    for row in rows:
        assert row["split"] == "test" and row["type"] == "sintesis" and row["tag"] == "ensayo"
        assert set(row["hardware"]) == {"cpu", "ram_gb", "gpu", "vram_gb"}
        assert row["versions"] == effective["versions"] and "tokens" in row
        assert row["retrieval_ms_original"] >= 0
        assert datetime.fromisoformat(row["timestamp"]).utcoffset().total_seconds() == 0


def test_main_prints_summary_and_progress(experiment, capsys, caplog):
    with caplog.at_level("INFO"):
        assert runner.main(experiment.argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert set(summary) == {"tag", "calls", "errors", "seconds", "file"}
    assert summary["seconds"] >= 0 and summary["tag"] == "ensayo"
    assert any("10/16" in r.message and "%" in r.message and "ETA" in r.message for r in caplog.records)


@pytest.mark.parametrize("extra", [("--runs", "0"), ("--k", "0"),
                                   ("--qids", "Q99"), ("--tag", "../otro")])
def test_invalid_arguments_fail_before_services(experiment, extra):
    with pytest.raises(ValueError):
        experiment.execute(*extra)
    assert experiment.hardware_calls == 0 and not experiment.retrieved


def test_hardware_with_fake_proc_and_gpu(monkeypatch):
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path.as_posix() == "/proc/cpuinfo":
            return "processor : 0\nmodel name : CPU ejemplo\n"
        if path.as_posix() == "/proc/meminfo":
            return "MemTotal: 16777216 kB\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)

    def command(args, **kwargs):
        assert args[:6] == ["docker", "compose", "exec", "-T", "ollama", "nvidia-smi"]
        assert kwargs["timeout"] > 0
        return SimpleNamespace(returncode=0, stdout="GPU ejemplo, 8192\n")

    monkeypatch.setattr(subprocess, "run", command)
    assert runner.collect_hardware() == dict(cpu="CPU ejemplo", ram_gb=16, gpu="GPU ejemplo", vram_gb=8)


def test_hardware_and_versions_unavailable_are_null(monkeypatch):
    original = Path.read_text

    def unavailable(*args, **kwargs):
        raise OSError("no disponible")

    def read(path, *args, **kwargs):
        if path.as_posix().startswith("/proc/"):
            return unavailable()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(subprocess, "run", unavailable)
    monkeypatch.setattr(requests, "get", unavailable)
    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(MilvusClient=unavailable))
    assert runner.collect_hardware() == dict(cpu=None, ram_gb=None, gpu=None, vram_gb=None)
    versions = runner.collect_versions()
    assert versions["ollama"] is versions["milvus"] is versions["solr"] is None
    assert versions["python"] and set(versions["packages"]) == set(runner.PACKAGES)


def test_versions_with_fake_http_and_milvus(monkeypatch):
    urls, closed = [], []

    def get(url, **kwargs):
        urls.append(url)
        assert kwargs["timeout"] > 0
        payload = {"version": "0.40"} if url.endswith("/api/version") else {"lucene": {"solr-spec-version": "9.10"}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    class Client:
        def __init__(self, *, uri, timeout):
            assert uri == runner.config.MILVUS_URI and timeout > 0

        def get_server_version(self, **kwargs):
            return "2.6"

        def close(self):
            closed.append(True)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setitem(sys.modules, "pymilvus", SimpleNamespace(MilvusClient=Client))
    versions = runner.collect_versions()
    assert [versions[k] for k in ("ollama", "solr", "milvus")] == ["0.40", "9.10", "2.6"]
    assert urls[1].endswith("/solr/admin/info/system") and closed == []


@pytest.mark.parametrize("text,expected", [
    ("rag-api-1,12.50%,256MiB / 2GiB\nrag-ollama-1,99%,1GiB / 4GiB\nrag-solr-1,2%,1MB / 2GB\n",
     [("rag-api-1", 12.5, 256, 2048), ("rag-ollama-1", 99, 1024, 4096)]),
    ("api,0%,1GB / 2GB\n", [("api", 0, 1e9 / 2**20, 2e9 / 2**20)]),
    ("", []), ("texto incorrecto\n", []),
])
def test_parse_docker_stats(text, expected):
    rows = sysmon.parse_docker_stats(text)
    assert [(r["container"], r["cpu_pct"], r["mem_used_mb"], r["mem_limit_mb"]) for r in rows] == expected


@pytest.mark.parametrize("text,expected", [
    ("1024, 8192, 25\n", dict(gpu_mem_used_mb=1024, gpu_mem_total_mb=8192, gpu_util_pct=25)),
    ("", dict(gpu_mem_used_mb="", gpu_mem_total_mb="", gpu_util_pct="")),
    ("[N/A], [N/A], [N/A]\n", dict(gpu_mem_used_mb="", gpu_mem_total_mb="", gpu_util_pct="")),
])
def test_parse_nvidia_smi(text, expected):
    assert sysmon.parse_nvidia_smi(text) == expected


def test_monitor_failed_commands_and_duration(monkeypatch, tmp_path):
    import csv

    count = []

    def fail(args, **kwargs):
        count.append(args)
        raise OSError("docker ausente")

    clock = [0.0]
    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(sysmon.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sysmon.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    out = tmp_path / "monitor.csv"
    assert sysmon.main(["--out", str(out), "--interval", "2", "--duration", "3"]) == 0
    with out.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2 and len(count) == 4 and clock[0] == 3
    assert all(row["timestamp"] and all(value == "" for key, value in row.items()
                                      if key != "timestamp") for row in rows)


def test_monitor_interrupt_and_successful_sample(monkeypatch, tmp_path):
    import csv

    commands = []

    def command(args, **kwargs):
        commands.append(args)
        text = "api,3%,512MiB / 2GiB\n" if args[1] == "stats" else "100, 8000, 20\n"
        return SimpleNamespace(returncode=0, stdout=text)

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", command)
    monkeypatch.setattr(sysmon.time, "sleep", interrupt)
    out = tmp_path / "monitor.csv"
    assert sysmon.main(["--out", str(out)]) == 0
    with out.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1 and rows[0]["gpu_util_pct"] == "20.0"
    assert rows[0]["container"] == "api" and rows[0]["mem_used_mb"] == "512.0"
    assert commands[0] == ["docker", "stats", "--no-stream", "--format", "{{.Name}},{{.CPUPerc}},{{.MemUsage}}"]
    assert commands[1][:6] == ["docker", "compose", "exec", "-T", "ollama", "nvidia-smi"]


def test_monitor_nonzero_commands_ignore_stdout(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout="api,3%,512MiB / 2GiB\n"))
    rows = sysmon.sample()
    assert len(rows) == 1 and rows[0]["timestamp"]
    assert all(value == "" for key, value in rows[0].items() if key != "timestamp")


def test_frozen_evidence_with_other_params_is_rejected(experiment):
    experiment.execute()
    with pytest.raises(ValueError, match="otros parámetros"):
        experiment.execute("--k", "3")


def test_unload_failure_does_not_abort(experiment, monkeypatch):
    def unload(model):
        raise RuntimeError("modelo no descargado")
    monkeypatch.setattr(runner, "unload", unload)
    result = experiment.execute()
    assert result["calls"] == 16 and result["errors"] == 0


def test_resume_preserves_original_config(experiment):
    experiment.execute()
    original = experiment.out / "ensayo_config.json"
    before = original.read_bytes()
    experiment.execute("--resume")
    assert original.read_bytes() == before
    resumed = list(experiment.out.glob("ensayo_config_resume_????????_????.json"))
    assert len(resumed) == 1
    assert json.loads(resumed[0].read_text(encoding="utf-8"))["resume"] is True
