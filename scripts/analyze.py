"""Genera tablas, figuras y resultados a partir de archivos locales."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.citations import is_abstention
from app.metrics import bootstrap_ci
from app.privacy import mask_handles


QUALITY = ("abstention_correct", "cites_relevant", "citations_valid_rate")
LABELS = {"abstention_correct": "Abstención correcta", "cites_relevant": "Cita relevante",
          "citations_valid_rate": "Validez de citas", "correccion": "Corrección",
          "fidelidad": "Fidelidad", "relevancia": "Relevancia"}
BACKEND_LABELS = {"solr": "Solr", "milvus": "Milvus", "hybrid": "Híbrido"}
TYPE_LABELS = {"inyeccion": "inyección", "sin_respuesta": "sin respuesta"}
PRINCIPAL = {f"main_{engine}_tok256_e5small" for engine in ("solr", "milvus", "hybrid")}


def notice(message):
    print(f"Aviso: {message}", file=sys.stderr)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def read_rows(path, warn=True):
    path = Path(path)
    if not path.exists():
        if warn:
            notice(f"falta {path.as_posix()}; se omite su contenido.")
        return []
    with path.open(encoding="utf-8-sig", newline="") as source:
        if path.suffix == ".csv":
            return list(csv.DictReader(source))
        rows = []
        for index, line in enumerate(source, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    notice(f"registro incompleto en {path.as_posix()}, línea {index}; se omite.")
        return rows


def read_json(path):
    path = Path(path)
    if not path.exists():
        notice(f"falta {path.as_posix()}; se omite su contenido.")
        return None
    with path.open(encoding="utf-8-sig") as source:
        return json.load(source)


def sourced(rows, path):
    return [dict(row, source=Path(path).as_posix()) for row in rows]


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from flatten(value[key], f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{index}]")
    elif value is not None:
        yield {"campo": prefix, "valor": value}


def formatted(value, field=""):
    if value is None:
        return ""
    if field in ("digest", "model", "backend", "qid", "run_id", "parameter_size", "quantization_level", "family", "tag", "condition", "group", "type"):
        return str(value)
    numeric = number(value)
    return f"{numeric:.3f}" if numeric is not None and not isinstance(value, bool) else str(value)


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
                    "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def display_rows(rows):
    result = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if key in ("ci_low", "ci_high") or key.endswith(("_ci_low", "_ci_high")):
                continue
            item[key] = formatted(value, key)
            prefix = key[:-5] if key.endswith("_mean") else "" if key == "mean" else None
            if prefix is not None:
                low_key, high_key = (prefix + "_ci_low", prefix + "_ci_high") if prefix else ("ci_low", "ci_high")
                low, high = number(row.get(low_key)), number(row.get(high_key))
                if low is not None and high is not None:
                    item[prefix + "_ic" if prefix else "ic"] = f"[{low:.3f}, {high:.3f}]"
        result.append(item)
    return result


def write_table(name, rows, reports, tex):
    if not rows:
        notice(f"sin datos para la tabla {name}; se omite.")
        return
    rows = display_rows(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (reports / "tables" / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [r"\begin{tabular}{" + "l" * len(fields) + "}", r"\toprule",
             " & ".join(map(tex_escape, fields)) + r" \\", r"\midrule"]
    lines.extend(" & ".join(tex_escape(row.get(f, "")) for f in fields) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tex / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def backend(row):
    return row.get("backend") or (row.get("config") or {}).get("backend") or ""


def select_cases(records):
    """Escoge casos por evidencia explícita y en orden estable."""
    reasons = {
        "answered_without_evidence": "Respuesta sin citas que no se abstiene",
        "fidelidad_cero": "Fidelidad nula del juez; posible alucinación",
        "abstencion_correcta_sin_respuesta": "Abstención correcta en pregunta sin respuesta",
        "abstencion_incorrecta": "Abstención incorrecta en pregunta con respuesta",
        "discrepancia_modelos": "Los modelos discrepan en la cita relevante",
        "injection_success": "Éxito de inyección registrado",
    }
    groups = defaultdict(list)
    for row in records:
        if not row.get("error") or row.get("error") == "0":
            groups[(row.get("qid"), backend(row))].append(row)
    out = {key: [] for key in reasons}
    order = lambda r: (str(r.get("qid", "")), str(r.get("model", "")), str(backend(r)),
                       str(r.get("run_id", "")), json.dumps(r, sort_keys=True, ensure_ascii=False))
    for row in sorted(records, key=order):
        if row.get("error") and row.get("error") != "0":
            continue
        automatic = row.get("automatic") or {}
        judge = row.get("judge") or {}
        correct = number(automatic.get("abstention_correct"))
        expected = row.get("type") == "sin_respuesta"
        abstained = row.get("abstained")
        if abstained is None:
            abstained = (correct == 0 and not expected) or is_abstention(row.get("answer") or "")
        peers = groups[(row.get("qid"), backend(row))]
        hit = number(automatic.get("cites_relevant"))
        discrepancy = hit in (0, 1) and any(
            peer.get("model") != row.get("model") and
            number((peer.get("automatic") or {}).get("cites_relevant")) == 1 - hit for peer in peers)
        flags = {
            "answered_without_evidence": number(automatic.get("answered_without_evidence")) == 1,
            "fidelidad_cero": number(judge.get("fidelidad")) == 0,
            "abstencion_correcta_sin_respuesta": expected and correct == 1,
            "abstencion_incorrecta": bool(row.get("type")) and not expected and correct == 0 and abstained,
            "discrepancia_modelos": discrepancy,
            "injection_success": number(automatic.get("injection_success")) == 1,
        }
        for category, matches in flags.items():
            if matches and len(out[category]) < 5:
                source = row.get("source", "")
                if category == "fidelidad_cero" and row.get("judge_source"):
                    source = "; ".join(filter(None, (source, row["judge_source"])))
                out[category].append(dict(qid=row.get("qid", ""), model=row.get("model", ""),
                    backend=backend(row), answer=mask_handles(row.get("answer") or "")[:300],
                    reason=reasons[category], source=source or row.get("source", "")))
    return out


def case_records(reports, annotated, calls, judges):
    """Enlaza evaluaciones por llamada sin confundir repeticiones."""
    scores = {r["run_id"]: r for r in calls if r.get("run_id")}
    judgment = {r["run_id"]: r for r in judges if r.get("run_id")}
    records = []
    for original in annotated:
        row = dict(original)
        score = scores.get(row.get("run_id"), {})
        row["backend"] = backend(row) or score.get("backend", "")
        row["automatic"] = {**(row.get("automatic") or {}), **{k: v for k, v in score.items() if k in (*QUALITY, "injection_success", "answered_without_evidence", "abstention_with_citation")}}
        row["judge"] = judgment.get(row.get("run_id")) or row.get("judge")
        row["source"] = (reports / "generation/annotated_runs.jsonl").as_posix()
        if row.get("run_id") in judgment:
            row["judge_source"] = (reports / "generation/judge.jsonl").as_posix()
        if score:
            row["source"] += "; " + (reports / "generation/scores_per_call.csv").as_posix()
        records.append(row)
    return records


def judge_summary(rows, source):
    groups = defaultdict(list)
    for row in rows:
        if not row.get("parse_error"):
            groups[(row.get("model", ""), backend(row))].append(row)
    result = []
    for (model, engine), records in sorted(groups.items()):
        item = dict(model=model, backend=engine, source=source)
        for metric in ("correccion", "fidelidad", "relevancia"):
            values = [v for v in (number(r.get(metric)) for r in records) if v is not None]
            if values:
                low, high = bootstrap_ci(values)
                item.update({metric + "_mean": sum(values) / len(values), metric + "_ci_low": low,
                             metric + "_ci_high": high, metric + "_n": len(values)})
        if len(item) > 3:
            result.append(item)
    return result


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo else None
    except (TypeError, ValueError, OverflowError):
        return None


def performance_rows(reports, latencies=None):
    reports = Path(reports)
    if latencies is None:
        latencies = read_rows(reports / "generation/latency_summary.csv", warn=False)
    weighted = defaultdict(list)
    for row in latencies:
        if row.get("tag") == "main" and row.get("metric") == "generation_ms":
            speed, n = number(row.get("tokens_per_s_mean")), number(row.get("n"))
            if speed is not None and n is not None and n > 0:
                weighted[row["model"]].append((speed, n))
    raw = []
    for path in sorted((reports / "runs/raw").glob("*.jsonl")):
        if re.fullmatch(r".+_\d{8}_\d{4}\.jsonl", path.name):
            raw.extend(sourced(read_rows(path), path))
    intervals, fallback = [], defaultdict(list)
    for row in raw:
        if row.get("error"):
            continue
        model = row.get("model")
        end = timestamp(row.get("timestamp"))
        duration = number((row.get("timings") or {}).get("generation_ms"))
        if model and end is not None and duration is not None and duration > 0:
            intervals.append((end - duration / 1000, end, model))
        speed = number((row.get("tokens") or {}).get("tokens_per_s"))
        if model and speed is not None and row.get("tag", "main") == "main":
            fallback[model].append((speed, row["source"]))
    memory, memory_sources = defaultdict(list), defaultdict(set)
    for path in sorted((reports / "runs/raw").glob("sysmon_*.csv")):
        for row in read_rows(path):
            time, value = timestamp(row.get("timestamp")), number(row.get("gpu_mem_used_mb"))
            if time is None or value is None:
                continue
            models = {model for start, end, model in intervals if start <= time <= end}
            if len(models) == 1:
                model = models.pop()
                memory[model].append(value)
                memory_sources[model].add(path.as_posix())
    result = []
    for model in sorted(set(weighted) | set(fallback)):
        if weighted[model]:
            speed = sum(v * n for v, n in weighted[model]) / sum(n for _, n in weighted[model])
            source = (reports / "generation/latency_summary.csv").as_posix()
        else:
            speed = sum(v for v, _ in fallback[model]) / len(fallback[model])
            source = "; ".join(sorted({p for _, p in fallback[model]}))
        item = dict(model=model, tokens_per_s=speed, source=source)
        if memory[model]:
            item.update(vram_max_mb=max(memory[model]), vram_source="; ".join(sorted(memory_sources[model])),
                        interval_source="; ".join(sorted({r["source"] for r in raw if r.get("model") == model})))
        result.append(item)
    return result


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def errorbars(ax, x, rows, mean, low, high, **kwargs):
    """Traza cada estimación conservando los IC que sí existen."""
    points = [(pos, row, number(row.get(mean))) for pos, row in zip(x, rows)]
    points = [(pos, row, value) for pos, row, value in points if value is not None]
    if not points:
        return
    xs, ys = [p[0] for p in points], [p[2] for p in points]
    line = ax.plot(xs, ys, marker="o", **kwargs)[0]
    for pos, row, value in points:
        lo, hi = number(row.get(low)), number(row.get(high))
        if lo is not None and hi is not None and lo <= value <= hi:
            ax.errorbar([pos], [value], yerr=[[value - lo], [hi - value]], fmt="none", capsize=3, color=line.get_color())


def grouped_figure(rows, metrics, path, title, ylabel="Proporción"):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.2), 5))
    x = np.arange(len(rows))
    for index, metric in enumerate(metrics):
        errorbars(ax, x + (index - (len(metrics) - 1) / 2) * .18, rows,
                  metric + "_mean", metric + "_ci_low", metric + "_ci_high",
                  linestyle="none", label=LABELS.get(metric, metric))
    ax.set_xticks(x, [f"{r.get('model', '')} × {backend(r)}" for r in rows], rotation=30, ha="right")
    ax.set(title=title, xlabel="Modelo y motor", ylabel=ylabel)
    if ax.lines:
        ax.legend()
    save_figure(fig, path)


def figures(reports, metrics, scores, by_type, latency, judges, performance):
    out = reports / "figures"
    principal = [r for r in metrics if r.get("condition") in PRINCIPAL and number(r.get("k")) in (1, 3, 5, 10) and r.get("metric") in ("recall", "ndcg") and number(r.get("mean")) is not None]
    if principal:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric, title in zip(axes, ("recall", "ndcg"), ("Recall según k", "nDCG según k")):
            for condition in sorted({r["condition"] for r in principal}):
                rows = sorted([r for r in principal if r["condition"] == condition and r["metric"] == metric], key=lambda r: float(r["k"]))
                engine = condition.split("_")[1]
                errorbars(ax, [float(r["k"]) for r in rows], rows, "mean", "ci_low", "ci_high", label=BACKEND_LABELS.get(engine, engine))
            ax.set(title=title, xlabel="Cantidad de resultados (k)", ylabel="Proporción", xticks=[1, 3, 5, 10])
            if ax.lines:
                ax.legend(title="Motor")
        save_figure(fig, out / "recuperacion_por_k.png")
    ablations = [r for r in metrics if r.get("condition", "").startswith(("ablations_", "abl_")) and r.get("metric") == "recall" and number(r.get("k")) == 5 and number(r.get("mean")) is not None]
    if ablations:
        rows = [r for r in principal if r["metric"] == "recall" and number(r["k"]) == 5] + ablations
        fig, ax = plt.subplots(figsize=(max(8, len(rows)), 5))
        errorbars(ax, range(len(rows)), rows, "mean", "ci_low", "ci_high", linestyle="none")
        ax.set_xticks(range(len(rows)), [r["condition"] for r in rows], rotation=35, ha="right")
        ax.set(title="Ablaciones frente al experimento principal", xlabel="Condición", ylabel="Recall")
        save_figure(fig, out / "ablaciones_recuperacion.png")
    main = [r for r in scores if r.get("tag") == "main"]
    grouped_figure(main, QUALITY, out / "generacion_por_modelo.png", "Calidad de generación por modelo y motor")
    grouped_figure(judges, ("correccion", "fidelidad", "relevancia"), out / "juez_por_modelo.png", "Evaluación complementaria del juez", "Puntaje medio")
    quality = {(r.get("model"), backend(r)): number(r.get("cites_relevant_mean")) for r in main}
    warm = [r for r in latency if r.get("tag") == "main" and r.get("metric") == "generation_ms" and str(r.get("cold")).lower() in ("false", "0") and number(r.get("p95")) is not None and quality.get((r.get("model"), backend(r))) is not None]
    if warm:
        fig, ax = plt.subplots(figsize=(8, 5))
        offsets = ((0, 12, "center", "bottom"), (0, -12, "center", "top"),
                   (-12, 0, "right", "center"), (12, 0, "left", "center"))
        for index, row in enumerate(warm):
            x, y = number(row["p95"]), quality[(row.get("model"), backend(row))]
            ax.scatter(x, y)
            dx, dy, ha, va = offsets[index % len(offsets)]
            ax.annotate(f"{row['model']} × {BACKEND_LABELS.get(backend(row), backend(row))}",
                        (x, y), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=9)
        ax.set(title="Latencia y calidad de generación en caliente", xlabel="Latencia p95 (ms)", ylabel="Tasa de cita relevante")
        save_figure(fig, out / "latencia_vs_calidad.png")
    if performance:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(performance)), [r["tokens_per_s"] for r in performance], width=.5, label="Velocidad")
        ax.set_xticks(range(len(performance)), [r["model"] for r in performance], rotation=25, ha="right")
        ax.set(title="Velocidad de generación y memoria observada", xlabel="Modelo", ylabel="Tokens por segundo")
        available = [(i, r["vram_max_mb"]) for i, r in enumerate(performance) if "vram_max_mb" in r]
        if available:
            second = ax.twinx()
            second.plot([i for i, _ in available], [v for _, v in available], "o", color="tab:red")
            second.set_ylabel("VRAM máxima observada (MB)")
        else:
            notice("VRAM sin asignación temporal; se muestra solo tokens/s.")
        save_figure(fig, out / "tokens_vram.png")
    types = [r for r in by_type if r.get("tag") == "main" and number(r.get("abstention_correct_mean")) is not None]
    if types:
        models = sorted({r["model"] for r in types})
        kinds = sorted({r["type"] for r in types})
        matrix = np.full((len(kinds), len(models)), np.nan)
        for i, kind in enumerate(kinds):
            for j, model in enumerate(models):
                rows = [r for r in types if r["type"] == kind and r["model"] == model and (number(r.get("n")) or 0) > 0]
                if rows:
                    matrix[i, j] = sum(float(r["abstention_correct_mean"]) * float(r["n"]) for r in rows) / sum(float(r["n"]) for r in rows)
        fig, ax = plt.subplots(figsize=(max(6, len(models)), max(4, len(kinds) * .6)))
        view = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(models)), models, rotation=30, ha="right")
        ax.set_yticks(range(len(kinds)), [TYPE_LABELS.get(kind, kind) for kind in kinds])
        ax.set(title="Abstención correcta por tipo de pregunta", xlabel="Modelo (motores combinados por tamaño de muestra)", ylabel="Tipo de pregunta")
        fig.colorbar(view, ax=ax, label="Tasa de abstención correcta")
        save_figure(fig, out / "errores_por_tipo.png")


def markdown_value(value, field, row):
    numeric = number(value)
    if numeric is None or isinstance(value, bool):
        return formatted(value, field)
    leaf = field.rsplit(".", 1)[-1]
    if leaf in ("k", "n", "rows", "count", "total") or leaf.startswith(("n_", "count_")) or leaf.endswith(("_n", "_count", "_chunks")):
        return f"{numeric:.0f}"
    if "_ms" in field or (str(row.get("metric", "")).endswith("_ms") and field in ("p50", "p95", "p99", "mean", "std", "min", "max")):
        return f"{numeric:.1f}"
    return formatted(value, field)


def markdown_table(lines, rows, columns=None):
    """Agrupa por procedencia para citar una sola vez cada tabla."""
    batches = defaultdict(list)
    for row in rows:
        sources = tuple(dict.fromkeys(str(row[key]) for key in row
                        if (key == "source" or key.endswith("_source")) and row[key]))
        batches[sources].append(row)
    safe = lambda value: str(value).replace("\r", " ").replace("\n", " ").replace("|", "&#124;").replace("`", "'")
    for sources, batch in batches.items():
        fields = list(dict.fromkeys(key for row in batch for key in row
                      if key != "source" and not key.endswith(("_source", "_ci_low", "_ci_high"))
                      and key not in ("ci_low", "ci_high")))
        specs = columns or [(key, key) for key in fields]
        lines.append("| " + " | ".join(label for _, label in specs) + " |")
        lines.append("| " + " | ".join("---" for _ in specs) + " |")
        for row in batch:
            cells = []
            for key, _ in specs:
                if key == "ic":
                    low, high = row.get("ci_low"), row.get("ci_high")
                    cell = f"[{markdown_value(low, 'mean', row)}, {markdown_value(high, 'mean', row)}]" if number(low) is not None and number(high) is not None else ""
                else:
                    field = row.get("campo", key) if key == "valor" else key
                    cell = markdown_value(row.get(key, ""), field, row)
                    prefix = key[:-5] if key.endswith("_mean") else "" if key == "mean" else None
                    if prefix is not None and columns is None:
                        low_key, high_key = (prefix + "_ci_low", prefix + "_ci_high") if prefix else ("ci_low", "ci_high")
                        low, high = row.get(low_key), row.get(high_key)
                        if number(low) is not None and number(high) is not None:
                            cell += f" [{markdown_value(low, key, row)}, {markdown_value(high, key, row)}]"
                cells.append(safe(cell))
            lines.append("| " + " | ".join(cells) + " |")
        lines.extend(["", "Fuente: " + "; ".join(f"`{source}`" for source in sources), ""])


def report_section(lines, title, rows):
    if not rows:
        return
    lines.extend(["", f"## {title}", ""])
    markdown_table(lines, rows)


def analyze(reports, tex):
    reports, tex = Path(reports), Path(tex)
    for path in (reports, reports / "tables", reports / "figures", tex):
        path.mkdir(parents=True, exist_ok=True)
    def load(relative):
        return sourced(read_rows(reports / relative), reports / relative)
    metrics = load("retrieval/metrics.csv")
    groups = load("retrieval/metrics_by_group.csv")
    per_question = load("retrieval/per_question.csv")
    retrieval_latency = load("retrieval/latency.csv")
    scores = load("generation/scores_summary.csv")
    by_type = load("generation/scores_by_type.csv")
    latency = load("generation/latency_summary.csv")
    calls = load("generation/scores_per_call.csv")
    annotated = read_rows(reports / "generation/annotated_runs.jsonl")
    judges_raw = read_rows(reports / "generation/judge.jsonl")
    judges = judge_summary(judges_raw, (reports / "generation/judge.jsonl").as_posix())
    human = load("generation/human_eval.csv")
    tables = {
        "recuperacion_principal": [r for r in metrics if r.get("condition") in PRINCIPAL],
        "recuperacion_ablaciones": [r for r in metrics if r.get("condition", "").startswith(("ablations_", "abl_"))],
        "recuperacion_filtros": [r for r in metrics if r.get("condition", "").startswith("filters_")],
        "recuperacion_10k": [r for r in metrics if r.get("condition", "").startswith("10k_")],
        "latencia_recuperacion": retrieval_latency,
        "generacion_principal": [r for r in scores if r.get("tag") == "main"],
        "generacion_ablaciones": [r for r in scores if r.get("tag", "").startswith("abl_")],
        "latencia_generacion": [dict(r, phase="cold" if str(r.get("cold")).lower() in ("true", "1") else "warm") for r in latency if str(r.get("cold")).lower() in ("true", "false", "1", "0")],
        "juez": judges, "humana": human,
    }
    models = []
    config_paths = sorted((reports / "runs/raw").glob("*_config.json"))
    if not config_paths:
        notice("faltan configuraciones de modelos; se omite su tabla.")
    for path in config_paths:
        config = read_json(path) or {}
        for model, info in sorted((config.get("model_info") or {}).items()):
            models.append(dict(model=model, parameter_size=info.get("parameter_size", ""),
                quantization_level=info.get("quantization_level", ""), family=info.get("family", ""),
                digest=str(info.get("digest") or "")[:12], source=path.as_posix()))
    tables["modelos"] = models
    sensitivity = read_json(reports / "embeddings/sensibilidad.json")
    embedding_rows = sourced(list(flatten(sensitivity)), reports / "embeddings/sensibilidad.json") if sensitivity is not None else []
    tables["embeddings_sensibilidad"] = embedding_rows
    overlap = load("embeddings/solapamiento.csv")
    context = []
    for relative in ("perfilado/perfil.json", "chunking/stats.json"):
        data = read_json(reports / relative)
        if data is not None:
            context.extend(sourced(list(flatten(data)), reports / relative))
    for name, rows in tables.items():
        write_table(name, rows, reports, tex)
    performance = performance_rows(reports, latency)
    figures(reports, metrics, scores, by_type, latency, judges, performance)
    lines = ["# Resultados", "", "Valores observados y agregaciones de las fuentes indicadas. Los datos ausentes se omiten."]
    retrieval_groups = [(label, [r for r in tables[name]
                         if number(r.get("k")) in (1, 5, 10)
                         and r.get("metric") in ("hit", "recall", "mrr", "ndcg", "fragment_hit")])
                        for label, name in (("Principal", "recuperacion_principal"),
                                            ("Ablaciones", "recuperacion_ablaciones"),
                                            ("Filtros", "recuperacion_filtros"),
                                            ("10k", "recuperacion_10k"))]
    if any(rows for _, rows in retrieval_groups):
        lines.extend(["", "## Recuperación", ""])
        for label, rows in retrieval_groups:
            if rows:
                lines.extend([f"### {label}", ""])
                markdown_table(lines, rows, [("condition", "condición"), ("k", "k"),
                    ("metric", "métrica"), ("n", "n"), ("mean", "media"), ("ic", "IC 95 %")])
    report_section(lines, "Generación", scores + judges + human)
    report_section(lines, "Rendimiento", retrieval_latency + tables["latencia_generacion"] + performance + models + context)
    report_section(lines, "Embeddings", embedding_rows + overlap)
    robustness = []
    for row in scores:
        if number(row.get("injection_success_mean")) is not None:
            robustness.append({k: v for k, v in row.items() if k in ("tag", "model", "backend", "source") or k.startswith("injection_success_")})
    report_section(lines, "Robustez (inyección)", robustness)
    cases = select_cases(case_records(reports, annotated, calls, judges_raw))
    if any(cases.values()):
        lines.extend(["", "## Casos para análisis de errores", ""])
        for category, rows in cases.items():
            if rows:
                lines.extend([f"### {category}", ""])
                markdown_table(lines, rows, [("qid", "qid"), ("model", "modelo"),
                    ("backend", "motor"), ("answer", "respuesta"), ("reason", "razón")])
    (reports / "RESULTADOS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--tex", type=Path, default=Path("informe/tablas"))
    args = parser.parse_args(argv)
    analyze(args.reports, args.tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
