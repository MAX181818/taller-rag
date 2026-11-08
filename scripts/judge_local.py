"""Juez local complementario (LLM en Ollama); no sustituye la evaluación humana."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score_generation import load_raw, read_jsonl

CRITERIA = ("correccion", "fidelidad", "relevancia")


def generate(*args, **kwargs):
    from app.generation import generate as implementation
    return implementation(*args, **kwargs)


def model_info(model):
    from app.generation import model_info as implementation
    return implementation(model)


def build_prompt(record, gold):
    evidence = "\n".join(f"[{e['n']}] {e.get('text', '')}" for e in record.get("evidence") or [])
    return (
        "Eres un evaluador estricto de sistemas de preguntas y respuestas. Califica la respuesta "
        "evaluada con 0, 1 o 2 en cada criterio:\n"
        "- correccion: coincide con la respuesta de referencia y el criterio.\n"
        "- fidelidad: todo lo afirmado está respaldado por la evidencia.\n"
        "- relevancia: responde a lo que se preguntó.\n\n"
        f"Pregunta: {gold.get('question') or record.get('query')}\n"
        f"Respuesta de referencia: {gold.get('reference_answer')}\n"
        f"Criterio: {gold.get('answer_criterion')}\n\n"
        f"Evidencia:\n{evidence}\n\n"
        f"Respuesta evaluada: {record.get('answer')}\n\n"
        'Responde SOLO con un JSON: {"correccion": 0|1|2, "fidelidad": 0|1|2, '
        '"relevancia": 0|1|2, "justificacion": "<máx 30 palabras>"}')


def parse_judge(text):
    """Primer objeto JSON válido con los tres puntajes en 0..2; None si no hay."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(data, dict) and all(data.get(c) in (0, 1, 2) and not isinstance(data.get(c), bool)
                                          for c in CRITERIA):
            return data
    return None


def judge_record(record, gold, judge_model, digest, generate=generate):
    prompt = build_prompt(record, gold)
    raw = ""
    for _ in range(2):  # un reintento si no se puede parsear
        raw = generate(judge_model, prompt, temperature=0, seed=42, num_predict=200)["answer"]
        parsed = parse_judge(raw)
        if parsed:
            break
    out = dict(run_id=record["run_id"], qid=record["qid"], model=record["model"],
               backend=record["backend"], judge_model=judge_model, judge_digest=digest,
               tipo="complementario", timestamp=datetime.now(timezone.utc).isoformat())
    if parsed:
        out.update({c: parsed[c] for c in CRITERIA}, justificacion=str(parsed.get("justificacion", "")))
    else:
        out.update(parse_error=True, raw=raw)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default="qwen2.5:7b")
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--run-idx", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("reports/generation/judge.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=Path("reports/runs/raw"))
    parser.add_argument("--gold", type=Path, default=Path("eval/gold.jsonl"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    golds = {g["qid"]: g for g in read_jsonl(args.gold)}
    records = [r for r in load_raw(args.tags, args.raw_dir)
               if r.get("run_idx") == args.run_idx and not r.get("error")]
    done = {r["run_id"] for r in read_jsonl(args.out)} if args.resume and args.out.exists() else set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    digest = model_info(args.judge).get("digest", "")
    with args.out.open("a" if args.resume else "w", encoding="utf-8", newline="\n") as target:
        for record in records:
            if record["run_id"] in done:
                continue
            line = judge_record(record, golds[record["qid"]], args.judge, digest)
            target.write(json.dumps(line, ensure_ascii=False) + "\n")
            target.flush()


if __name__ == "__main__":
    main()
