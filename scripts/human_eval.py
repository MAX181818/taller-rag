"""Hojas ciegas para evaluación humana y cálculo de acuerdo entre evaluadores."""
import argparse
import csv
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score_generation import load_raw, read_jsonl
from app.privacy import mask_handles
from scripts.validate_gold import weighted_kappa

CRITERIA = ("correccion", "fidelidad", "relevancia", "calidad_citas", "abstencion")
SHEET = ("item_id", "pregunta", "respuesta", "evidencia", "respuesta_referencia", "criterio",
         *CRITERIA, "comentario")
KEY = ("item_id", "run_id", "model", "backend", "qid")


def _write(path, fields, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _choose(records, golds, n, rng):
    """Reparto parejo por modelo×backend; `sin_respuesta` solo si hace falta para cubrir modelos."""
    def cells(pool):
        groups = {}
        for r in pool:
            groups.setdefault((r["model"], r["backend"]), []).append(r)
        for rows in groups.values():
            rng.shuffle(rows)
        return groups

    def draw(groups, limit):
        chosen = []
        order = sorted(groups)
        while len(chosen) < limit and any(groups.values()):
            for cell in order:
                if groups[cell] and len(chosen) < limit:
                    chosen.append(groups[cell].pop())
        return chosen

    normal = [r for r in records if golds[r["qid"]].get("type") != "sin_respuesta"]
    chosen = draw(cells(normal), n)
    missing = {r["model"] for r in records} - {r["model"] for r in chosen}
    if missing:  # se reemplaza al final de la lista: alcanza para cubrir todos los modelos
        extra = [r for r in records if r["model"] in missing and r not in chosen]
        rng.shuffle(extra)
        for model in missing:
            pick = next((r for r in extra if r["model"] == model), None)
            if pick:
                chosen[-1 - list(missing).index(model)] = pick
    return chosen


def sample(records, golds, n, seed, out):
    pool = [r for r in records if r.get("run_idx") == 1 and not r.get("error")]
    chosen = _choose(pool, golds, min(n, len(pool)), random.Random(seed))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for i, r in enumerate(chosen, 1):
        g = golds[r["qid"]]
        evidence = "\n".join(f"[{e['n']}] {mask_handles(e.get('text', ''))[:400]}"
                             for e in r.get("evidence") or [])
        items.append(dict(item_id=i, pregunta=g.get("question") or r.get("query"), respuesta=r["answer"],
                          evidencia=evidence, respuesta_referencia=g.get("reference_answer"),
                          criterio=g.get("answer_criterion"),
                          **{c: "" for c in CRITERIA}, comentario=""))
    for name, order_seed in (("A", 42), ("B", 43)):
        shuffled = items[:]
        random.Random(order_seed).shuffle(shuffled)
        _write(out / f"hoja_evaluador_{name}.csv", SHEET, shuffled)
    _write(out / "clave_items.csv", KEY,
           [dict(item_id=i, run_id=r["run_id"], model=r["model"], backend=r["backend"], qid=r["qid"])
            for i, r in enumerate(chosen, 1)])
    return items


def _scores(path):
    result = {}
    for row in _read(path):
        if not any((row.get(c) or "").strip() for c in CRITERIA):
            continue  # respuesta que nadie calificó
        for c in CRITERIA:
            value = (row.get(c) or "").strip()
            if value not in ("0", "1", "2"):
                raise ValueError(f"Falta o es inválido el puntaje del ítem {row['item_id']}, "
                                 f"criterio {c} ({path}): {value!r}")
            result[row["item_id"], c] = int(value)
    return result


def agree(a_path, b_path, key_path, out):
    a, b = _scores(a_path), _scores(b_path)
    key = [k for k in _read(key_path) if (k["item_id"], CRITERIA[0]) in a]
    ids = [k["item_id"] for k in key]
    if not ids or a.keys() != b.keys() or {i for i, _ in a} != set(ids):
        raise ValueError("Las hojas y la clave no tienen los mismos ítems.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kappa, exact = {}, {}
    for c in CRITERIA:
        xs, ys = [a[i, c] for i in ids], [b[i, c] for i in ids]
        kappa[c] = weighted_kappa(xs, ys)
        exact[c] = sum(x == y for x, y in zip(xs, ys)) / len(ids)
    mean = {(i, c): (a[i, c] + b[i, c]) / 2 for i in ids for c in CRITERIA}
    _write(out / "human_eval_items.csv", ("item_id", "run_id", *CRITERIA),
           [dict(item_id=k["item_id"], run_id=k["run_id"], **{c: mean[k["item_id"], c] for c in CRITERIA})
            for k in key])
    summary, groups = [], {}
    for k in key:
        groups.setdefault((k["model"], k["backend"]), []).append(k["item_id"])
    for (model, backend), members in sorted(groups.items()):
        for c in CRITERIA:
            summary.append(dict(model=model, backend=backend, criterio=c, n=len(members),
                                promedio=sum(mean[i, c] for i in members) / len(members),
                                kappa_global=kappa[c], acuerdo_exacto_global=exact[c]))
    _write(out / "human_eval.csv", list(summary[0]), summary)
    by_id = {k["item_id"]: k for k in key}
    _write(out / "human_disagreements.csv", ("item_id", "run_id", "criterio", "evaluador_A", "evaluador_B"),
           [dict(item_id=i, run_id=by_id[i]["run_id"], criterio=c, evaluador_A=a[i, c], evaluador_B=b[i, c])
            for i in ids for c in CRITERIA if abs(a[i, c] - b[i, c]) >= 2])
    return dict(kappa=kappa, exact=exact)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--tags", nargs="+", required=True)
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", type=Path, default=Path("eval/human"))
    s.add_argument("--raw-dir", type=Path, default=Path("reports/runs/raw"))
    s.add_argument("--gold", type=Path, default=Path("eval/gold.jsonl"))
    g = sub.add_parser("agree")
    g.add_argument("--a", type=Path, default=Path("eval/human/hoja_evaluador_A.csv"))
    g.add_argument("--b", type=Path, default=Path("eval/human/hoja_evaluador_B.csv"))
    g.add_argument("--key", type=Path, default=Path("eval/human/clave_items.csv"))
    g.add_argument("--out", type=Path, default=Path("reports/generation"))
    args = parser.parse_args(argv)
    if args.command == "sample":
        golds = {r["qid"]: r for r in read_jsonl(args.gold)}
        sample(load_raw(args.tags, args.raw_dir), golds, args.n, args.seed, args.out)
    else:
        agree(args.a, args.b, args.key, args.out)


if __name__ == "__main__":
    main()
