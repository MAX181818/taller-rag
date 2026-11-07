"""Muestras estratificadas reproducibles sin texto ni datos de autores."""
import argparse
from datetime import datetime, timezone
import heapq
from pathlib import Path

import pandas as pd

COLUMNS = ["document_id", "text_sha256", "collection_name", "provenance_ids"]


def normalize_created_at(s: str | None) -> int:
    """Convierte una fecha ISO a segundos UTC; devuelve -1 si no es válida."""
    if not isinstance(s, str) or not s.strip():
        return -1
    try:
        value = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp())
    except (ValueError, OverflowError, OSError):
        return -1


def make_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Deduplica por hash y toma prefijos por estrato con cuotas monótonas."""
    valid = df.loc[df["text"].fillna("").str.strip().ne("")].copy()
    valid = valid.sort_values("document_id", kind="stable")
    provenance = valid.groupby("text_sha256")["document_id"].agg(lambda ids: "|".join(sorted(ids)))
    unique = valid.drop_duplicates("text_sha256").copy()
    unique["provenance_ids"] = unique["text_sha256"].map(provenance)
    if not isinstance(n, int) or n < 0 or n > len(unique):
        raise ValueError(f"n debe estar entre 0 y {len(unique)}")
    groups = [group.sample(frac=1, random_state=seed) for _, group in
              unique.groupby("collection_name", sort=True, dropna=False)]
    # Webster: asignar plazas por prioridad evita perder cuotas al aumentar n.
    quotas = [0] * len(groups)
    heap = [(0.5 / len(group), i) for i, group in enumerate(groups)]
    heapq.heapify(heap)
    for _ in range(n):
        _, i = heapq.heappop(heap)
        quotas[i] += 1
        if quotas[i] < len(groups[i]):
            heapq.heappush(heap, ((quotas[i] + 0.5) / len(groups[i]), i))
    if not groups:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([g.iloc[:q] for g, q in zip(groups, quotas)])[COLUMNS].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/documentos_rag.parquet", help="Parquet de entrada")
    parser.add_argument("--n", type=int, required=True, help="Número de documentos")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible")
    parser.add_argument("--out", required=True, help="CSV de salida")
    args = parser.parse_args()
    df = pd.read_parquet(args.input, columns=["document_id", "text_sha256", "collection_name", "text"])
    sample = make_sample(df, args.n, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(f"Muestra guardada: {len(sample)} filas; por colección: {sample.collection_name.value_counts().to_dict()}")


if __name__ == "__main__":
    main()


