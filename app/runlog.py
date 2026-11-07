"""Registro acumulativo de ejecuciones en formato JSONL."""
from datetime import datetime, timezone
import json
from pathlib import Path


def log_run(record: dict, path: Path | None = None) -> None:
    now = datetime.now(timezone.utc)
    if path is None:
        path = Path(__file__).resolve().parents[1] / "reports" / "runs" / f"api_{now:%Y%m%d}.jsonl"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record, timestamp=now.isoformat()), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
