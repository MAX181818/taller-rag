"""Registra recursos de los contenedores desde el host."""
import argparse
import csv
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import re
import subprocess
import time

COLUMNS = ("timestamp", "container", "cpu_pct", "mem_used_mb", "mem_limit_mb",
           "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_util_pct")
STATS_COMMAND = ["docker", "stats", "--no-stream", "--format", "{{.Name}},{{.CPUPerc}},{{.MemUsage}}"]
GPU_COMMAND = ["docker", "compose", "exec", "-T", "ollama", "nvidia-smi",
               "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
LOGGER = logging.getLogger(__name__)


def _memory_mb(value):
    match = re.fullmatch(r"\s*([\d.]+)\s*([a-zA-Z]+)\s*", value)
    if not match:
        raise ValueError("Memoria no reconocida")
    units = {"B": 1, "kB": 1000, "KB": 1000, "MB": 1000**2, "GB": 1000**3,
             "TB": 1000**4, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
    return float(match[1]) * units[match[2]] / 1024**2


def parse_docker_stats(text):
    """Normaliza porcentajes y memoria a MiB y filtra los servicios medidos."""
    result = []
    for row in csv.reader(text.splitlines()):
        if len(row) != 3:
            continue
        name, cpu, memory = (value.strip() for value in row)
        if not any(service in name.lower() for service in ("ollama", "api")):
            continue
        record = dict(container=name, cpu_pct="", mem_used_mb="", mem_limit_mb="")
        try:
            record["cpu_pct"] = float(cpu.rstrip("%"))
        except ValueError:
            pass
        try:
            used, limit = memory.split("/", 1)
            record.update(mem_used_mb=_memory_mb(used), mem_limit_mb=_memory_mb(limit))
        except (ValueError, KeyError):
            pass
        result.append(record)
    return result


def parse_nvidia_smi(text):
    """Suma la memoria de varias GPU y promedia su utilización disponible."""
    names = ("gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_util_pct")
    values = {name: [] for name in names}
    for row in csv.reader(text.splitlines()):
        if len(row) != 3:
            continue
        for name, value in zip(names, row):
            try:
                number = float(value.strip())
                if math.isfinite(number):
                    values[name].append(number)
            except ValueError:
                pass
    return {name: (sum(items) / len(items) if name == "gpu_util_pct" else sum(items))
            if items else "" for name, items in values.items()}


def _command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
        if result.returncode == 0:
            return result.stdout
        LOGGER.warning("Comando fallido (%s): %s", result.returncode, args)
    except (OSError, subprocess.SubprocessError) as error:
        LOGGER.warning("Comando no disponible: %s", error)
    return ""


def sample():
    containers = parse_docker_stats(_command(STATS_COMMAND))
    gpu = parse_nvidia_smi(_command(GPU_COMMAND))
    timestamp = datetime.now(timezone.utc).isoformat()
    return [{**dict.fromkeys(COLUMNS, ""), **row, **gpu, "timestamp": timestamp}
            for row in (containers or [{}])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval debe ser positivo y finito.")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration debe ser cero o un número positivo y finito.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists() or args.out.stat().st_size == 0
    start = time.monotonic()
    try:
        with args.out.open("a", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=COLUMNS)
            if write_header:
                writer.writeheader()
                target.flush()
            while not args.duration or time.monotonic() - start < args.duration:
                tick = time.monotonic()
                writer.writerows(sample())
                target.flush()
                delay = max(0, args.interval - (time.monotonic() - tick))
                if args.duration:
                    delay = min(delay, max(0, args.duration - (time.monotonic() - start)))
                if delay:
                    time.sleep(delay)
    except KeyboardInterrupt:
        LOGGER.info("Monitor detenido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
