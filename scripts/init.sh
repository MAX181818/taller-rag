#!/usr/bin/env bash
# Inicializa el piloto desde la raíz del repositorio. Requiere Bash y Python preparado.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
MODELOS="${MODELOS:-qwen2.5:1.5b llama3.2:1b gemma2:2b llama3.2:3b qwen2.5:7b}"
ESPERA_MAX="${ESPERA_MAX:-600}"
if [[ ! "$ESPERA_MAX" =~ ^[1-9][0-9]*$ ]]; then
    echo 'ESPERA_MAX debe ser un entero positivo en segundos.' >&2
    exit 1
fi
[[ -f .env ]] || cp .env.example .env
if [[ ! -f data/documentos_rag.parquet ]]; then
    echo 'Falta data/documentos_rag.parquet. Siga el paso 3 del README: paquete open_corpus_co_es==0.1.1a4, corpus dragons_rag y copia del Parquet.' >&2
    exit 1
fi
"$PYTHON" -c 'import hashlib,sys; from pathlib import Path; p=Path("data/documentos_rag.parquet"); h=hashlib.file_digest(p.open("rb"), "sha256").hexdigest(); expected="f4a669f17eaedd357f72b70c2192e685b6d534a94f6ffaeadbced7290b181728"; print("SHA-256:",h); sys.exit(0 if h == expected else "SHA-256 incorrecto. Repita el paso 3 del README.")'
if [[ ! -f data/synthetic/injection_docs.jsonl ]]; then
    echo 'Falta data/synthetic/injection_docs.jsonl; restaure los cuatro documentos sintéticos antes de inicializar.' >&2
    exit 1
fi
docker compose up -d --build
limite=$((SECONDS + ESPERA_MAX))
while true; do
    sanos=1
    for servicio in etcd minio milvus solr ollama api frontend; do
        id="$(docker compose ps -q "$servicio")"
        estado='ausente'
        if [[ -n "$id" ]]; then
            estado="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}sin-healthcheck{{end}}' "$id")"
        fi
        if [[ "$estado" != healthy ]]; then
            sanos=0
            echo "Esperando $servicio: $estado"
        fi
    done
    if (( sanos )); then break; fi
    if (( SECONDS >= limite )); then
        echo 'Tiempo máximo agotado. Revise docker compose logs y los requisitos del README.' >&2
        exit 1
    fi
    sleep 5
done
read -r -a lista_modelos <<< "$MODELOS"
for modelo in "${lista_modelos[@]}"; do
    docker compose exec -T ollama ollama pull "$modelo"
done
if [[ ! -f data/samples/sample_700.csv ]]; then
    "$PYTHON" -m scripts.make_sample --input data/documentos_rag.parquet --n 700 --seed 42 --out data/samples/sample_700.csv
fi
if [[ ! -f data/derived/chunks_tok256.jsonl ]]; then
    "$PYTHON" -m scripts.build_chunks --input data/documentos_rag.parquet --sample data/samples/sample_700.csv --extra data/synthetic/injection_docs.jsonl --chunker tok256 --out data/derived/chunks_tok256.jsonl
fi
docker compose exec -T api python -m scripts.index_milvus --chunks data/derived/chunks_tok256.jsonl --encoder e5small --resume
docker compose exec -T api python -m scripts.index_solr --chunks data/derived/chunks_tok256.jsonl --resume
echo 'Piloto listo. API: http://localhost:8000/docs | UI: http://localhost:8501'
