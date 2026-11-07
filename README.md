# Taller de RAG con Milvus: repositorio demostrativo

Este repositorio forma parte del taller de generación aumentada por recuperación (RAG) de la Pontificia Universidad Javeriana. Es el punto de partida para las actividades prácticas de ingesta, recuperación, generación y evaluación del taller.

**Proyecto base académico (scaffold), no solución terminada ni rendimiento certificado.** Únicamente Milvus; frontend Streamlit; FastAPI; Ollama local; Python y Docker Compose. Se deben descargar imágenes y pesos de modelos gratuitos, sin API de pago. Por compatibilidad, verificar imágenes disponibles y licencias de cada modelo antes de la clase.

## Requisitos

Docker Engine/Compose v2, al menos 16 GB RAM recomendados para la pila completa (depende de hardware y modelos), espacio para embeddings, datos y pesos. Puede ser mayor a escala completa; empezar por una muestra. Docker Compose incluye Milvus, etcd, MinIO, Ollama, API, frontend.

## Inicio

1. `cp .env.example .env` (Windows: `copy .env.example .env`).
2. Copiar **el archivo real** a `data/documentos_rag.parquet` (la descripción Markdown no es el Parquet).
3. `docker compose up -d --build`.
4. `docker compose exec ollama ollama pull qwen2.5:1.5b`
5. `docker compose exec api python -m scripts.ingest --input data/documentos_rag.parquet --limit 500`
6. Abrir http://localhost:8501, documentación API http://localhost:8000/docs.
7. Comparadores opcionales: `docker compose exec ollama ollama pull llama3.2:1b`, `llama3.2:3b`, `gemma2:2b`.

**Importante:** los ejemplos son candidatos sujetos a licencias/recursos disponibles. El prefijo `query:` / `passage:` se usa porque el modelo de embeddings predeterminado es E5. Al cambiar encoder se debe cambiar nombre de colección, rehacer embeddings y validar dimensión/política de prefijos.

## Endpoints

- `GET /health`: disponibilidad de API (no prueba dependencia).
- `POST /retrieve`: chunks recuperados.
- `POST /ask`: respuesta, modelo, evidencia y latencia. JSON `{"query":"¿Qué señalan los documentos?","model":"qwen2.5:1.5b","k":5}`.

## Diseño y limitaciones deliberadas

- Esqueleto funcional en condiciones compatibles, **no ejecutado end-to-end en este entorno**.
- Chunking inicial por caracteres con solapamiento; estudiantes deben investigar tokenizer, límites, distribución y compararlo con chunking semántico.
- Indexación piloto idempotente por `chunk_id` con `upsert`; falta checkpoint robusto de procesamiento, reintentos y control de lotes grandes.
- Modelo de embedding fijo para comparar generadores sin sesgo de recuperación. Para ablations de embedding, colecciones separadas.
- Metadatos base y offsets; estudiantes ampliarán filtros, control de calidad, deduplicación, reranking, instrumentación y seguridad.
- Se guarda texto en Milvus por simplicidad educativa; evaluar limites y almacenamiento alterno para producción.
- **No publicar documentos ni datos personales** en GitHub: publicar código, instrucciones, muestras autorizadas y artefactos no sensibles.
- El corpus incluye publicaciones sociales y noticias: distinguir alegaciones, atribuciones y evidencia; no equiparar cobertura con veracidad.

## Propuesta de ramas y entregas

`main` infraestructura; `feature/ingesta`, `feature/retrieval`, `feature/ui`, `feature/evaluacion`; abrir PR con pruebas y resultados. Entregar commits, instrucciones desde cero, matriz de experimentos, benchmark, informe, demo y registro de equipo/hardware.

## Reproducibilidad

Fijar versiones y digests de imágenes finales, registrar sha256 del Parquet, semilla, nombres/versiones/hashes de los modelos, hardware, cuantización, parámetros de inferencia, tamaños de chunk, k, tiempos por etapa, subconjunto y selección aleatoria estratificada. La plantilla **no** descarga el Parquet automáticamente ni provee gold estándar.

## Contacto

- **Luis Gabriel Moreno Sandoval**
- Pontificia Universidad Javeriana
- Correo institucional: [morenoluis@javeriana.edu.co](mailto:morenoluis@javeriana.edu.co)
