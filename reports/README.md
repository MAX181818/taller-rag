Resultados de los experimentos. Todo se regenera con los comandos del README.

- `retrieval/`, `tables/` y `figures/`: métricas de recuperación y generación, cuadros y gráficas del informe.
- `generation/annotated_runs.jsonl`: una línea por llamada al modelo (pregunta, respuesta, evidencia citada, configuración, tiempos, tokens y puntajes). Son los resultados crudos de generación.
- `generation/judge.jsonl` y `generation/human_eval*.csv`: juez local y evaluación humana.
- `runs/raw/`: configuración de cada corrida y monitoreo de recursos (`sysmon_*.csv`).
- `demo/`: capturas de la interfaz y `tres_modelos.csv`, la misma pregunta respondida por tres modelos con cada motor.
- `perfilado/`, `chunking/`, `milvus/` y `embeddings/`: perfil del corpus, segmentado, esquemas de las colecciones y pruebas de embeddings.

Los JSONL completos de cada corrida guardan el texto del contexto recuperado, con fragmentos del corpus y URL originales. Por la licencia del corpus y por datos personales no se publican; `run_generation` los vuelve a crear.
