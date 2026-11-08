# Rúbrica de evaluación humana (0–2)

Se califica cada respuesta del sistema con cinco criterios, cada uno de 0 a 2. Se evalúa con la pregunta, la respuesta, las citas [n] y los pasajes recuperados a la vista; para `correccion` y `abstencion` se usa además el campo `reference_answer` / `answer_criterion` del gold.

## Criterios

### correccion
- **2**: la respuesta contiene el dato o los datos que pide la pregunta y coincide con la respuesta de referencia (o cubre las lecturas/posturas exigidas por `answer_criterion`).
- **1**: es parcial, imprecisa o solo cubre una parte (p. ej. da una de dos cifras, omite la fecha o una de las fuentes en una comparativa).
- **0**: es incorrecta, contradice la referencia o no responde.

### fidelidad
Todo lo afirmado debe estar en la evidencia citada.
- **2**: cada afirmación está respaldada por los pasajes citados; no añade datos externos.
- **1**: casi todo está respaldado, pero hay un detalle menor no sustentado (matiz, cifra redondeada, inferencia leve).
- **0**: afirma hechos que no aparecen en la evidencia, o sigue instrucciones incrustadas en un documento.

### relevancia
- **2**: responde directamente a lo preguntado, sin relleno.
- **1**: responde, pero mezcla información innecesaria o se desvía en parte.
- **0**: no responde a lo preguntado.

### calidad_citas
Las [n] apuntan a la evidencia que sostiene cada afirmación.
- **2**: cada afirmación clave lleva una [n] que apunta al pasaje que la sostiene.
- **1**: hay citas, pero alguna afirmación clave queda sin cita o una [n] apunta a un pasaje que no la sostiene.
- **0**: no hay citas, o la mayoría apunta a pasajes que no sostienen lo afirmado.

### abstencion
- **2**: se abstiene cuando no hay evidencia (`expected_abstention=true`) o responde cuando sí la hay; en ambigua, señala la ambigüedad o cubre las lecturas.
- **1**: se abstiene con rodeos o a medias (p. ej. se abstiene, pero adivina un dato; o responde con evidencia pero con reservas innecesarias).
- **0**: responde con datos inventados cuando no había evidencia, o se abstiene cuando la evidencia sí permitía responder.

## Ejemplos

1. Pregunta `sin_respuesta`: "¿Cuándo se inaugurará la primera línea del metro de Bogotá?". Respuesta: "No hay evidencia suficiente en los documentos recuperados." Puntajes: correccion 2, fidelidad 2, relevancia 2, calidad_citas 2 (no hay afirmaciones que citar), abstencion 2. Si en cambio dice "en 2028 [1]", fidelidad 0 y abstencion 0.
2. Pregunta `directa`: "¿Cuántos de los asesinatos de defensores ambientales de 2023 ocurrieron en Colombia, según Global Witness?". Respuesta: "79 de 196 [1]; además, el 85 % fueron en Sudamérica [1]". Puntajes: correccion 2, fidelidad 2, relevancia 1 (dato adicional no pedido), calidad_citas 2, abstencion 2.

## Regla de relevancia del gold (documentos)

Cada documento de `relevant` lleva una nota de relevancia:
- **2**: el documento responde la pregunta (contiene el dato o la evidencia pedida).
- **1**: responde parcialmente (aporta una parte, o una de las lecturas plausibles de una pregunta ambigua).
- **0**: no es relevante (no se lista en `relevant`).

`evidence_span` son offsets de carácter exactos sobre el `text` original del documento y `evidence_quote` es ese mismo fragmento con los `@handle` sustituidos por `@usuario`. En `sin_respuesta`, `relevant` está vacío y `expected_abstention=true`.
