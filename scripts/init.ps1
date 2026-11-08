# Compatible con Windows PowerShell 5.1. Ejecutar con el entorno Python preparado.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$PythonPrograma = $env:PYTHON
if ([string]::IsNullOrWhiteSpace($PythonPrograma)) { $PythonPrograma = 'python' }
$ModelosTexto = $env:MODELOS
if ([string]::IsNullOrWhiteSpace($ModelosTexto)) {
    $ModelosTexto = 'qwen2.5:1.5b llama3.2:1b gemma2:2b llama3.2:3b qwen2.5:7b'
}
$EsperaMax = 600
if (-not [string]::IsNullOrWhiteSpace($env:ESPERA_MAX)) {
    $EsperaMax = [int]$env:ESPERA_MAX
}
if ($EsperaMax -le 0) { throw 'ESPERA_MAX debe ser positivo, en segundos.' }

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Fallo al ejecutar $Program (salida $LASTEXITCODE)." }
}

try {
    if (-not (Test-Path -LiteralPath '.env' -PathType Leaf)) {
        Copy-Item -LiteralPath '.env.example' -Destination '.env'
    }
    if (-not (Test-Path -LiteralPath 'data/documentos_rag.parquet' -PathType Leaf)) {
        throw 'Falta data/documentos_rag.parquet. Siga el paso 2 del README: paquete open_corpus_co_es==0.1.1a4, corpus dragons_rag y copia del Parquet.'
    }
    $Esperado = 'f4a669f17eaedd357f72b70c2192e685b6d534a94f6ffaeadbced7290b181728'
    $Actual = (Get-FileHash -LiteralPath 'data/documentos_rag.parquet' -Algorithm SHA256).Hash
    if ($Actual -ne $Esperado) { throw 'SHA-256 incorrecto. Repita el paso 2 del README.' }
    if (-not (Test-Path -LiteralPath 'data/synthetic/injection_docs.jsonl' -PathType Leaf)) {
        throw 'Falta data/synthetic/injection_docs.jsonl; restaure los cuatro documentos sintéticos.'
    }
    Invoke-Checked 'docker' @('compose', 'up', '-d', '--build')
    $Limite = [DateTime]::UtcNow.AddSeconds($EsperaMax)
    while ($true) {
        $Sanos = $true
        foreach ($Servicio in @('etcd', 'minio', 'milvus', 'solr', 'ollama', 'api', 'frontend')) {
            $Identificador = & docker compose ps -q $Servicio
            if ($LASTEXITCODE -ne 0) { throw "No se pudo consultar $Servicio." }
            $Estado = 'ausente'
            if ($Identificador) {
                $Estado = & docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}sin-healthcheck{{end}}' $Identificador
                if ($LASTEXITCODE -ne 0) { throw "No se pudo inspeccionar $Servicio." }
            }
            if ($Estado -ne 'healthy') {
                $Sanos = $false
                Write-Host "Esperando ${Servicio}: $Estado"
            }
        }
        if ($Sanos) { break }
        if ([DateTime]::UtcNow -ge $Limite) {
            throw 'Tiempo máximo agotado. Revise docker compose logs y los requisitos del README.'
        }
        Start-Sleep -Seconds 5
    }
    foreach ($Modelo in ($ModelosTexto.Trim() -split '\s+')) {
        Invoke-Checked 'docker' @('compose', 'exec', '-T', 'ollama', 'ollama', 'pull', $Modelo)
    }
    if (-not (Test-Path -LiteralPath 'data/samples/sample_700.csv' -PathType Leaf)) {
        Invoke-Checked $PythonPrograma @('-m', 'scripts.make_sample', '--input', 'data/documentos_rag.parquet', '--n', '700', '--seed', '42', '--out', 'data/samples/sample_700.csv')
    }
    if (-not (Test-Path -LiteralPath 'data/derived/chunks_tok256.jsonl' -PathType Leaf)) {
        Invoke-Checked $PythonPrograma @('-m', 'scripts.build_chunks', '--input', 'data/documentos_rag.parquet', '--sample', 'data/samples/sample_700.csv', '--extra', 'data/synthetic/injection_docs.jsonl', '--chunker', 'tok256', '--out', 'data/derived/chunks_tok256.jsonl')
    }
    Invoke-Checked 'docker' @('compose', 'exec', '-T', 'api', 'python', '-m', 'scripts.index_milvus', '--chunks', 'data/derived/chunks_tok256.jsonl', '--encoder', 'e5small', '--resume')
    Invoke-Checked 'docker' @('compose', 'exec', '-T', 'api', 'python', '-m', 'scripts.index_solr', '--chunks', 'data/derived/chunks_tok256.jsonl', '--resume')
    Write-Host 'Piloto listo. API: http://localhost:8000/docs | UI: http://localhost:8501'
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
