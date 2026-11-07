"""Configuración compartida mediante variables de entorno."""
import os


MILVUS_URI = os.getenv("MILVUS_URI", "http://milvus:19530")
SOLR_URL = os.getenv("SOLR_URL", "http://solr:8983/solr/rag")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
ENCODER = os.getenv("ENCODER", "e5small")
CHUNKER = os.getenv("CHUNKER", "tok256")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "rag_v3")
DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:1.5b")
API_URL = os.getenv("API_URL", "http://api:8000")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def collection_name(encoder: str, chunker: str) -> str:
    return f"c_{encoder}_{chunker}"
