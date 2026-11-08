"""Cliente de Ollama con métricas y errores de generación explícitos."""
import requests

from app.config import OLLAMA_URL


class GenerationError(Exception):
    """Ollama no pudo completar la solicitud."""


class ModelNotAvailable(GenerationError):
    """El modelo solicitado no está disponible en Ollama."""

    def __init__(self, model):
        self.model = model
        super().__init__(f"El modelo {model} no está disponible en Ollama.")


def _request(method, endpoint, *, model=None, timeout=300, **kwargs) -> dict:
    try:
        response = getattr(requests, method)(
            OLLAMA_URL.rstrip("/") + endpoint, timeout=timeout, **kwargs)
        if response.status_code == 404 and model:
            raise ModelNotAvailable(model)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as error:
        raise GenerationError("Se agotó el tiempo de espera de Ollama.") from error
    except requests.RequestException as error:
        raise GenerationError(f"No se pudo completar la solicitud a Ollama: {error}") from error
    except ValueError as error:
        raise GenerationError("Ollama devolvió una respuesta que no es JSON válido.") from error
    if not isinstance(payload, dict) or payload.get("error"):
        raise GenerationError(f"Ollama devolvió un error: {payload}")
    return payload


def generate(model, prompt, *, temperature=0.0, seed=42, num_ctx=4096,
             num_predict=384, keep_alive="10m", timeout=300) -> dict:
    payload = _request("post", "/api/generate", model=model, timeout=timeout, json={
        "model": model, "prompt": prompt, "stream": False, "keep_alive": keep_alive,
        "options": {"temperature": temperature, "seed": seed,
                    "num_ctx": num_ctx, "num_predict": num_predict},
    })
    eval_s = payload.get("eval_duration", 0) / 1e9
    completion_tokens = payload.get("eval_count", 0)
    return {
        "answer": payload.get("response", ""),
        "prompt_tokens": payload.get("prompt_eval_count", 0),
        "completion_tokens": completion_tokens,
        "load_s": payload.get("load_duration", 0) / 1e9,
        "prompt_eval_s": payload.get("prompt_eval_duration", 0) / 1e9,
        "eval_s": eval_s,
        "total_s": payload.get("total_duration", 0) / 1e9,
        "tokens_per_s": completion_tokens / eval_s if eval_s else 0.0,
    }


def list_models() -> list[str]:
    return [row["name"] for row in _request("get", "/api/tags").get("models", [])]


def model_info(model) -> dict:
    details = _request("post", "/api/show", model=model, json={"model": model}).get("details", {})
    tags = _request("get", "/api/tags", model=model).get("models", [])

    def canonical(name):
        return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"

    digest = next((row.get("digest", "") for row in tags
                   if canonical(row.get("name", "")) == canonical(model)), "")
    return {"parameter_size": details.get("parameter_size", ""),
            "quantization_level": details.get("quantization_level", ""),
            "digest": digest, "family": details.get("family", "")}


def unload(model):
    _request("post", "/api/generate", model=model,
             json={"model": model, "prompt": "", "stream": False, "keep_alive": 0})
