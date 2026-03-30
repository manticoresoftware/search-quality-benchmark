from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import sleep
from typing import Any
from urllib import error, request

from .config import AppConfig


@dataclass
class EmbeddingUsage:
    input_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class EmbeddingBatchResult:
    vectors: list[list[float]]
    usage: EmbeddingUsage


def resolved_embedding_model(cfg: AppConfig) -> str:
    model = str(cfg.embedding.model or "").strip()
    if not model:
        raise RuntimeError("No embedding model configured. Set embedding.model.")
    return model


def resolved_embedding_similarity(cfg: AppConfig) -> str:
    return str(cfg.manticore.vector_similarity or "cosine").strip().lower()


def resolved_embedding_api_key(cfg: AppConfig) -> str:
    key = str(cfg.embedding.api_key or "").strip()
    if key:
        return key
    key_file = str(cfg.embedding.api_key_file or "").strip()
    if not key_file:
        return ""
    p = Path(key_file)
    if not p.exists():
        key_file = ""
    else:
        return p.read_text(encoding="utf-8").strip()
    legacy_key = str(cfg.manticore.vector_api_key or "").strip()
    if legacy_key:
        return legacy_key
    legacy_key_file = str(cfg.manticore.vector_api_key_file or "").strip()
    if not legacy_key_file:
        return ""
    p = Path(legacy_key_file)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()


def _resolved_embeddings_url(cfg: AppConfig) -> str:
    explicit = str(cfg.embedding.base_url or "").strip().rstrip("/")
    if explicit:
        return explicit + "/embeddings"
    provider = str(cfg.embedding.provider or "openai").strip().lower()
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1/embeddings"
    return "https://api.openai.com/v1/embeddings"


class EmbeddingClient:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        if not texts:
            return EmbeddingBatchResult(vectors=[], usage=EmbeddingUsage())
        api_key = resolved_embedding_api_key(self.cfg)
        if not api_key:
            raise RuntimeError(
                "No embedding API key configured. Set embedding.api_key or embedding.api_key_file."
            )
        url = _resolved_embeddings_url(self.cfg)
        payload: dict[str, Any] = {
            "model": resolved_embedding_model(self.cfg),
            "input": texts,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        provider = str(self.cfg.embedding.provider or "openai").strip().lower()
        if provider == "openrouter":
            site_url = str(self.cfg.embedding.site_url or "").strip()
            app_name = str(self.cfg.embedding.app_name or "").strip()
            if site_url:
                headers["HTTP-Referer"] = site_url
            if app_name:
                headers["X-Title"] = app_name
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        timeout = max(1, int(self.cfg.embedding.timeout_seconds))
        attempts = max(1, int(self.cfg.embedding.retry_attempts or 1))
        base_backoff = max(0.0, float(self.cfg.embedding.retry_backoff_seconds or 0.0))
        data: dict[str, Any] | None = None
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                with request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                rows = data.get("data") or []
                vectors: list[list[float]] = []
                for row in rows:
                    emb = row.get("embedding") or []
                    vectors.append([float(x) for x in emb])
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"Embedding response count mismatch: requested={len(texts)} got={len(vectors)}"
                    )
                usage = data.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                ppm = float(self.cfg.embedding.price_per_million_input_tokens or 0.0)
                est_cost = (prompt_tokens / 1_000_000.0) * ppm if ppm > 0 else 0.0
                return EmbeddingBatchResult(
                    vectors=vectors,
                    usage=EmbeddingUsage(
                        input_tokens=prompt_tokens,
                        estimated_cost_usd=est_cost,
                    ),
                )
            except Exception as exc:
                last_exc = exc
                should_retry = True
                if isinstance(exc, error.HTTPError):
                    should_retry = exc.code in {408, 409, 425, 429} or 500 <= int(exc.code) < 600
                if attempt >= attempts or not should_retry:
                    break
                if base_backoff > 0:
                    sleep(base_backoff * (2 ** (attempt - 1)))
        raise RuntimeError(f"Embedding request failed after {attempts} attempt(s): {last_exc}") from last_exc
