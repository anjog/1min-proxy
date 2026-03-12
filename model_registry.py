"""
Model registry for 1min-proxy.

Fetches and caches the list of models available on 1min.ai, including
vision support flags and credit cost metadata.
Falls back to a built-in model list when the API is unreachable.
"""

import os
import time
import threading
import logging
import requests

logger = logging.getLogger("1min-proxy")


FALLBACK_MODELS = [
    # --- Alibaba ---
    "qwen3-vl-plus", "qwen3-vl-flash", "qwen3-max",
    "qwen-vl-plus", "qwen-vl-max", "qwen-plus", "qwen-max", "qwen-flash",

    # --- Anthropic ---
    "claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514",
    "claude-opus-4-5-20251101", "claude-opus-4-20250514", "claude-opus-4-1-20250805",
    "claude-haiku-4-5-20251001",

    # --- Cohere ---
    "command-r-08-2024",

    # --- DeepSeek ---
    "deepseek-reasoner", "deepseek-chat",

    # --- Google ---
    "gemini-3-pro-preview", "gemini-3-flash-preview",
    "gemini-2.5-pro", "gemini-2.5-flash",

    # --- Mistral ---
    "magistral-small-latest", "magistral-medium-latest",
    "ministral-14b-latest", "open-mistral-nemo",
    "mistral-small-latest", "mistral-medium-latest", "mistral-large-latest",

    # --- OpenAI ---
    "gpt-5.1-codex-mini", "gpt-5.1-codex",
    "o4-mini", "o3-mini",
    "gpt-5.2-pro", "gpt-5.2", "gpt-5.1",
    "gpt-5-nano", "gpt-5-mini", "gpt-5-chat-latest", "gpt-5",
    "gpt-4o-mini", "gpt-4o",
    "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
    "gpt-4-turbo", "gpt-3.5-turbo",
    "o4-mini-deep-research", "o3-pro", "o3-deep-research", "o3",

    # --- Perplexity ---
    "sonar-reasoning-pro", "sonar-pro", "sonar-deep-research", "sonar",

    # --- xAI ---
    "grok-4-fast-reasoning", "grok-4-fast-non-reasoning", "grok-4-0709",
    "grok-3-mini", "grok-3",

    # --- Meta ---
    "meta/meta-llama-3.1-405b-instruct", "meta/meta-llama-3-70b-instruct",
    "meta/llama-4-scout-instruct", "meta/llama-4-maverick-instruct",
    "meta/llama-2-70b-chat",

    # --- OpenAI OSS ---
    "openai/gpt-oss-20b", "openai/gpt-oss-120b",
]

FALLBACK_VISION_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
    "gpt-4.1", "gpt-4.1-mini",
    "gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2",
    "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-3-pro-preview", "gemini-3-flash-preview",
    "claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514",
    "claude-opus-4-5-20251101", "claude-opus-4-20250514",
    "claude-haiku-4-5-20251001",
    "qwen3-vl-plus", "qwen3-vl-flash",
    "qwen-vl-plus", "qwen-vl-max",
}


class ModelRegistry:
    _FETCH_TIMEOUT = 8

    def __init__(self, base_url, ttl, permitted):
        self._base_url  = base_url
        self._ttl       = ttl
        self._permitted = permitted   # leere Liste = alle
        self._lock      = threading.Lock()
        self._all_models    = []
        self._vision_models = set()
        self._metadata      = {}   # modelId → creditMetadata dict
        self._fetched_at    = 0.0
        self._cache_valid   = False

    def get_available_models(self) -> list[str]:
        all_models = self._get_all_models()
        if self._permitted:
            permitted_set = set(self._permitted)
            return [m for m in all_models if m in permitted_set]
        return all_models

    def get_model_meta(self, model: str) -> dict | None:
        self._ensure_fresh()
        return self._metadata.get(model)

    def is_vision_model(self, model: str) -> bool:
        self._ensure_fresh()
        if self._cache_valid:
            return model in self._vision_models
        return model in FALLBACK_VISION_MODELS

    def refresh(self):
        with self._lock:
            self._fetch_and_store()

    def _get_all_models(self):
        self._ensure_fresh()
        return list(self._all_models) if self._cache_valid else list(FALLBACK_MODELS)

    def _ensure_fresh(self):
        if self._cache_valid and (time.monotonic() - self._fetched_at) < self._ttl:
            return  # Fast path ohne Lock
        with self._lock:
            if self._cache_valid and (time.monotonic() - self._fetched_at) < self._ttl:
                return
            self._fetch_and_store()

    def _fetch_and_store(self):
        try:
            chat_models = self._fetch_feature("UNIFY_CHAT_WITH_AI")
            new_all     = [m["modelId"] for m in chat_models]
            new_vision  = {m["modelId"] for m in chat_models
                           if "CHAT_WITH_IMAGE" in m.get("features", [])}
            changed = (new_all != self._all_models or new_vision != self._vision_models)
            self._all_models    = new_all
            self._vision_models = new_vision
            self._metadata      = {m["modelId"]: m["creditMetadata"]
                                   for m in chat_models if m.get("creditMetadata")}
            self._fetched_at  = time.monotonic()
            self._cache_valid = True
            if changed:
                logger.info("ModelRegistry: %d Chat-Modelle, %d Vision-Modelle",
                            len(self._all_models), len(self._vision_models))
            else:
                logger.debug("ModelRegistry: Refresh ohne Änderungen")
        except Exception as exc:
            logger.warning("ModelRegistry: Fetch fehlgeschlagen (%s) — %s", exc,
                           "nutze Stale-Cache" if self._cache_valid else "nutze Fallback-Liste")

    def _fetch_feature(self, feature: str) -> list[dict]:
        url  = f"{self._base_url}?feature={feature}"
        resp = requests.get(url, timeout=self._FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data.get("models"), list):
            raise ValueError(f"Unerwartetes API-Format für feature={feature}")
        return data["models"]
