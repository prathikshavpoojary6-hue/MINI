"""Factory for instantiating the configured modular LLM backend."""

from __future__ import annotations

from config import AppSettings
from llm.base import BaseLLMClient
from llm.ollama_client import OllamaClient, OpenAICompatibleClient


def get_llm_client(settings: AppSettings) -> BaseLLMClient:
    """Return an LLM client instance for the configured provider.

    Designed so Ministral 3 3B (local Ollama, custom Modelfile with LoRA/QLoRA adapter,
    or an OpenAI-compatible vLLM endpoint) can be swapped via settings without UI changes.
    """
    provider = (settings.llm_provider or "ollama").strip().lower()
    if provider == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=settings.openai_base_url,
            model_id=settings.model_id,
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout,
        )

    return OllamaClient(
        base_url=settings.ollama_base_url,
        model_id=settings.model_id,
        allow_explicit_fallback=settings.allow_explicit_fallback,
        fallback_model_id=settings.fallback_model_id,
        timeout=settings.request_timeout,
    )
