"""Ollama and OpenAI-compatible LLM clients with strict Ministral 3 3B policy."""

from __future__ import annotations

import time
from typing import Dict, List

import requests

from config import INTENDED_MODEL_NAME
from llm.base import BaseLLMClient, LLMResponse, ModelStatus


def _is_llama_model(model_name: str) -> bool:
    """Return True if a model identifier belongs to the Llama family."""
    lowered = model_name.lower()
    return "llama" in lowered


def _match_exact_model(configured: str, available: List[str]) -> str | None:
    """Match configured model identifier strictly against installed Ollama tags."""
    if not configured or not available:
        return None
    target = configured.strip()
    if target in available:
        return target
    if ":" not in target and f"{target}:latest" in available:
        return f"{target}:latest"
    target_lower = target.lower()
    for candidate in available:
        if candidate.lower() == target_lower:
            return candidate
    return None


class OllamaClient(BaseLLMClient):
    """Local Ollama REST client designed around Ministral 3 3B."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_id: str = "ministral-3:3b",
        allow_explicit_fallback: bool = False,
        fallback_model_id: str = "",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id.strip()
        self.allow_explicit_fallback = allow_explicit_fallback
        self.fallback_model_id = fallback_model_id.strip()
        self.timeout = timeout

    def list_models(self) -> List[str]:
        """Query /api/tags and return exact model identifiers currently installed."""
        response = requests.get(f"{self.base_url}/api/tags", timeout=3)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        return [m.get("name", "").strip() for m in models if m.get("name")]

    def check_status(self) -> ModelStatus:
        """Verify Ollama connectivity and whether Ministral 3 3B (never Llama default) is ready."""
        if _is_llama_model(self.model_id):
            return ModelStatus(
                provider="ollama",
                intended_model=INTENDED_MODEL_NAME,
                configured_model=self.model_id,
                active_model=None,
                ollama_connected=False,
                model_connected=False,
                ai_available=False,
                status_message=(
                    f"Configured model '{self.model_id}' is blocked by project policy. "
                    "Llama models cannot be used as the default model."
                ),
                action_hint="Set GATE_LLM_MODEL=ministral-3:3b in settings or .env.",
            )

        try:
            available = self.list_models()
        except requests.RequestException as exc:
            return ModelStatus(
                provider="ollama",
                intended_model=INTENDED_MODEL_NAME,
                configured_model=self.model_id,
                active_model=None,
                ollama_connected=False,
                model_connected=False,
                ai_available=False,
                available_models=[],
                status_message=(
                    f"Ollama is not reachable at {self.base_url}. "
                    "Install or start Ollama to enable AI responses."
                ),
                action_hint=(
                    f"Details: {exc.__class__.__name__}. Install Ollama, run `ollama serve`, "
                    f"and pull `{self.model_id}`."
                ),
            )

        matched = _match_exact_model(self.model_id, available)
        if matched:
            return ModelStatus(
                provider="ollama",
                intended_model=INTENDED_MODEL_NAME,
                configured_model=self.model_id,
                active_model=matched,
                ollama_connected=True,
                model_connected=True,
                ai_available=True,
                is_fallback=False,
                available_models=available,
                status_message=f"Connected to {matched} via Ollama.",
                action_hint="",
            )

        # Intended model is missing: check whether explicit non-Llama fallback was requested.
        if self.allow_explicit_fallback and self.fallback_model_id:
            if _is_llama_model(self.fallback_model_id):
                return ModelStatus(
                    provider="ollama",
                    intended_model=INTENDED_MODEL_NAME,
                    configured_model=self.model_id,
                    active_model=None,
                    ollama_connected=True,
                    model_connected=False,
                    ai_available=False,
                    available_models=available,
                    status_message=(
                        f"Fallback model '{self.fallback_model_id}' is a Llama model and is blocked."
                    ),
                    action_hint=f"Pull '{self.model_id}' or select a non-Llama fallback model.",
                )
            fb_match = _match_exact_model(self.fallback_model_id, available)
            if fb_match:
                return ModelStatus(
                    provider="ollama",
                    intended_model=INTENDED_MODEL_NAME,
                    configured_model=self.model_id,
                    active_model=fb_match,
                    ollama_connected=True,
                    model_connected=True,
                    ai_available=True,
                    is_fallback=True,
                    available_models=available,
                    status_message=(
                        f"Intended model '{self.model_id}' not found. "
                        f"Using temporary fallback model: {fb_match}."
                    ),
                    action_hint=f"Run `ollama pull {self.model_id}` to restore Ministral 3 3B.",
                )

        # Check if another ministral tag exists so we can inform the user accurately
        ministral_variants = [m for m in available if "ministral" in m.lower()]
        non_llama_variants = [m for m in available if not _is_llama_model(m)]
        hint = f"Run `ollama pull {self.model_id}` to install Ministral 3 3B."
        if ministral_variants:
            hint = (
                f"Detected Ministral variant(s): {', '.join(ministral_variants)}. "
                "Select one in Settings or update GATE_LLM_MODEL."
            )
        elif non_llama_variants:
            hint = (
                f"Installed non-Llama model(s): {', '.join(non_llama_variants)}. "
                f"Run `ollama pull {self.model_id}` or enable an explicit fallback in Settings."
            )

        return ModelStatus(
            provider="ollama",
            intended_model=INTENDED_MODEL_NAME,
            configured_model=self.model_id,
            active_model=None,
            ollama_connected=True,
            model_connected=False,
            ai_available=False,
            available_models=available,
            status_message=(
                f"Ollama is running, but configured model '{self.model_id}' is not installed."
            ),
            action_hint=hint,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.25,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        """Send a chat completion request to Ollama /api/chat."""
        status = self.check_status()
        if not status.ai_available or not status.active_model:
            return LLMResponse(
                success=False,
                content="",
                model_used=None,
                is_fallback=False,
                error=f"{status.status_message} {status.action_hint}".strip(),
            )

        start = time.perf_counter()
        payload = {
            "model": status.active_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("message") or {}).get("content", "").strip()
            elapsed = int((time.perf_counter() - start) * 1000)
            if not text:
                return LLMResponse(
                    success=False,
                    content="",
                    model_used=status.active_model,
                    is_fallback=status.is_fallback,
                    error="Model returned an empty response.",
                    latency_ms=elapsed,
                )
            return LLMResponse(
                success=True,
                content=text,
                model_used=status.active_model,
                is_fallback=status.is_fallback,
                latency_ms=elapsed,
            )
        except requests.Timeout:
            return LLMResponse(
                success=False,
                content="",
                model_used=status.active_model,
                is_fallback=status.is_fallback,
                error="Ollama request timed out. Try a shorter prompt or increase timeout in Settings.",
            )
        except requests.RequestException as exc:
            return LLMResponse(
                success=False,
                content="",
                model_used=status.active_model,
                is_fallback=status.is_fallback,
                error=f"Ollama communication error: {exc}",
            )


class OpenAICompatibleClient(BaseLLMClient):
    """Modular OpenAI-compatible HTTP client for future vLLM, TGI, or LoRA-tuned endpoints."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str = "",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id.strip()
        self.api_key = api_key.strip()
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_models(self) -> List[str]:
        resp = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=4)
        resp.raise_for_status()
        data = resp.json()
        return [item.get("id", "").strip() for item in data.get("data", []) if item.get("id")]

    def check_status(self) -> ModelStatus:
        if _is_llama_model(self.model_id):
            return ModelStatus(
                provider="openai_compatible",
                intended_model=INTENDED_MODEL_NAME,
                configured_model=self.model_id,
                active_model=None,
                ollama_connected=False,
                model_connected=False,
                ai_available=False,
                status_message="Llama models are disallowed as the default model.",
                action_hint="Configure a Ministral 3 3B endpoint or switch provider to Ollama.",
            )
        try:
            models = self.list_models()
        except requests.RequestException as exc:
            return ModelStatus(
                provider="openai_compatible",
                intended_model=INTENDED_MODEL_NAME,
                configured_model=self.model_id,
                active_model=None,
                ollama_connected=False,
                model_connected=False,
                ai_available=False,
                status_message=f"OpenAI-compatible endpoint unreachable at {self.base_url}.",
                action_hint=f"Details: {exc.__class__.__name__}",
            )

        matched = _match_exact_model(self.model_id, models)
        if matched:
            return ModelStatus(
                provider="openai_compatible",
                intended_model=INTENDED_MODEL_NAME,
                configured_model=self.model_id,
                active_model=matched,
                ollama_connected=False,
                model_connected=True,
                ai_available=True,
                available_models=models,
                status_message=f"Connected to {matched} via OpenAI-compatible API.",
            )
        return ModelStatus(
            provider="openai_compatible",
            intended_model=INTENDED_MODEL_NAME,
            configured_model=self.model_id,
            active_model=None,
            ollama_connected=False,
            model_connected=False,
            ai_available=False,
            available_models=models,
            status_message=f"Endpoint reachable, but '{self.model_id}' is not served.",
            action_hint=f"Available models: {', '.join(models) if models else 'none'}.",
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.25,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        status = self.check_status()
        if not status.ai_available or not status.active_model:
            return LLMResponse(
                success=False,
                content="",
                error=f"{status.status_message} {status.action_hint}".strip(),
            )
        start = time.perf_counter()
        payload = {
            "model": status.active_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = (
                ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
            )
            elapsed = int((time.perf_counter() - start) * 1000)
            if not text:
                return LLMResponse(
                    success=False,
                    content="",
                    model_used=status.active_model,
                    error="Endpoint returned an empty completion.",
                    latency_ms=elapsed,
                )
            return LLMResponse(
                success=True,
                content=text,
                model_used=status.active_model,
                latency_ms=elapsed,
            )
        except requests.RequestException as exc:
            return LLMResponse(
                success=False,
                content="",
                model_used=status.active_model,
                error=f"Endpoint request failed: {exc}",
            )
