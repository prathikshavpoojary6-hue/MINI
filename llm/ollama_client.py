"""Ollama and OpenAI-compatible LLM clients with strict Ministral 3 3B policy.

Memory-optimised for 16 GB RAM laptops:
- context window capped at 2048 tokens (num_ctx)
- streaming supported via chat_stream() generator
- single status check cached per chat() call to avoid redundant HTTP round-trips
"""

from __future__ import annotations

import json
import time
from typing import Dict, Generator, Iterator, List

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
    """Local Ollama REST client designed around Ministral 3 3B.

    Memory-safe defaults for 16 GB RAM machines:
      - num_ctx = 2048   (keeps model RAM usage ~4 GB for 3B Q4)
      - num_keep = -1    (Ollama manages KV-cache; avoids full reload per turn)
      - Streaming via chat_stream() — first token appears in < 1 s
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_id: str = "ministral-3:3b",
        allow_explicit_fallback: bool = False,
        fallback_model_id: str = "",
        timeout: int = 120,
        num_ctx: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id.strip()
        self.allow_explicit_fallback = allow_explicit_fallback
        self.fallback_model_id = fallback_model_id.strip()
        self.timeout = timeout
        self.num_ctx = num_ctx  # context window size (keep ≤ 2048 for 16 GB RAM)

    # ------------------------------------------------------------------ #
    # Provider inspection                                                   #
    # ------------------------------------------------------------------ #

    def list_models(self) -> List[str]:
        """Query /api/tags and return exact model identifiers currently installed."""
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        return [m.get("name", "").strip() for m in models if m.get("name")]

    def check_status(self) -> ModelStatus:
        """Verify Ollama connectivity and whether Ministral 3 3B is ready (never Llama)."""
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
                    "Start Ollama to enable AI responses."
                ),
                action_hint=(
                    f"Details: {exc.__class__.__name__}. Run `ollama serve` "
                    f"and ensure `{self.model_id}` is pulled."
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

        # Intended model missing — check whether explicit non-Llama fallback was requested
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

    # ------------------------------------------------------------------ #
    # Non-streaming chat (kept for MCQ / solver which need full JSON)      #
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.25,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send a chat completion request to Ollama /api/chat (non-streaming)."""
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
                "num_ctx": self.num_ctx,   # memory cap: 2048 tokens context
                "num_keep": -1,            # let Ollama manage KV cache (no full reload)
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

    # ------------------------------------------------------------------ #
    # Streaming chat — yields text tokens as they arrive                  #
    # ------------------------------------------------------------------ #

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.25,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        """Stream chat tokens from Ollama /api/chat (stream=True).

        Yields str chunks as they arrive. Caller must handle LLM errors
        by wrapping in try/except; if status is not ai_available, yields
        a single error string and returns.
        """
        status = self.check_status()
        if not status.ai_available or not status.active_model:
            yield f"**AI unavailable:** {status.status_message} {status.action_hint}"
            return

        payload = {
            "model": status.active_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                "num_keep": -1,
            },
        }
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk_data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    token = (chunk_data.get("message") or {}).get("content", "")
                    if token:
                        yield token
                    if chunk_data.get("done"):
                        break
        except requests.Timeout:
            yield "\n\n**[Timeout]** Ollama did not respond in time. Try a shorter prompt."
        except requests.RequestException as exc:
            yield f"\n\n**[Connection Error]** {exc}"

    def get_active_model(self) -> str | None:
        """Quick helper to get the currently active model name, or None."""
        try:
            status = self.check_status()
            return status.active_model if status.ai_available else None
        except Exception:
            return None


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
        max_tokens: int = 1024,
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

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.25,
        max_tokens: int = 1024,
    ):
        """Streaming not yet implemented for OpenAI-compatible; falls back to non-streaming."""
        result = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        if result.success:
            yield result.content
        else:
            yield f"**Error:** {result.error}"
