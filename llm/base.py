"""Abstract base classes and data models for modular LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelStatus:
    """Live runtime status of the configured LLM provider and model."""

    provider: str
    intended_model: str
    configured_model: str
    active_model: str | None
    ollama_connected: bool
    model_connected: bool
    ai_available: bool
    is_fallback: bool = False
    available_models: List[str] = field(default_factory=list)
    status_message: str = ""
    action_hint: str = ""


@dataclass
class LLMResponse:
    """Normalized response returned by any LLM backend."""

    success: bool
    content: str
    model_used: str | None = None
    is_fallback: bool = False
    error: str | None = None
    latency_ms: int = 0


class BaseLLMClient(ABC):
    """Interface that any LLM provider (Ollama, vLLM, LoRA endpoint, etc.) must implement."""

    @abstractmethod
    def check_status(self) -> ModelStatus:
        """Inspect provider reachability, installed models, and active model readiness."""

    @abstractmethod
    def list_models(self) -> List[str]:
        """Return exact model identifiers reported by the backend."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.25,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        """Send chat history messages and return a normalized LLMResponse."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.25,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        """Convenience wrapper around chat() for single-turn prompts."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)
