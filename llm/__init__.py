"""Modular LLM backend package for GATE AI Tutor."""

from llm.base import BaseLLMClient, LLMResponse, ModelStatus
from llm.factory import get_llm_client
from llm.ollama_client import OllamaClient, OpenAICompatibleClient

__all__ = [
    "BaseLLMClient",
    "LLMResponse",
    "ModelStatus",
    "OllamaClient",
    "OpenAICompatibleClient",
    "get_llm_client",
]
