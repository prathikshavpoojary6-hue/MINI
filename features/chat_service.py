"""Chat and Step-by-Step Solver orchestration with strict RAG source attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from llm.base import BaseLLMClient
from prompts.gate_prompts import build_chat_system_prompt, build_solver_prompt
from rag.vector_store import LocalKnowledgeBase


@dataclass
class ChatTurnResult:
    """Result of a tutor or solver turn including grounding metadata."""

    success: bool
    answer: str
    model_used: str | None = None
    is_fallback: bool = False
    rag_used: bool = False
    sources: List[Dict] = field(default_factory=list)
    retrieved_chunks: List[Dict] = field(default_factory=list)
    error: str = ""
    latency_ms: int = 0


def _summarize_sources(chunks: List[Dict]) -> List[Dict]:
    """Group retrieved chunks by document name with sorted unique page numbers."""
    by_doc: Dict[str, Dict] = {}
    for chunk in chunks:
        name = chunk.get("source", "Unknown PDF")
        page = chunk.get("page")
        score = float(chunk.get("score", 0.0))
        entry = by_doc.setdefault(name, {"source": name, "pages": set(), "max_score": 0.0})
        if page is not None:
            entry["pages"].add(int(page))
        entry["max_score"] = max(entry["max_score"], score)

    summarized = []
    for entry in by_doc.values():
        summarized.append(
            {
                "source": entry["source"],
                "pages": sorted(entry["pages"]),
                "max_score": entry["max_score"],
            }
        )
    return sorted(summarized, key=lambda item: item["max_score"], reverse=True)


def _retrieve_if_enabled(
    kb: LocalKnowledgeBase | None,
    query: str,
    use_rag: bool,
    top_k: int,
    min_score: float,
) -> List[Dict]:
    if not use_rag or kb is None or not kb.chunks:
        return []
    return kb.search(query=query, top_k=top_k, min_score=min_score)


def answer_chat_query(
    client: BaseLLMClient,
    kb: LocalKnowledgeBase | None,
    query: str,
    subject: str,
    mode: str,
    history: List[Dict[str, str]] | None = None,
    use_rag: bool = True,
    top_k: int = 4,
    min_score: float = 0.22,
    temperature: float = 0.25,
    max_tokens: int = 1200,
) -> ChatTurnResult:
    """Generate a GATE-focused chat response, optionally grounded in uploaded PDFs."""
    retrieved = _retrieve_if_enabled(kb, query, use_rag, top_k, min_score)
    system_prompt = build_chat_system_prompt(
        subject=subject,
        mode=mode,
        retrieved_chunks=retrieved if retrieved else None,
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history[-8:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})

    llm_resp = client.chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not llm_resp.success:
        return ChatTurnResult(
            success=False,
            answer="",
            model_used=llm_resp.model_used,
            is_fallback=llm_resp.is_fallback,
            rag_used=bool(retrieved),
            sources=_summarize_sources(retrieved) if retrieved else [],
            retrieved_chunks=retrieved,
            error=llm_resp.error or "Unable to generate a response.",
            latency_ms=llm_resp.latency_ms,
        )

    return ChatTurnResult(
        success=True,
        answer=llm_resp.content,
        model_used=llm_resp.model_used,
        is_fallback=llm_resp.is_fallback,
        rag_used=bool(retrieved),
        sources=_summarize_sources(retrieved) if retrieved else [],
        retrieved_chunks=retrieved,
        latency_ms=llm_resp.latency_ms,
    )


def answer_solver_query(
    client: BaseLLMClient,
    kb: LocalKnowledgeBase | None,
    problem_text: str,
    subject: str,
    use_rag: bool = True,
    top_k: int = 4,
    min_score: float = 0.22,
    temperature: float = 0.15,
    max_tokens: int = 1400,
) -> ChatTurnResult:
    """Solve a numerical or technical GATE problem step-by-step."""
    retrieved = _retrieve_if_enabled(kb, problem_text, use_rag, top_k, min_score)
    system_prompt = build_solver_prompt(
        subject=subject,
        retrieved_chunks=retrieved if retrieved else None,
    )

    llm_resp = client.generate(
        system_prompt=system_prompt,
        user_prompt=problem_text,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not llm_resp.success:
        return ChatTurnResult(
            success=False,
            answer="",
            model_used=llm_resp.model_used,
            is_fallback=llm_resp.is_fallback,
            rag_used=bool(retrieved),
            sources=_summarize_sources(retrieved) if retrieved else [],
            retrieved_chunks=retrieved,
            error=llm_resp.error or "Unable to solve problem.",
            latency_ms=llm_resp.latency_ms,
        )

    return ChatTurnResult(
        success=True,
        answer=llm_resp.content,
        model_used=llm_resp.model_used,
        is_fallback=llm_resp.is_fallback,
        rag_used=bool(retrieved),
        sources=_summarize_sources(retrieved) if retrieved else [],
        retrieved_chunks=retrieved,
        latency_ms=llm_resp.latency_ms,
    )
