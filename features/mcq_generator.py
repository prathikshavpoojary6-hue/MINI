"""GATE-style 5-MCQ generator and resilient JSON parser."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

from llm.base import BaseLLMClient
from prompts.gate_prompts import build_mcq_prompt
from rag.vector_store import LocalKnowledgeBase


@dataclass
class MCQQuestion:
    """Single 4-option GATE practice MCQ."""

    question: str
    options: List[str]
    correct_index: int
    explanation: str


@dataclass
class MCQGenerationResult:
    """Result returned by MCQ generation."""

    success: bool
    questions: List[MCQQuestion] = field(default_factory=list)
    model_used: str | None = None
    is_fallback: bool = False
    rag_used: bool = False
    sources: List[Dict] = field(default_factory=list)
    error: str = ""
    raw_output: str = ""


def _normalize_correct_index(raw_val: object) -> int:
    if isinstance(raw_val, int) and 0 <= raw_val <= 3:
        return raw_val
    if isinstance(raw_val, str):
        cleaned = raw_val.strip().upper()
        letter_map = {"A": 0, "B": 1, "C": 2, "D": 3, "0": 0, "1": 1, "2": 2, "3": 3}
        if cleaned in letter_map:
            return letter_map[cleaned]
        if cleaned and cleaned[0] in letter_map:
            return letter_map[cleaned[0]]
    return 0


def parse_mcq_json(raw_text: str) -> List[MCQQuestion]:
    """Extract and validate MCQQuestion objects from model JSON output."""
    text = raw_text.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Locate outermost JSON object or array
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    arr_start = text.find("[")
    arr_end = text.rfind("]")

    candidate = text
    if obj_start != -1 and obj_end > obj_start:
        candidate = text[obj_start : obj_end + 1]
    elif arr_start != -1 and arr_end > arr_start:
        candidate = text[arr_start : arr_end + 1]

    data = json.loads(candidate)
    raw_items = data.get("questions", []) if isinstance(data, dict) else data
    if not isinstance(raw_items, list):
        raise ValueError("JSON payload did not contain a list of questions.")

    parsed: List[MCQQuestion] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        q_text = str(item.get("question", "")).strip()
        opts = item.get("options", [])
        if not q_text or not isinstance(opts, list) or len(opts) < 4:
            continue
        clean_opts = [
            re.sub(r"^[A-Da-d][\)\.\:]\s*", "", str(opt).strip()) for opt in opts[:4]
        ]
        if any(not opt for opt in clean_opts):
            continue
        correct_idx = _normalize_correct_index(
            item.get("correct_index", item.get("answer", 0))
        )
        explanation = str(
            item.get("explanation", "Review the core GATE concept for this topic.")
        ).strip()
        parsed.append(
            MCQQuestion(
                question=q_text,
                options=clean_opts,
                correct_index=correct_idx,
                explanation=explanation,
            )
        )

    if not parsed:
        raise ValueError("No valid 4-option MCQs were found in the model response.")
    return parsed[:5]


def generate_mcqs(
    client: BaseLLMClient,
    kb: LocalKnowledgeBase | None,
    subject: str,
    topic: str = "",
    difficulty: str = "GATE Standard",
    use_rag: bool = True,
    top_k: int = 4,
    min_score: float = 0.22,
    num_questions: int = 5,
) -> MCQGenerationResult:
    """Generate 5 GATE-style MCQs using the active LLM and optional PDF context."""
    retrieved: List[Dict] = []
    if use_rag and kb is not None and kb.chunks:
        search_q = f"{subject} {topic}".strip()
        retrieved = kb.search(query=search_q, top_k=top_k, min_score=min_score)

    sys_prompt, usr_prompt = build_mcq_prompt(
        subject=subject,
        topic=topic,
        difficulty=difficulty,
        num_questions=num_questions,
        retrieved_chunks=retrieved if retrieved else None,
    )

    resp = client.generate(
        system_prompt=sys_prompt,
        user_prompt=usr_prompt,
        temperature=0.2,
        max_tokens=1800,
    )
    if not resp.success:
        return MCQGenerationResult(
            success=False,
            model_used=resp.model_used,
            is_fallback=resp.is_fallback,
            rag_used=bool(retrieved),
            error=resp.error or "Failed to generate MCQs.",
        )

    try:
        questions = parse_mcq_json(resp.content)
        sources = []
        if retrieved:
            seen = {}
            for c in retrieved:
                src = c.get("source", "Unknown PDF")
                seen.setdefault(src, set()).add(int(c.get("page", 1)))
            sources = [
                {"source": k, "pages": sorted(v)} for k, v in seen.items()
            ]
        return MCQGenerationResult(
            success=True,
            questions=questions,
            model_used=resp.model_used,
            is_fallback=resp.is_fallback,
            rag_used=bool(retrieved),
            sources=sources,
            raw_output=resp.content,
        )
    except Exception as exc:
        return MCQGenerationResult(
            success=False,
            model_used=resp.model_used,
            is_fallback=resp.is_fallback,
            rag_used=bool(retrieved),
            error=f"Model output could not be parsed as structured MCQs ({exc}). Try again.",
            raw_output=resp.content,
        )
