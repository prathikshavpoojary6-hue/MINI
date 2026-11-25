"""System and task-specific prompt builders aligned to the GATE AI/ML & CS syllabus."""

from __future__ import annotations

from typing import List, Tuple

MODE_SIMPLE_LEARNING = "Simple Learning Mode"
MODE_GATE_EXAM = "GATE Exam Mode"

BASE_GATE_IDENTITY = """You are GATE AI Tutor, an expert academic tutor dedicated to helping students prepare for the GATE (Graduate Aptitude Test in Engineering) examination, especially GATE Data Science & Artificial Intelligence (DA) and Computer Science (CS).

Core Teaching Principles:
1. Prioritize GATE syllabus concepts, rigor, and problem-solving clarity.
2. Avoid unnecessary filler or off-topic chatter; keep explanations focused and structured.
3. Match technical depth to the selected mode while maintaining mathematical accuracy.
4. Provide clean LaTeX math ($...$ inline and $$...$$ display) whenever formulas or derivations help.
5. Never fabricate facts, formulas, theorems, or citations. If uncertain, state clearly what you are uncertain about.
6. Strict Document Honesty Rule: Only reference uploaded study notes if retrieved document excerpts are explicitly provided below. Never claim an answer came from uploaded notes when no excerpts were provided."""


SIMPLE_LEARNING_INSTRUCTIONS = """Explanation Mode: SIMPLE LEARNING MODE
Structure your explanation for a learner building strong intuition from scratch:
- Start with an intuitive, beginner-friendly summary.
- Define any technical terms clearly before using them.
- Include a relatable real-world or visual analogy.
- Break the concept down into small, numbered steps.
- End with one concrete GATE-relevant example showing why this concept matters on the exam."""


GATE_EXAM_INSTRUCTIONS = """Explanation Mode: GATE EXAM MODE
Present your response in a crisp, high-yield exam-oriented structure using these exact sections:
### Definition
(Formal GATE-level definition)
### Concept
(Core intuition and theoretical conditions/assumptions)
### Formula
(Key equations/theorems in LaTeX, defining every symbol)
### Explanation
(Concise technical walkthrough)
### Example
(A worked GATE-style numerical or conceptual example)
### Important Exam Points
(Common traps, edge cases, time/space bounds, and high-yield bullet points)"""


STEP_BY_STEP_SOLVER_INSTRUCTIONS = """Task Mode: STEP-BY-STEP GATE PROBLEM SOLVER
Solve the numerical or technical GATE problem rigorously using these exact headings:
### 1. Given Information
(Extract every given value, constraint, and assumption)
### 2. To Find
(State the exact target quantity or proof goal)
### 3. Formula / Method Selected
(State the theorem, algorithm, or equation and why it applies)
### 4. Step-by-Step Solution
(Show every intermediate substitution and calculation clearly in LaTeX)
### 5. Final Answer
(Highlight the exact final value with units/notation)
### 6. Why This Method Works
(Briefly explain the underlying intuition and mention any common pitfalls to avoid)"""


def format_rag_context(retrieved_chunks: List[dict]) -> str:
    """Format retrieved GATE material chunks into a prioritized context block."""
    if not retrieved_chunks:
        return ""
    lines = [
        "LOCAL GATE STUDY MATERIAL EXCERPTS (Retrieved via RAG):",
        "INSTRUCTIONS FOR USING THESE EXCERPTS:",
        "1. PRIORITIZE these retrieved GATE study material excerpts as your primary source.",
        "2. Base your answer on these excerpts first and cite the exact filename and page number (e.g., `[DA2025.pdf, p. 4]`).",
        "3. If the excerpts only partially answer the question or do not contain the full answer, explicitly state what is NOT in the documents before adding general GATE knowledge under a clear heading or note.",
        "4. Remember this is RAG retrieval — never claim you were trained or fine-tuned on these files.",
        "",
    ]
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "Unknown document")
        page = chunk.get("page", "?")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "").strip()
        lines.append(f"[Excerpt {idx} | Source: {source} | Page: {page} | Relevance: {score:.2f}]")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def build_chat_system_prompt(
    subject: str,
    mode: str,
    retrieved_chunks: List[dict] | None = None,
) -> str:
    """Build the system prompt for Ask AI / Learn Concepts chat."""
    mode_block = (
        GATE_EXAM_INSTRUCTIONS if mode == MODE_GATE_EXAM else SIMPLE_LEARNING_INSTRUCTIONS
    )
    parts = [
        BASE_GATE_IDENTITY,
        f"Active GATE Subject Focus: {subject}",
        mode_block,
    ]
    if retrieved_chunks:
        parts.append(format_rag_context(retrieved_chunks))
    else:
        parts.append(
            "NO RELEVANT EXCERPTS WERE FOUND IN THE LOCAL GATE DOCUMENTS FOR THIS QUERY. "
            "Begin or note clearly that this answer is from general GATE knowledge (not from your study documents), "
            "and do NOT claim to cite any local or uploaded files."
        )
    return "\n\n".join(parts)


def build_solver_prompt(
    subject: str,
    retrieved_chunks: List[dict] | None = None,
) -> str:
    """Build the system prompt for the Step-by-Step Numerical & Technical Solver."""
    parts = [
        BASE_GATE_IDENTITY,
        f"Active GATE Subject Focus: {subject}",
        STEP_BY_STEP_SOLVER_INSTRUCTIONS,
    ]
    if retrieved_chunks:
        parts.append(format_rag_context(retrieved_chunks))
    else:
        parts.append(
            "NO RELEVANT EXCERPTS WERE FOUND IN THE LOCAL GATE DOCUMENTS FOR THIS PROBLEM. "
            "Solve from general GATE knowledge and do NOT claim to quote local or uploaded documents."
        )
    return "\n\n".join(parts)


def build_mcq_prompt(
    subject: str,
    topic: str = "",
    difficulty: str = "GATE Standard",
    num_questions: int = 5,
    retrieved_chunks: List[dict] | None = None,
) -> Tuple[str, str]:
    """Build (system_prompt, user_prompt) for generating structured GATE MCQs."""
    topic_label = topic.strip() if topic.strip() else f"core syllabus topics in {subject}"
    system_prompt = "\n\n".join(
        [
            BASE_GATE_IDENTITY,
            (
                "You are generating practice MCQs for GATE preparation. "
                "Return ONLY valid JSON — no markdown prose outside the JSON object."
            ),
        ]
        + ([format_rag_context(retrieved_chunks)] if retrieved_chunks else [])
    )

    user_prompt = f"""Generate exactly {num_questions} high-quality GATE-style multiple-choice questions.
Subject: {subject}
Topic Focus: {topic_label}
Difficulty: {difficulty}

Requirements:
- Each question must have exactly 4 distinct options labeled A, B, C, D (store only the option text in the options array, without duplicating 'A.' prefixes).
- Mix conceptual, analytical, and numerical GATE-style questions where appropriate.
- Provide the exact 0-based index (`correct_index`: 0, 1, 2, or 3) AND a clear step-by-step explanation.
- Do not fabricate facts. Ensure the correct option matches the explanation mathematically and logically.

Return strictly this JSON schema:
{{
  "questions": [
    {{
      "question": "Question text here",
      "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
      "correct_index": 0,
      "explanation": "Concise GATE-oriented explanation of why the correct option is right and others are wrong."
    }}
  ]
}}"""
    return system_prompt, user_prompt
