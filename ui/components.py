"""Reusable UI rendering components for GATE AI Tutor."""

from __future__ import annotations

import html
from typing import Dict, List

import streamlit as st

from config import SUGGESTED_QUESTIONS
from llm.base import ModelStatus
from rag.vector_store import KBStatus


def _esc(text: object) -> str:
    return html.escape(str(text))


def render_top_header(
    active_page: str,
    subject: str,
    explanation_mode: str,
    model_status: ModelStatus,
    kb_status: KBStatus,
) -> None:
    """Render the top status and context header bar."""
    if model_status.ai_available and not model_status.is_fallback:
        ai_pill = f'<span class="pill pill-green">● AI Ready: {_esc(model_status.active_model)}</span>'
    elif model_status.ai_available and model_status.is_fallback:
        ai_pill = f'<span class="pill pill-amber">● Fallback Model: {_esc(model_status.active_model)}</span>'
    else:
        ai_pill = '<span class="pill pill-red">● AI Unavailable</span>'

    kb_pill_cls = "pill-cyan" if kb_status.document_count > 0 else "pill"
    kb_text = (
        f"KB: {kb_status.document_count} PDF(s)"
        if kb_status.document_count > 0
        else "KB: Empty"
    )

    st.markdown(
        f"""
        <div class="top-header-bar">
            <div>
                <div style="font-size:1.05rem;font-weight:700;color:#f8fafc;">{_esc(active_page)}</div>
                <div style="font-size:0.78rem;color:#94a3b8;">GATE AI/ML &amp; CS Syllabus Companion</div>
            </div>
            <div class="header-pills">
                <span class="pill pill-indigo">Subject: {_esc(subject)}</span>
                <span class="pill">{_esc(explanation_mode)}</span>
                <span class="pill {kb_pill_cls}">{_esc(kb_text)}</span>
                {ai_pill}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_banner(model_status: ModelStatus) -> None:
    """Display clear notice when Ministral 3 3B is unavailable or a fallback model is active."""
    if model_status.ai_available and not model_status.is_fallback:
        return

    if model_status.is_fallback and model_status.active_model:
        st.markdown(
            f"""
            <div class="notice-banner notice-warning">
                <strong>Temporary Fallback Active:</strong> Intended model
                <code>{_esc(model_status.configured_model)}</code> ({_esc(model_status.intended_model)})
                was not found. Currently running <code>{_esc(model_status.active_model)}</code>.
                {_esc(model_status.action_hint)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="notice-banner notice-danger">
            <strong>AI Chat Unavailable ({_esc(model_status.intended_model)} not connected):</strong>
            {_esc(model_status.status_message)}
            <br><span style="opacity:0.9;">{_esc(model_status.action_hint)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome_screen() -> str | None:
    """Render the Welcome Screen and return a clicked suggested question if selected."""
    st.markdown(
        """
        <div class="welcome-hero">
            <span class="welcome-badge">GATE AI/ML &amp; CS • Ministral 3 3B Architecture</span>
            <h1 class="welcome-title">Your personal AI companion for GATE preparation.</h1>
            <p class="welcome-subtitle">Understand concepts, solve problems, and practice smarter.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Suggested GATE Questions")
    clicked_question: str | None = None
    cols = st.columns(2)
    for idx, question in enumerate(SUGGESTED_QUESTIONS):
        target_col = cols[idx % 2]
        with target_col:
            if st.button(
                f"✦  {question}",
                key=f"suggested_q_{idx}",
                use_container_width=True,
            ):
                clicked_question = question
    return clicked_question


def render_grounding_badge(
    rag_used: bool,
    sources: List[Dict],
    model_used: str | None = None,
    is_fallback: bool = False,
    latency_ms: int = 0,
    retrieved_chunks: List[Dict] | None = None,
) -> None:
    """Render explicit source-honesty badge and optional retrieved excerpts expander."""
    badges = []
    if rag_used and sources:
        doc_labels = []
        for s in sources:
            pages = s.get("pages", [])
            p_str = f" (p. {', '.join(str(p) for p in pages)})" if pages else ""
            doc_labels.append(f"{s['source']}{p_str}")
        badges.append(
            f'<span class="pill pill-green">Document-Grounded (RAG) • {_esc("; ".join(doc_labels))}</span>'
        )
    else:
        badges.append(
            '<span class="pill">General Model Knowledge • Not found in local GATE documents</span>'
        )

    if model_used:
        cls = "pill-amber" if is_fallback else "pill-indigo"
        label = f"Fallback: {model_used}" if is_fallback else f"Model: {model_used}"
        badges.append(f'<span class="pill {cls}">{_esc(label)}</span>')

    if latency_ms > 0:
        badges.append(f'<span class="pill">{latency_ms / 1000:.1f}s</span>')

    st.markdown(
        f'<div class="grounding-row">{"".join(badges)}</div>',
        unsafe_allow_html=True,
    )

    if rag_used and retrieved_chunks:
        with st.expander("View Retrieved Study Excerpts", expanded=False):
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(
                    f"**Excerpt {idx}** — `{chunk.get('source', 'PDF')}` "
                    f"(Page {chunk.get('page', '?')}, similarity `{chunk.get('score', 0.0):.2f}`)"
                )
                st.caption(chunk.get("text", ""))


def render_status_panel(
    model_status: ModelStatus,
    kb_status: KBStatus,
    selected_subject: str,
) -> None:
    """Render the full System Status grid covering all required diagnostic fields."""
    selected_model_label = (
        f"{model_status.active_model} (fallback)"
        if model_status.is_fallback and model_status.active_model
        else (model_status.active_model or model_status.configured_model)
    )
    ollama_state = "Connected" if model_status.ollama_connected else "Not Connected"
    model_state = "Connected" if model_status.model_connected else "Unavailable"
    ai_state = "Available" if model_status.ai_available else "Unavailable"

    st.markdown(
        f"""
        <div class="status-grid">
            <div class="status-tile">
                <div class="status-tile-label">Selected Model Name</div>
                <div class="status-tile-value">{_esc(selected_model_label)}</div>
            </div>
            <div class="status-tile">
                <div class="status-tile-label">Model Connection</div>
                <div class="status-tile-value">{_esc(model_state)}</div>
            </div>
            <div class="status-tile">
                <div class="status-tile-label">Ollama Connection</div>
                <div class="status-tile-value">{_esc(ollama_state)}</div>
            </div>
            <div class="status-tile">
                <div class="status-tile-label">AI Availability</div>
                <div class="status-tile-value">{_esc(ai_state)}</div>
            </div>
            <div class="status-tile">
                <div class="status-tile-label">Uploaded Documents</div>
                <div class="status-tile-value">{kb_status.document_count} PDF(s) / {kb_status.chunk_count} chunks</div>
            </div>
            <div class="status-tile">
                <div class="status-tile-label">Knowledge Base Status</div>
                <div class="status-tile-value">{_esc(kb_status.index_backend)}</div>
            </div>
            <div class="status-tile">
                <div class="status-tile-label">Current Subject</div>
                <div class="status-tile-value">{_esc(selected_subject)}</div>
            </div>
            <div class="status-tile">
                <div class="status-tile-label">Embedder Backend</div>
                <div class="status-tile-value">{_esc(kb_status.embedder_backend)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
