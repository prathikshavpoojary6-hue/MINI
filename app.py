"""GATE AI Tutor - Streamlit Web Application.

Syllabus-based AI Tutor for GATE AI/ML & CS preparation designed around Ministral 3 3B.
Run with: streamlit run app.py

Changes v2 (LLM Integration):
- Ask AI page now uses streaming via st.write_stream() + chat_stream()
- Memory-safe: num_ctx=2048, num_keep=-1 (Ollama manages KV cache)
- All other pages (Learn Concepts, MCQs, Solver) use non-streaming for structured output
- Exact model: ministral-3:3b verified via ollama list before writing this file
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List

import streamlit as st

from config import (
    DEFAULT_OLLAMA_MODEL,
    GATE_SUBJECTS,
    INTENDED_MODEL_NAME,
    SOLVER_SAMPLES,
    SUBJECT_TOPICS,
    AppSettings,
)
from features.chat_service import (
    answer_chat_query,
    answer_solver_query,
    stream_chat_query,
)
from features.mcq_generator import generate_mcqs
from llm.base import ModelStatus
from llm.factory import get_llm_client
from prompts.gate_prompts import MODE_GATE_EXAM, MODE_SIMPLE_LEARNING
from rag.vector_store import LocalKnowledgeBase
from ui.components import (
    render_grounding_badge,
    render_model_banner,
    render_status_panel,
    render_top_header,
    render_welcome_screen,
)
from ui.styles import inject_custom_css

NAV_PAGES = [
    "Ask AI",
    "Learn Concepts",
    "Practice MCQs",
    "Step-by-Step Solver",
    "Upload Study Material",
]


def _init_session_state() -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = AppSettings.load()

    if "kb" not in st.session_state:
        s: AppSettings = st.session_state.settings
        st.session_state.kb = LocalKnowledgeBase(
            embedding_model=s.embedding_model,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
        )

    if "selected_subject" not in st.session_state:
        st.session_state.selected_subject = "Machine Learning"

    if "explanation_mode" not in st.session_state:
        st.session_state.explanation_mode = MODE_SIMPLE_LEARNING

    if "active_page" not in st.session_state:
        st.session_state.active_page = "Ask AI"

    if "nav_radio" not in st.session_state:
        st.session_state.nav_radio = st.session_state.active_page

    if "chats" not in st.session_state:
        first_id = uuid.uuid4().hex[:8]
        st.session_state.chats = {
            first_id: {
                "title": "New Chat",
                "subject": st.session_state.selected_subject,
                "messages": [],
            }
        }
        st.session_state.active_chat_id = first_id

    if "mcq_state" not in st.session_state:
        st.session_state.mcq_state = {
            "questions": [],
            "submitted": False,
            "answers": {},
            "subject": st.session_state.selected_subject,
            "topic": "",
            "rag_used": False,
            "sources": [],
            "model_used": None,
            "is_fallback": False,
            "error": "",
        }

    if "concept_result" not in st.session_state:
        st.session_state.concept_result = None

    if "solver_result" not in st.session_state:
        st.session_state.solver_result = None

    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True


def _create_new_chat() -> None:
    chat_id = uuid.uuid4().hex[:8]
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "subject": st.session_state.selected_subject,
        "messages": [],
    }
    st.session_state.active_chat_id = chat_id
    st.session_state.active_page = "Ask AI"
    st.session_state.nav_radio = "Ask AI"


def _select_chat(chat_id: str) -> None:
    """Callback to switch to an existing chat and navigate to Ask AI before rerun."""
    st.session_state.active_chat_id = chat_id
    st.session_state.active_page = "Ask AI"
    st.session_state.nav_radio = "Ask AI"


def _on_nav_change() -> None:
    """Sync active_page when the user clicks the sidebar navigation radio."""
    st.session_state.active_page = st.session_state.nav_radio


def _clear_active_chat() -> None:
    cid = st.session_state.active_chat_id
    if cid in st.session_state.chats:
        st.session_state.chats[cid]["messages"] = []
        st.session_state.chats[cid]["title"] = "New Chat"


# ---------------------------------------------------------------------------
# Ask AI: streaming submit (replaces old _submit_chat_prompt)
# ---------------------------------------------------------------------------

def _submit_chat_stream(prompt: str, client, model_status: ModelStatus) -> None:
    """Submit a question to Ministral 3 3B and stream the response token-by-token."""
    cid = st.session_state.active_chat_id
    chat = st.session_state.chats[cid]
    clean_prompt = prompt.strip()
    if not clean_prompt:
        return

    # Update chat title from first message
    if chat["title"] == "New Chat":
        chat["title"] = clean_prompt[:36] + ("..." if len(clean_prompt) > 36 else "")

    # Store user message immediately so it shows before the stream starts
    chat["messages"].append({"role": "user", "content": clean_prompt})

    # --- AI unavailable path ---
    if not model_status.ai_available:
        chat["messages"].append(
            {
                "role": "assistant",
                "content": (
                    f"**AI Chat Unavailable ({INTENDED_MODEL_NAME} not connected)**\n\n"
                    f"{model_status.status_message}\n\n"
                    f"**Next Step:** {model_status.action_hint}"
                ),
                "rag_used": False,
                "sources": [],
                "retrieved_chunks": [],
                "model_used": None,
                "is_fallback": False,
                "is_error": True,
            }
        )
        return

    settings: AppSettings = st.session_state.settings
    kb: LocalKnowledgeBase = st.session_state.kb

    prior_history = [
        {"role": m["role"], "content": m["content"]}
        for m in chat["messages"][:-1]  # exclude the just-added user message
        if m.get("role") in ("user", "assistant") and not m.get("is_error")
    ]

    # Get stream generator + RAG metadata
    token_gen, retrieved_chunks, rag_used, is_fallback = stream_chat_query(
        client=client,
        kb=kb,
        query=clean_prompt,
        subject=st.session_state.selected_subject,
        mode=st.session_state.explanation_mode,
        history=prior_history,
        use_rag=st.session_state.use_rag,
        top_k=settings.top_k_retrieval,
        min_score=settings.min_similarity_score,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    # Stream response live inside a chat message bubble
    t_start = time.perf_counter()
    full_response = ""
    with st.chat_message("assistant"):
        full_response = st.write_stream(token_gen)

    latency_ms = int((time.perf_counter() - t_start) * 1000)

    # Summarize sources
    from features.chat_service import _summarize_sources
    sources = _summarize_sources(retrieved_chunks) if retrieved_chunks else []

    # Render grounding badge immediately below the streamed response
    render_grounding_badge(
        rag_used=rag_used,
        sources=sources,
        model_used=model_status.active_model,
        is_fallback=is_fallback,
        latency_ms=latency_ms,
        retrieved_chunks=retrieved_chunks,
    )

    # Persist assistant message in chat history
    chat["messages"].append(
        {
            "role": "assistant",
            "content": full_response,
            "rag_used": rag_used,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "model_used": model_status.active_model,
            "is_fallback": is_fallback,
            "latency_ms": latency_ms,
            "is_error": False,
        }
    )


def _render_sidebar(client, model_status: ModelStatus, kb: LocalKnowledgeBase) -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-card">
                <div class="brand-title">🎓 GATE AI Tutor</div>
                <div class="brand-subtitle">Syllabus-Based Prep • {INTENDED_MODEL_NAME}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            "＋  New Chat",
            type="primary",
            use_container_width=True,
            on_click=_create_new_chat,
        )

        st.markdown('<div class="sidebar-section-label">Workspace</div>', unsafe_allow_html=True)
        chosen_page = st.radio(
            "Navigation",
            options=NAV_PAGES,
            key="nav_radio",
            on_change=_on_nav_change,
            label_visibility="collapsed",
        )
        st.session_state.active_page = chosen_page

        st.markdown('<div class="sidebar-section-label">GATE Subject</div>', unsafe_allow_html=True)
        subject = st.selectbox(
            "GATE Subject",
            options=GATE_SUBJECTS,
            index=GATE_SUBJECTS.index(st.session_state.selected_subject),
            key="subject_select",
            label_visibility="collapsed",
        )
        st.session_state.selected_subject = subject

        st.markdown('<div class="sidebar-section-label">Explanation Mode</div>', unsafe_allow_html=True)
        mode_choice = st.radio(
            "Explanation Mode",
            options=[MODE_SIMPLE_LEARNING, MODE_GATE_EXAM],
            index=0 if st.session_state.explanation_mode == MODE_SIMPLE_LEARNING else 1,
            key="mode_radio",
            label_visibility="collapsed",
        )
        st.session_state.explanation_mode = mode_choice

        st.markdown('<div class="sidebar-section-label">Session Chat History</div>', unsafe_allow_html=True)
        chat_ids = list(reversed(list(st.session_state.chats.keys())))
        for cid in chat_ids[:6]:
            c_info = st.session_state.chats[cid]
            is_current = cid == st.session_state.active_chat_id
            prefix = "▸ " if is_current else "  "
            st.button(
                f"{prefix}{c_info['title']}",
                key=f"hist_{cid}",
                use_container_width=True,
                on_click=_select_chat,
                args=(cid,),
            )

        st.markdown('<div class="sidebar-section-label">Status & Settings</div>', unsafe_allow_html=True)
        kb_status = kb.get_status()
        status_dot = "🟢" if (model_status.ai_available and not model_status.is_fallback) else (
            "🟡" if model_status.is_fallback else "🔴"
        )
        st.caption(
            f"{status_dot} **Model:** `{model_status.active_model or model_status.configured_model}`\n\n"
            f"📚 **Knowledge Base:** {kb_status.document_count} docs ({kb_status.chunk_count} chunks)"
        )

        with st.expander("Settings & Diagnostics", expanded=False):
            _render_settings_form(model_status)


def _render_settings_form(model_status: ModelStatus) -> None:
    s: AppSettings = st.session_state.settings
    provider = st.selectbox(
        "Model Provider",
        options=["ollama", "openai_compatible"],
        index=0 if s.llm_provider == "ollama" else 1,
    )
    model_id = st.text_input(
        "Primary Model Identifier",
        value=s.model_id,
        help=f"Intended default: {DEFAULT_OLLAMA_MODEL} (Ministral 3 3B). Llama defaults are disallowed.",
    )
    ollama_url = st.text_input("Ollama Base URL", value=s.ollama_base_url)

    openai_url = s.openai_base_url
    openai_key = s.openai_api_key
    if provider == "openai_compatible":
        openai_url = st.text_input("OpenAI-Compatible URL", value=s.openai_base_url)
        openai_key = st.text_input("API Key (optional)", value=s.openai_api_key, type="password")

    non_llama_installed = [
        m for m in model_status.available_models if "llama" not in m.lower()
    ]
    allow_fb = st.checkbox(
        "Allow explicit non-Llama fallback",
        value=s.allow_explicit_fallback,
        help="Never switches silently to Llama. Only uses an explicitly selected non-Llama model if Ministral 3 3B is missing.",
    )
    fb_model = s.fallback_model_id
    if allow_fb:
        if non_llama_installed:
            idx = (
                non_llama_installed.index(s.fallback_model_id)
                if s.fallback_model_id in non_llama_installed
                else 0
            )
            fb_model = st.selectbox("Fallback Model (Non-Llama)", options=non_llama_installed, index=idx)
        else:
            fb_model = st.text_input("Fallback Model Name (Non-Llama)", value=s.fallback_model_id)

    temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(s.temperature), step=0.05)
    top_k = st.slider("RAG Top-K Chunks", min_value=1, max_value=8, value=int(s.top_k_retrieval))
    num_ctx = st.slider(
        "Context Window (num_ctx)",
        min_value=512, max_value=4096, value=int(s.num_ctx), step=256,
        help="Keep at 2048 for 16 GB RAM. Increasing this uses more memory.",
    )

    if st.button("Save & Refresh Status", use_container_width=True):
        s.llm_provider = provider
        s.model_id = model_id.strip() or DEFAULT_OLLAMA_MODEL
        s.ollama_base_url = ollama_url.strip().rstrip("/")
        s.openai_base_url = openai_url.strip().rstrip("/")
        s.openai_api_key = openai_key.strip()
        s.allow_explicit_fallback = allow_fb
        s.fallback_model_id = fb_model.strip()
        s.temperature = temp
        s.top_k_retrieval = top_k
        s.num_ctx = num_ctx
        s.save()
        st.success("Settings saved.")
        st.rerun()


# ---------------------------------------------------------------------------
# Ask AI page — uses streaming
# ---------------------------------------------------------------------------

def _render_ask_ai_page(client, model_status: ModelStatus, kb: LocalKnowledgeBase) -> None:
    render_model_banner(model_status)

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.2, 1.5, 1])
    with ctrl_col1:
        st.session_state.use_rag = st.checkbox(
            "Prioritize local GATE Knowledge Base (RAG)",
            value=st.session_state.use_rag,
            disabled=len(kb.chunks) == 0,
            help="Retrieves relevant chunks from your extracted GATE materials & uploaded PDFs.",
        )
    with ctrl_col2:
        st.caption(f"Mode: **{st.session_state.explanation_mode}**")
    with ctrl_col3:
        if st.button("Clear Chat", use_container_width=True):
            _clear_active_chat()
            st.rerun()

    chat = st.session_state.chats[st.session_state.active_chat_id]
    messages: List[Dict] = chat["messages"]

    if not messages:
        # Welcome screen — clicking a suggestion triggers a stream right away
        clicked = render_welcome_screen()
        if clicked:
            with st.chat_message("user"):
                st.markdown(clicked)
            _submit_chat_stream(clicked, client, model_status)
            st.rerun()
    else:
        # Render existing history
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
            if msg["role"] == "assistant" and not msg.get("is_error"):
                render_grounding_badge(
                    rag_used=bool(msg.get("rag_used")),
                    sources=msg.get("sources") or [],
                    model_used=msg.get("model_used"),
                    is_fallback=bool(msg.get("is_fallback")),
                    latency_ms=int(msg.get("latency_ms") or 0),
                    retrieved_chunks=msg.get("retrieved_chunks") or [],
                )

    # Chat input at the bottom
    user_prompt = st.chat_input(
        f"Ask a {st.session_state.selected_subject} question for GATE..."
    )
    if user_prompt:
        # Show the user bubble immediately, then stream
        with st.chat_message("user"):
            st.markdown(user_prompt)
        _submit_chat_stream(user_prompt, client, model_status)
        st.rerun()


# ---------------------------------------------------------------------------
# Learn Concepts page (non-streaming — structured output)
# ---------------------------------------------------------------------------

def _render_learn_concepts_page(client, model_status: ModelStatus, kb: LocalKnowledgeBase) -> None:
    render_model_banner(model_status)
    subject = st.session_state.selected_subject
    mode = st.session_state.explanation_mode
    settings: AppSettings = st.session_state.settings

    st.markdown(
        f"""
        <div class="glass-card">
            <div style="font-weight:700;font-size:1.05rem;color:#f8fafc;">Learn Concepts — {subject}</div>
            <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.2rem;">
                Select a core GATE syllabus topic or type your own. Currently using <strong>{mode}</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    topics = SUBJECT_TOPICS.get(subject, [])
    selected_topic = ""
    if topics:
        st.markdown("###### Core Syllabus Topics")
        t_cols = st.columns(2)
        for idx, t in enumerate(topics):
            with t_cols[idx % 2]:
                if st.button(f"📘 {t}", key=f"topic_btn_{idx}", use_container_width=True):
                    selected_topic = t

    with st.form("learn_concept_form", clear_on_submit=False):
        custom_topic = st.text_input(
            "Or enter any concept/theorem to learn",
            placeholder=f"e.g., What is Bias-Variance Tradeoff in {subject}?",
        )
        use_rag = st.checkbox(
            "Ground explanation in uploaded PDF notes if relevant",
            value=len(kb.documents) > 0,
            disabled=len(kb.documents) == 0,
        )
        submitted = st.form_submit_button("Explain Concept", type="primary")

    target_topic = selected_topic or (custom_topic.strip() if submitted else "")
    if target_topic:
        if not model_status.ai_available:
            st.error(
                f"AI unavailable ({INTENDED_MODEL_NAME} not connected). "
                f"{model_status.status_message} {model_status.action_hint}"
            )
        else:
            prompt = (
                f"Teach me the GATE {subject} concept: '{target_topic}'. "
                "Include clear intuition, equations where needed, and a GATE-relevant example."
            )
            with st.spinner(f"Generating {mode} explanation with {model_status.active_model}..."):
                res = answer_chat_query(
                    client=client,
                    kb=kb,
                    query=prompt,
                    subject=subject,
                    mode=mode,
                    use_rag=use_rag,
                    top_k=settings.top_k_retrieval,
                    min_score=settings.min_similarity_score,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                )
            st.session_state.concept_result = {"topic": target_topic, "result": res}

    if st.session_state.concept_result:
        item = st.session_state.concept_result
        res = item["result"]
        st.markdown("---")
        st.markdown(f"#### Topic: {item['topic']}")
        if res.success:
            st.markdown(res.answer)
            render_grounding_badge(
                rag_used=res.rag_used,
                sources=res.sources,
                model_used=res.model_used,
                is_fallback=res.is_fallback,
                latency_ms=res.latency_ms,
                retrieved_chunks=res.retrieved_chunks,
            )
        else:
            st.error(res.error)


# ---------------------------------------------------------------------------
# MCQ Practice page
# ---------------------------------------------------------------------------

def _render_mcq_page(client, model_status: ModelStatus, kb: LocalKnowledgeBase) -> None:
    render_model_banner(model_status)
    settings: AppSettings = st.session_state.settings

    st.markdown(
        """
        <div class="glass-card">
            <div style="font-weight:700;font-size:1.05rem;color:#f8fafc;">GATE MCQ Practice Arena</div>
            <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.2rem;">
                Generate 5 GATE-style 4-option MCQs, submit your answers, and review detailed solutions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1.4, 1.5, 1.1])
    with col1:
        mcq_subject = st.selectbox(
            "Subject",
            options=GATE_SUBJECTS,
            index=GATE_SUBJECTS.index(st.session_state.selected_subject),
            key="mcq_subject_select",
        )
    with col2:
        mcq_topic = st.text_input(
            "Topic Focus (optional)",
            placeholder="e.g., Conditional Probability, B+ Trees, Eigenvalues",
            key="mcq_topic_input",
        )
    with col3:
        difficulty = st.selectbox(
            "Difficulty",
            options=["GATE Easy", "GATE Standard", "GATE Tricky"],
            index=1,
        )

    use_rag = st.checkbox(
        "Use uploaded study PDFs when generating questions",
        value=len(kb.documents) > 0,
        disabled=len(kb.documents) == 0,
        key="mcq_use_rag",
    )

    gen_col, reset_col = st.columns([1.5, 1])
    with gen_col:
        gen_clicked = st.button("⚡ Generate 5 GATE MCQs", type="primary", use_container_width=True)
    with reset_col:
        if st.button("Reset Quiz", use_container_width=True):
            st.session_state.mcq_state = {
                "questions": [],
                "submitted": False,
                "answers": {},
                "subject": mcq_subject,
                "topic": mcq_topic,
                "rag_used": False,
                "sources": [],
                "model_used": None,
                "is_fallback": False,
                "error": "",
            }
            st.rerun()

    if gen_clicked:
        if not model_status.ai_available:
            st.error(
                f"Cannot generate MCQs: {INTENDED_MODEL_NAME} is not connected. "
                f"{model_status.status_message} {model_status.action_hint}"
            )
        else:
            with st.spinner(f"Generating 5 {mcq_subject} MCQs via {model_status.active_model}..."):
                mcq_res = generate_mcqs(
                    client=client,
                    kb=kb,
                    subject=mcq_subject,
                    topic=mcq_topic,
                    difficulty=difficulty,
                    use_rag=use_rag,
                    top_k=settings.top_k_retrieval,
                    min_score=settings.min_similarity_score,
                    num_questions=5,
                )
            if mcq_res.success:
                st.session_state.mcq_state = {
                    "questions": mcq_res.questions,
                    "submitted": False,
                    "answers": {},
                    "subject": mcq_subject,
                    "topic": mcq_topic,
                    "rag_used": mcq_res.rag_used,
                    "sources": mcq_res.sources,
                    "model_used": mcq_res.model_used,
                    "is_fallback": mcq_res.is_fallback,
                    "error": "",
                }
            else:
                st.session_state.mcq_state["error"] = mcq_res.error
            st.rerun()

    mcq_state = st.session_state.mcq_state
    if mcq_state.get("error"):
        st.error(mcq_state["error"])

    questions = mcq_state.get("questions", [])
    if not questions:
        st.info("Select a subject and click **Generate 5 GATE MCQs** to start a practice quiz.")
        return

    render_grounding_badge(
        rag_used=bool(mcq_state.get("rag_used")),
        sources=mcq_state.get("sources") or [],
        model_used=mcq_state.get("model_used"),
        is_fallback=bool(mcq_state.get("is_fallback")),
    )

    option_letters = ["A", "B", "C", "D"]
    with st.form("mcq_quiz_form"):
        for idx, q in enumerate(questions):
            st.markdown(
                f'<div class="mcq-card"><div class="mcq-q-title">Q{idx + 1}. {q.question}</div></div>',
                unsafe_allow_html=True,
            )
            labels = [f"{option_letters[i]}. {opt}" for i, opt in enumerate(q.options)]
            prev_choice = mcq_state["answers"].get(idx)
            default_idx = prev_choice if isinstance(prev_choice, int) else 0
            selected_label = st.radio(
                f"Select answer for Q{idx + 1}",
                options=labels,
                index=default_idx,
                key=f"mcq_radio_{idx}",
                label_visibility="collapsed",
                disabled=mcq_state["submitted"],
            )
            mcq_state["answers"][idx] = labels.index(selected_label)

        submit_quiz = st.form_submit_button(
            "Submit Answers",
            type="primary",
            disabled=mcq_state["submitted"],
        )
        if submit_quiz:
            mcq_state["submitted"] = True
            st.rerun()

    if mcq_state["submitted"]:
        total = len(questions)
        score = sum(
            1
            for idx, q in enumerate(questions)
            if mcq_state["answers"].get(idx) == q.correct_index
        )
        pct = round((score / total) * 100) if total else 0
        st.success(f"**Quiz Score: {score} / {total} ({pct}%)**")

        for idx, q in enumerate(questions):
            chosen_idx = mcq_state["answers"].get(idx, 0)
            is_correct = chosen_idx == q.correct_index
            chosen_letter = option_letters[chosen_idx]
            correct_letter = option_letters[q.correct_index]
            header = (
                f"✅ Q{idx + 1}: Correct ({correct_letter})"
                if is_correct
                else f"❌ Q{idx + 1}: Incorrect (You chose {chosen_letter} • Correct: {correct_letter})"
            )
            with st.expander(header, expanded=not is_correct):
                st.markdown(f"**Question:** {q.question}")
                st.markdown(f"**Correct Answer:** {correct_letter}. {q.options[q.correct_index]}")
                st.markdown(f"**Explanation:** {q.explanation}")


# ---------------------------------------------------------------------------
# Step-by-Step Solver page
# ---------------------------------------------------------------------------

def _render_solver_page(client, model_status: ModelStatus, kb: LocalKnowledgeBase) -> None:
    render_model_banner(model_status)
    subject = st.session_state.selected_subject
    settings: AppSettings = st.session_state.settings

    st.markdown(
        f"""
        <div class="glass-card">
            <div style="font-weight:700;font-size:1.05rem;color:#f8fafc;">Step-by-Step GATE Problem Solver</div>
            <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.2rem;">
                Breaks down numerical and technical questions into Given, To Find, Formula, Steps, Final Answer, and Method Rationale.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "solver_input" not in st.session_state:
        st.session_state.solver_input = ""

    sample_col1, sample_col2 = st.columns([2, 1])
    with sample_col1:
        st.caption(f"Subject context: **{subject}**")
    with sample_col2:
        if st.button("Load Sample Problem", use_container_width=True):
            st.session_state.solver_input = SOLVER_SAMPLES.get(subject, "")
            st.rerun()

    problem = st.text_area(
        "Enter numerical or technical GATE question",
        key="solver_input",
        height=135,
        placeholder="Paste a numerical GATE question, recurrence relation, SQL/relational problem, or derivation...",
    )
    use_rag = st.checkbox(
        "Check uploaded PDFs for relevant formulas/context",
        value=len(kb.documents) > 0,
        disabled=len(kb.documents) == 0,
        key="solver_use_rag",
    )

    if st.button("🧮 Solve Step-by-Step", type="primary"):
        if not problem.strip():
            st.warning("Please enter a problem statement first.")
        elif not model_status.ai_available:
            st.error(
                f"AI unavailable ({INTENDED_MODEL_NAME} not connected). "
                f"{model_status.status_message} {model_status.action_hint}"
            )
        else:
            with st.spinner(f"Solving step-by-step with {model_status.active_model}..."):
                res = answer_solver_query(
                    client=client,
                    kb=kb,
                    problem_text=problem.strip(),
                    subject=subject,
                    use_rag=use_rag,
                    top_k=settings.top_k_retrieval,
                    min_score=settings.min_similarity_score,
                    max_tokens=1200,
                )
            st.session_state.solver_result = res

    if st.session_state.solver_result:
        res = st.session_state.solver_result
        st.markdown("---")
        if res.success:
            st.markdown(res.answer)
            render_grounding_badge(
                rag_used=res.rag_used,
                sources=res.sources,
                model_used=res.model_used,
                is_fallback=res.is_fallback,
                latency_ms=res.latency_ms,
                retrieved_chunks=res.retrieved_chunks,
            )
        else:
            st.error(res.error)


# ---------------------------------------------------------------------------
# Upload & Knowledge Base page
# ---------------------------------------------------------------------------

def _render_upload_kb_page(model_status: ModelStatus, kb: LocalKnowledgeBase) -> None:
    st.markdown(
        """
        <div class="glass-card">
            <div style="font-weight:700;font-size:1.05rem;color:#f8fafc;">Knowledge Base &amp; Study Material (Local RAG)</div>
            <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.2rem;">
                Integrates extracted GATE study materials from <code>data/</code> and newly uploaded PDFs.
                Content is chunked locally, embedded with Sentence Transformers, and indexed in FAISS for grounded RAG retrieval
                (not model fine-tuning).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kb_status = kb.get_status()
    render_status_panel(model_status, kb_status, st.session_state.selected_subject)

    kb_act1, kb_act2 = st.columns([2, 1])
    with kb_act1:
        if st.button(
            "🔄 Rebuild Index from Local Data & Uploads",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Reading local GATE files, chunking, and rebuilding FAISS index..."):
                summary = kb.ingest_local_data(preserve_uploads=True)
            st.success(
                f"Knowledge base rebuilt: {summary['docs_indexed']} indexed document(s) "
                f"({summary['docs_total']} total), {summary['chunks_total']} chunk(s)."
            )
            for warn in summary.get("warnings", []):
                st.warning(warn)
            st.rerun()
    with kb_act2:
        if kb.documents and st.button("🗑️ Clear Index", use_container_width=True):
            kb.clear()
            st.rerun()

    uploaded_files = st.file_uploader(
        "Upload Additional GATE Study PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("📥 Process & Add Uploaded PDFs"):
        for uf in uploaded_files:
            with st.spinner(f"Extracting and indexing '{uf.name}'..."):
                res = kb.add_pdf(filename=uf.name, pdf_bytes=uf.read())
            if res.success:
                st.success(
                    f"Indexed **{res.filename}** ({res.page_count} page(s), {len(res.chunks)} chunk(s))."
                )
            else:
                st.error(res.error)

    st.markdown("#### Knowledge Base Documents")
    if not kb.documents:
        st.info(
            "No documents are currently indexed. Click **Rebuild Index from Local Data & Uploads** "
            "or upload PDFs above."
        )
    else:
        for doc in list(kb.documents):
            d_col1, d_col2 = st.columns([4.5, 1])
            with d_col1:
                tag = doc.get("subject_tag", "Study Material")
                origin = doc.get("origin", "local")
                status_label = doc.get("status", "Indexed")
                skipped_note = (
                    f", {doc['skipped_pages']} low-text page(s) skipped"
                    if doc.get("skipped_pages")
                    else ""
                )
                st.markdown(
                    f"**📄 {doc['filename']}** `{tag}` — {doc['pages']} page(s), "
                    f"**{doc['chunk_count']} chunk(s)**{skipped_note} • *{status_label}* (`{origin}`)"
                )
            with d_col2:
                if st.button("Remove", key=f"del_{doc['doc_id']}", use_container_width=True):
                    kb.delete_document(doc["doc_id"])
                    st.rerun()

    st.markdown("#### Test Knowledge Base Retrieval")
    test_q = st.text_input(
        "Ask a test retrieval query against your indexed GATE documents",
        placeholder="e.g., GATE DA probability syllabus, Eigenvalues, Python dictionary, Pigeonhole principle",
    )
    if test_q.strip():
        hits = kb.search(
            query=test_q.strip(),
            top_k=st.session_state.settings.top_k_retrieval,
            min_score=0.05,
        )
        if not hits:
            st.warning("No matching chunks found for that query.")
        else:
            for idx, hit in enumerate(hits, start=1):
                with st.expander(
                    f"Match {idx}: {hit['source']} (Page {hit['page']}) • score {hit['score']:.3f}",
                    expanded=(idx == 1),
                ):
                    st.write(hit["text"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="GATE AI Tutor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    _init_session_state()

    settings: AppSettings = st.session_state.settings
    kb: LocalKnowledgeBase = st.session_state.kb

    # Build client with num_ctx from settings (memory-safe default: 2048)
    from llm.ollama_client import OllamaClient
    if settings.llm_provider == "ollama":
        client = OllamaClient(
            base_url=settings.ollama_base_url,
            model_id=settings.model_id,
            allow_explicit_fallback=settings.allow_explicit_fallback,
            fallback_model_id=settings.fallback_model_id,
            timeout=settings.request_timeout,
            num_ctx=settings.num_ctx,
        )
    else:
        client = get_llm_client(settings)

    model_status = client.check_status()
    kb_status = kb.get_status()

    _render_sidebar(client, model_status, kb)
    render_top_header(
        active_page=st.session_state.active_page,
        subject=st.session_state.selected_subject,
        explanation_mode=st.session_state.explanation_mode,
        model_status=model_status,
        kb_status=kb_status,
    )

    page = st.session_state.active_page
    if page == "Ask AI":
        _render_ask_ai_page(client, model_status, kb)
    elif page == "Learn Concepts":
        _render_learn_concepts_page(client, model_status, kb)
    elif page == "Practice MCQs":
        _render_mcq_page(client, model_status, kb)
    elif page == "Step-by-Step Solver":
        _render_solver_page(client, model_status, kb)
    elif page == "Upload Study Material":
        _render_upload_kb_page(model_status, kb)


if __name__ == "__main__":
    main()
