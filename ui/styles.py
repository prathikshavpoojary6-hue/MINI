"""Premium dark-navy/charcoal educational dashboard CSS for Streamlit."""

from __future__ import annotations

import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #090d1a;
    --bg-secondary: #11182c;
    --bg-card: #141c33;
    --bg-card-hover: #192441;
    --border-subtle: rgba(148, 163, 184, 0.14);
    --border-accent: rgba(99, 102, 241, 0.45);
    --accent-indigo: #6366f1;
    --accent-violet: #8b5cf6;
    --accent-cyan: #38bdf8;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

code, pre, .stCode {
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
}

/* Main App Canvas */
.stApp {
    background: radial-gradient(circle at 18% 0%, rgba(99, 102, 241, 0.12), transparent 36%),
                radial-gradient(circle at 85% 12%, rgba(139, 92, 246, 0.10), transparent 32%),
                linear-gradient(180deg, #080c17 0%, #0b1020 100%);
    color: var(--text-primary);
}

/* Clean up default Streamlit chrome while keeping sidebar toggle accessible */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stDeployButton"] {display: none;}
header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1080px;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1324 0%, #0a0f1e 100%);
    border-right: 1px solid var(--border-subtle);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

.brand-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(139, 92, 246, 0.12));
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 14px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.9rem;
}

.brand-title {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f8fafc;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

.brand-subtitle {
    font-size: 0.76rem;
    color: #cbd5e1;
    margin-top: 0.25rem;
    opacity: 0.88;
}

.sidebar-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin: 0.9rem 0 0.4rem 0.15rem;
}

/* Top Dashboard Header */
.top-header-bar {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    background: rgba(20, 28, 51, 0.72);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 0.85rem 1.15rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}

.header-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: center;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.28rem 0.68rem;
    border-radius: 999px;
    border: 1px solid var(--border-subtle);
    background: rgba(15, 23, 42, 0.8);
    color: #cbd5e1;
}

.pill-indigo {
    border-color: rgba(99, 102, 241, 0.45);
    background: rgba(99, 102, 241, 0.16);
    color: #c7d2fe;
}

.pill-cyan {
    border-color: rgba(56, 189, 248, 0.4);
    background: rgba(56, 189, 248, 0.12);
    color: #bae6fd;
}

.pill-green {
    border-color: rgba(16, 185, 129, 0.45);
    background: rgba(16, 185, 129, 0.14);
    color: #a7f3d0;
}

.pill-amber {
    border-color: rgba(245, 158, 11, 0.45);
    background: rgba(245, 158, 11, 0.14);
    color: #fde68a;
}

.pill-red {
    border-color: rgba(239, 68, 68, 0.45);
    background: rgba(239, 68, 68, 0.14);
    color: #fecaca;
}

/* Welcome Hero */
.welcome-hero {
    background: linear-gradient(145deg, rgba(20, 28, 51, 0.92), rgba(15, 22, 41, 0.95));
    border: 1px solid rgba(99, 102, 241, 0.28);
    border-radius: 18px;
    padding: 2rem 1.8rem 1.6rem 1.8rem;
    margin: 0.6rem 0 1.25rem 0;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
}

.welcome-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #a5b4fc;
    background: rgba(99, 102, 241, 0.16);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
    margin-bottom: 0.85rem;
}

.welcome-title {
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1.22;
    letter-spacing: -0.025em;
    color: #f8fafc;
    margin: 0 0 0.5rem 0;
}

.welcome-subtitle {
    font-size: 1.02rem;
    color: #94a3b8;
    margin: 0;
}

/* Cards & Metrics */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.85rem;
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.status-tile {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 0.8rem 0.95rem;
}

.status-tile-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    margin-bottom: 0.25rem;
}

.status-tile-value {
    font-size: 0.92rem;
    font-weight: 700;
    color: #e2e8f0;
    word-break: break-word;
}

/* Notice Banners */
.notice-banner {
    border-radius: 12px;
    padding: 0.85rem 1.05rem;
    margin-bottom: 1rem;
    border: 1px solid;
    font-size: 0.88rem;
    line-height: 1.45;
}

.notice-warning {
    background: rgba(245, 158, 11, 0.10);
    border-color: rgba(245, 158, 11, 0.35);
    color: #fde68a;
}

.notice-danger {
    background: rgba(239, 68, 68, 0.10);
    border-color: rgba(239, 68, 68, 0.35);
    color: #fecaca;
}

.notice-info {
    background: rgba(56, 189, 248, 0.10);
    border-color: rgba(56, 189, 248, 0.32);
    color: #bae6fd;
}

/* Chat Messages & Grounding Badges */
[data-testid="stChatMessage"] {
    background: rgba(20, 28, 51, 0.72);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
}

.grounding-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
    margin-top: 0.6rem;
    padding-top: 0.55rem;
    border-top: 1px solid rgba(148, 163, 184, 0.12);
    font-size: 0.75rem;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: 1px solid var(--border-subtle);
    background: rgba(20, 28, 51, 0.9);
    color: #e2e8f0;
    font-weight: 600;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    border-color: var(--border-accent);
    background: rgba(99, 102, 241, 0.18);
    color: #ffffff;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    border: none;
    color: #ffffff;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4f46e5, #6d28d9);
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
    background-color: #11182c !important;
    border-color: rgba(148, 163, 184, 0.22) !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
}

/* MCQ Question Card */
.mcq-card {
    background: rgba(20, 28, 51, 0.85);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
}

.mcq-q-title {
    font-weight: 700;
    font-size: 0.98rem;
    color: #f8fafc;
    margin-bottom: 0.4rem;
}

/* Sidebar Radio Styling */
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.10);
    border-radius: 10px;
    padding: 0.42rem 0.65rem;
    margin-bottom: 0.28rem;
    transition: all 0.15s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    border-color: rgba(99, 102, 241, 0.42);
    background: rgba(99, 102, 241, 0.12);
}
</style>
"""


def inject_custom_css() -> None:
    """Inject the dark educational dashboard stylesheet."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
