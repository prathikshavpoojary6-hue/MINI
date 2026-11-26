# GATE AI Tutor

A syllabus-based, locally running AI Tutor web application for **GATE (Graduate Aptitude Test in Engineering)** preparation—tailored for **GATE Data Science & Artificial Intelligence (DA)** and **Computer Science (CS)** aspirants. Built around **Ministral 3 3B** (`ministral-3:3b`) via local Ollama inference and local **Retrieval-Augmented Generation (RAG)** with **FAISS** and **Sentence Transformers**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Key Features](#key-features)
4. [Supported GATE Subjects](#supported-gate-subjects)
5. [Knowledge Base & Local RAG Pipeline](#knowledge-base--local-rag-pipeline)
6. [16 GB RAM Memory Optimizations](#16-gb-ram-memory-optimizations)
7. [Model Policy & Source Honesty](#model-policy--source-honesty)
8. [Tech Stack](#tech-stack)
9. [Project Structure](#project-structure)
10. [Setup & Running (Online & Completely Offline)](#setup--running-online--completely-offline)
11. [Configuration & Environment Variables](#configuration--environment-variables)
12. [Testing](#testing)
13. [Future LoRA / QLoRA Fine-Tuning Support](#future-lora--qlora-fine-tuning-support)

---

## Project Overview

**GATE AI Tutor** helps engineering students and GATE aspirants:
- Understand core GATE AI/ML and CS concepts in either **Simple Learning Mode** or **GATE Exam Mode**.
- Ask technical follow-up questions in a **streaming multi-chat interface** with session history.
- Solve numerical and theoretical GATE problems step by step.
- Practice with auto-generated, 4-option **GATE-style MCQs** with automatic grading and explanations.
- Ground answers in local GATE syllabus documents, textbooks, and uploaded study PDFs using **FAISS vector search** with exact **filename and page number citations** (RAG—not model fine-tuning).
- Run **100% locally and offline** without any cloud API dependencies.

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                       Streamlit Frontend (app.py)                       │
│  Dark-themed UI (ui/styles.py, ui/components.py)                        │
│  Pages: Ask AI │ Learn Concepts │ Practice MCQs │ Solver │ Knowledge Base│
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────────┐                     ┌───────────────────────────┐
│  Feature Orchestrators│                     │   Local Knowledge Base    │
│  • chat_service.py    │◄─── Top-K Chunks ───│   (rag/vector_store.py)   │
│  • mcq_generator.py   │    (min_score=0.35) │   • extracted_loader.py   │
│  • gate_prompts.py    │                     │   • pdf_processor.py      │
└───────────┬───────────┘                     │   • embeddings.py         │
            │                                 │     (all-MiniLM-L6-v2)    │
            ▼                                 │   • FAISS IndexFlatIP     │
┌───────────────────────┐                     └───────────────────────────┘
│  Modular LLM Backend  │
│  • llm/base.py        │
│  • llm/factory.py     │
│  • llm/ollama_client.py
│    └─► Local Ollama REST API (http://localhost:11434)
│        Model: ministral-3:3b (num_ctx=2048, streaming enabled)
└───────────────────────┘
```

---

## Key Features

### 1. Ask AI (Streaming Chat Tutor)
- **Live Token Streaming**: Uses `OllamaClient.chat_stream()` and Streamlit's `st.write_stream()` so responses stream token-by-token.
- **Session-Persistent Multi-Chat**: Create new chats (`＋ New Chat`), switch between up to 6 recent session chats in the sidebar, or clear the active chat. Sends the last 8 conversation turns for coherent follow-up questions.
- **Welcome Screen**: Displays 5 clickable suggested GATE questions (*Bayes' theorem*, *PCA vs LDA*, *Gradient descent*, *Probability MCQs*, *Binary search time complexity*).
- **Rich Formatting**: Supports Markdown, LaTeX math (`$...$` inline and `$$...$$` display), and code blocks.

### 2. Two Explanation Modes
Switchable anytime from the sidebar:
- **Simple Learning Mode**: Beginner-friendly intuition, clear definitions of technical terms, relatable analogies, numbered steps, and a GATE-relevant example.
- **GATE Exam Mode**: High-yield exam structure with 6 sections:
  1. `Definition`
  2. `Concept`
  3. `Formula` (LaTeX with defined symbols)
  4. `Explanation`
  5. `Example`
  6. `Important Exam Points`

### 3. Learn Concepts
- Interactive syllabus explorer with **5 curated core topics per GATE subject** plus a custom concept/theorem input box.
- Optionally grounds explanations in the local Knowledge Base.

### 4. Practice MCQs
- Generates **5 GATE-style 4-option MCQs** for any selected subject, optional topic focus, and difficulty (`GATE Easy`, `GATE Standard`, `GATE Tricky`).
- Interactive radio-button quiz form with **Submit Answers**, automatic score calculation (`score / 5` and percentage), and expandable per-question explanations.
- Resilient JSON parsing in `features/mcq_generator.py` handles fenced or raw JSON outputs cleanly.

### 5. Step-by-Step Solver
- Includes a **Load Sample Problem** button tailored to each of the 8 GATE subjects.
- Structures solutions into 6 rigorous sections:
  1. **Given Information**
  2. **To Find**
  3. **Formula / Method Selected**
  4. **Step-by-Step Solution**
  5. **Final Answer**
  6. **Why This Method Works**

### 6. Upload Study Material & Knowledge Base Diagnostics
- Displays a live **System Status** grid: Selected Model Name, Model Connection, Ollama Connection, AI Availability, Uploaded Documents & Chunk Count, Knowledge Base Backend (`FAISS` / `NumPy`), Current Subject, and Embedder Backend.
- Allows uploading additional GATE study PDFs, removing individual documents, clearing the index, rebuilding the index from `data/`, and running live **Test Knowledge Base Retrieval** queries with similarity scores.

---

## Supported GATE Subjects

1. **Linear Algebra** — Vector Spaces, Rank & Nullity, Systems of Linear Equations, Eigenvalues & Eigenvectors, SVD, Orthogonality & Projections
2. **Probability and Statistics** — Bayes' Theorem & Conditional Probability, Random Variables & Distributions, Expectation/Variance/Covariance, CLT & Hypothesis Testing, MLE
3. **Calculus and Optimization** — Limits/Continuity/Differentiability, Partial Derivatives & Gradients, Taylor Series & Maxima/Minima, Gradient Descent & Convexity, Lagrange Multipliers
4. **Programming** — Python Data Structures & Functions, Recursion & Scope Rules, Pointers/Memory/Arrays in C, Time & Space Complexity Basics, OOP Concepts
5. **Data Structures and Algorithms** — Asymptotic Complexity & Recurrences, Binary Search & Divide and Conquer, Trees/BST/AVL/Heaps, Graphs (BFS, DFS, Shortest Paths), Dynamic Programming & Greedy
6. **Database Management Systems** — ER Model & Relational Algebra, SQL Queries/Joins/Aggregation, Normalization (1NF–BCNF), Transactions/ACID/Concurrency, Indexing & B+ Trees
7. **Machine Learning** — Supervised vs Unsupervised Learning, Bias-Variance Tradeoff & Regularization, Linear & Logistic Regression, PCA vs LDA, SVM/Decision Trees/Ensembles
8. **Artificial Intelligence** — Uninformed Search (BFS, DFS, UCS), Informed Search (A*, Heuristics), Adversarial Search & Alpha-Beta Pruning, Propositional & First-Order Logic, Bayesian Networks

---

## Knowledge Base & Local RAG Pipeline

- **Pre-Extracted Study Material Integration (`rag/extracted_loader.py`)**:
  - Reads local study files from `data/` (including `data/extracted_.json.gz`, `.json`, `.txt`, `.md`, and `.pdf`) in **read-only** mode without overwriting originals.
  - Preserves exact **filenames**, **page numbers**, and **subject tags**, while skipping low-text/empty pages (`skipped_low_text`).
  - Default indexed corpus from `data/extracted_.json.gz`: **12 GATE documents** (including GATE DA/CS syllabi, question papers, Python programming, and Discrete Structures materials) totaling **3,539 chunks**.
- **PDF Upload Processing (`rag/pdf_processor.py`)**:
  - Extracts page-by-page text from uploaded PDFs using **PyMuPDF (`fitz`)**.
  - Splits text into overlapping chunks (`chunk_size=750`, `chunk_overlap=120`) while preserving document name and page metadata.
- **Local Embeddings & Vector Store (`rag/embeddings.py`, `rag/vector_store.py`)**:
  - Generates normalized dense embeddings locally using `SentenceTransformer("all-MiniLM-L6-v2")` (with automatic **TF-IDF** fallback if Sentence Transformers is unavailable).
  - Stores vectors in a local **FAISS `IndexFlatIP`** index (`data/knowledge_base/index.faiss`, `chunks.json`, `documents.json`, `embeddings.npy`), with NumPy cosine-similarity fallback.
  - Uses `faiss.serialize_index` / `faiss.deserialize_index` with `pathlib.Path.write_bytes()` / `read_bytes()` to support Windows paths containing Unicode characters.

---

## 16 GB RAM Memory Optimizations

Designed specifically to run reliably on 16 GB RAM laptops without out-of-memory (OOM) errors:
- **Capped Context Window (`num_ctx=2048`)**: Keeps Ministral 3 3B's memory footprint bounded (~3–4 GB) instead of allocating oversized KV caches. Adjustable (`512`–`4096`) in **Settings & Diagnostics** or via `GATE_NUM_CTX`.
- **KV-Cache Reuse (`num_keep=-1`)**: Lets Ollama manage the KV cache across turns rather than reloading the model for every question.
- **Single Model Instance**: Avoids loading multiple LLMs simultaneously.
- **Bounded Generation & Timeouts**: `max_tokens=1024` and `request_timeout=180` seconds to handle CPU inference cleanly without hanging.

---

## Model Policy & Source Honesty

- **Strict Ministral 3 3B Default**: Configured for `ministral-3:3b`. The client (`llm/ollama_client.py`) actively blocks any `llama` model identifier from being used as the default or silent fallback.
- **Graceful Offline/Disconnected UI**: If Ollama is stopped or `ministral-3:3b` is not installed, the Streamlit UI still launches cleanly and displays a clear status banner with setup instructions.
- **Explicit Grounding Badges**:
  - When FAISS retrieves chunks above `min_similarity_score` (`0.35`): displays **`Document-Grounded (RAG)`** along with exact filenames, page numbers, and an expandable **View Retrieved Study Excerpts** section.
  - When no relevant chunks are found: displays **`General Model Knowledge • Not found in local GATE documents`** and instructs the LLM not to invent citations.
  - This project uses **RAG retrieval**, not model training or fine-tuning on the PDFs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend UI** | Streamlit (`>=1.38.0`), custom dark navy/charcoal CSS |
| **Local LLM** | Ministral 3 3B (`ministral-3:3b`) via Ollama REST API (`/api/tags`, `/api/chat`) |
| **Modular LLM Interface** | `BaseLLMClient` supporting `OllamaClient` and `OpenAICompatibleClient` |
| **PDF Extraction** | PyMuPDF (`fitz >=1.24.0`) |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) + `scikit-learn` TF-IDF fallback |
| **Vector Search** | `faiss-cpu` (`IndexFlatIP`) + `numpy` cosine similarity fallback |
| **HTTP Client** | `requests` |

---

## Project Structure

```text
.
├── app.py                         # Main Streamlit app, 5 pages, callbacks, streaming chat UI
├── config.py                      # AppSettings, GATE subjects/topics, sample problems, .env loader
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment configuration template
├── .gitignore                     # Ignores __pycache__/, .env, .venv/, data/settings.json
├── .streamlit/
│   └── config.toml                # Streamlit dark theme configuration
├── data/
│   ├── extracted_.json.gz         # Pre-extracted GATE study materials (read-only)
│   └── knowledge_base/            # Persisted FAISS index, embeddings.npy, chunks.json, documents.json
├── llm/
│   ├── base.py                    # Abstract BaseLLMClient, ModelStatus, LLMResponse
│   ├── ollama_client.py           # OllamaClient (streaming + num_ctx=2048) & OpenAICompatibleClient
│   └── factory.py                 # Modular provider factory
├── prompts/
│   └── gate_prompts.py            # GATE identity, Simple/Exam mode, Solver, MCQ, & RAG prompt builders
├── rag/
│   ├── extracted_loader.py        # Read-only loader for .json.gz/.json/.txt/.md/.pdf in data/
│   ├── pdf_processor.py           # PyMuPDF text extraction & overlapping chunker
│   ├── embeddings.py              # SentenceTransformer embedder + TF-IDF fallback
│   └── vector_store.py            # LocalKnowledgeBase (FAISS IndexFlatIP + NumPy fallback)
├── features/
│   ├── chat_service.py            # Non-streaming & streaming RAG chat + step-by-step solver
│   └── mcq_generator.py           # 5-MCQ generation & resilient JSON parser
├── ui/
│   ├── styles.py                  # Custom dark navy/charcoal CSS & card styling
│   └── components.py              # Top header, model banner, welcome screen, grounding badges, status grid
└── tests/
    └── test_core.py               # Automated unit & integration tests (8 tests)
```

---

## Setup & Running (Online & Completely Offline)

### 1. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Install Ollama & Pull Ministral 3 3B
1. Download and install Ollama from [https://ollama.com/download](https://ollama.com/download).
2. *(Optional if C: drive is low on space)* Set `OLLAMA_MODELS` to another drive before pulling:
   ```powershell
   $env:OLLAMA_MODELS = "E:\ollama_models"
   ```
3. Start Ollama and pull the exact model:
   ```powershell
   ollama serve
   ollama pull ministral-3:3b
   ollama list
   ```

### 3. Run the Application (Standard)
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 4. Run 100% Offline (No Internet / Wi-Fi Off)
Once `ministral-3:3b` is pulled and `all-MiniLM-L6-v2` is cached locally, set HuggingFace offline flags so `sentence-transformers` loads immediately from cache without attempting network checks:
```powershell
# Terminal 1 (if Ollama is not already running in background)
ollama serve

# Terminal 2
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
streamlit run app.py
```

---

## Configuration & Environment Variables

Settings can be configured via `.env` (copy `.env.example` to `.env`), shell environment variables, `data/settings.json`, or the **Settings & Diagnostics** expander in the sidebar:

| Variable | Default | Description |
|---|---|---|
| `GATE_LLM_PROVIDER` | `ollama` | LLM backend (`ollama` or `openai_compatible`) |
| `GATE_LLM_MODEL` | `ministral-3:3b` | Primary model identifier (Llama defaults are blocked) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server URL |
| `GATE_NUM_CTX` | `2048` | Ollama context window size (keep `2048` for 16 GB RAM) |
| `GATE_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformer model for local RAG |
| `GATE_OPENAI_BASE_URL` | `http://localhost:8000/v1` | Optional OpenAI-compatible server URL |
| `GATE_OPENAI_API_KEY` | *(empty)* | Optional API key for OpenAI-compatible endpoints |

---

## Testing

Run the automated test suite:
```powershell
python -m unittest discover -s tests -v
```

The test suite (`tests/test_core.py`, 8 tests) verifies:
- GATE subjects and default `ministral-3:3b` configuration
- Ollama Llama-blocking policy and unreachable-server handling
- PyMuPDF PDF extraction and invalid-PDF error handling
- Local FAISS persistence and retrieval
- Pre-extracted `data/` ingestion and live FAISS retrieval + negative off-topic threshold check
- Grounding honesty prompts (`NO RELEVANT EXCERPTS WERE FOUND` vs `LOCAL GATE STUDY MATERIAL EXCERPTS`)
- Structured 5-MCQ generation and JSON parsing
- Streamlit UI offline rendering, page navigation, and sidebar history/new-chat callback safety (`AppTest`)

---

## Future LoRA / QLoRA Fine-Tuning Support

No model fine-tuning is performed in this prototype. However, because all inference goes through `BaseLLMClient` (`llm/base.py`) and `get_llm_client()` (`llm/factory.py`), you can plug in a future fine-tuned Ministral 3 3B adapter with zero UI changes:

1. Fine-tune Ministral 3 3B externally using LoRA/QLoRA and export a GGUF adapter or merged weights.
2. Create an Ollama `Modelfile`:
   ```text
   FROM ministral-3:3b
   ADAPTER ./gate-ministral-lora.gguf
   ```
   Register and activate it:
   ```powershell
   ollama create ministral-3:3b-gate -f Modelfile
   ```
   Then set `GATE_LLM_MODEL=ministral-3:3b-gate` in `.env` or Settings.
3. Alternatively, serve the fine-tuned model via **vLLM** and switch `GATE_LLM_PROVIDER=openai_compatible`.
