# GATE AI Tutor

A syllabus-based AI Tutor web application for **GATE (Graduate Aptitude Test in Engineering)** preparation, tailored for **GATE AI/ML (DA) & CS** aspirants and built around **Ministral 3 3B** with modular local inference and PDF Retrieval-Augmented Generation (RAG).

---

## Key Features

- **Ask AI (Chat Tutor)**: Session-persistent multi-chat interface with Markdown, LaTeX equations, code blocks, and GATE syllabus focus.
- **Simple Learning Mode & GATE Exam Mode**:
  - *Simple Learning Mode*: Beginner-friendly explanations, clear definitions, analogies, step-by-step breakdowns, and a GATE example.
  - *GATE Exam Mode*: Structured into Definition, Concept, Formula, Explanation, Example, and Important Exam Points.
- **Learn Concepts**: Curated syllabus topic explorer across all 8 GATE subjects plus custom concept lookup.
- **Practice MCQs**: Generates 5 GATE-style 4-option MCQs with interactive answer selection, automatic scoring, and post-submission explanations.
- **Step-by-Step Solver**: Solves numerical and technical problems using a 6-part structure (Given, To Find, Formula/Method, Steps, Final Answer, Why It Works).
- **PDF Knowledge Base (RAG)**: Extracts text from uploaded study PDFs via PyMuPDF, chunks text with page metadata, generates embeddings via Sentence Transformers (with TF-IDF fallback), indexes locally with FAISS (with NumPy cosine fallback), and displays exact source document names & page numbers when retrieved context is used.
- **Strict Model & Citation Honesty**:
  - Designed around **Ministral 3 3B** (`ministral-3:3b`). Never silently switches to Llama.
  - Opens cleanly even when Ollama or Ministral 3 3B is unavailable, displaying clear connection status and setup instructions.
  - Explicitly labels answers as **Document-Grounded** (with filenames/pages) vs **General Model Knowledge**.

---

## Tech Stack

- **Frontend**: Streamlit (custom dark navy/charcoal educational dashboard UI)
- **LLM Inference**: Ministral 3 3B via Ollama REST API (modular `BaseLLMClient` supporting Ollama & OpenAI-compatible endpoints)
- **PDF Processing**: PyMuPDF (`fitz`)
- **Embeddings & Vector Store**: Sentence Transformers (`all-MiniLM-L6-v2`) + FAISS (`IndexFlatIP`), with graceful TF-IDF + NumPy fallbacks

---

## Project Structure

```text
.
├── app.py                     # Main Streamlit application entry point
├── config.py                  # Settings, GATE subjects, sample problems, env loading
├── requirements.txt           # Python dependencies
├── .env.example               # Environment configuration template
├── llm/
│   ├── base.py                # Abstract BaseLLMClient, ModelStatus, LLMResponse
│   ├── ollama_client.py       # Ollama & OpenAI-compatible clients (Ministral policy)
│   └── factory.py             # Modular provider factory
├── prompts/
│   └── gate_prompts.py        # GATE Focus, Simple/Exam mode, Solver, and MCQ prompts
├── rag/
│   ├── pdf_processor.py       # PyMuPDF extraction & overlapping text chunker
│   ├── embeddings.py          # SentenceTransformer embedder + TF-IDF fallback
│   └── vector_store.py        # Persistent FAISS / NumPy local knowledge base
├── features/
│   ├── chat_service.py        # Grounded chat & step-by-step solver orchestration
│   └── mcq_generator.py       # Structured 5-MCQ generation & resilient JSON parsing
├── ui/
│   ├── styles.py              # Custom dark theme CSS
│   └── components.py          # Welcome screen, headers, grounding badges, status grid
└── tests/
    └── test_core.py           # Automated test suite
```

---

## Setup & Running

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama & Ministral 3 3B (For Live AI Responses)
1. Install Ollama from https://ollama.com/download
2. *(Optional on low-space C: drives)* Set `OLLAMA_MODELS` to another drive with free space (e.g. `E:\ollama_models`).
3. Start Ollama and pull **Ministral 3 3B**:
   ```bash
   ollama serve
   ollama pull ministral-3:3b
   ```
   Verify installed models with:
   ```bash
   ollama list
   ```

### 3. Run the Application
```bash
streamlit run app.py
```

---

## Configuration & Future LoRA / QLoRA Support

Configure the model via environment variables (see `.env.example`), `data/settings.json`, or the **Settings & Diagnostics** panel inside the sidebar:

- `GATE_LLM_MODEL` (default: `ministral-3:3b`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `GATE_LLM_PROVIDER` (`ollama` or `openai_compatible`)

### Upgrading or Plugging in a Fine-Tuned LoRA / QLoRA Model Later
Because all model calls pass through `llm/base.py` (`BaseLLMClient`) and `llm/factory.py`:
1. Fine-tune Ministral 3 3B with LoRA/QLoRA externally and export a GGUF adapter or merged model.
2. Create an Ollama `Modelfile`:
   ```text
   FROM ministral-3:3b
   ADAPTER ./gate-ministral-lora.gguf
   ```
   Then run `ollama create ministral-3:3b-gate -f Modelfile` and set `GATE_LLM_MODEL=ministral-3:3b-gate` — zero UI changes required.
3. Alternatively, host with vLLM and switch `GATE_LLM_PROVIDER=openai_compatible`.
