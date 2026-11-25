"""Automated tests for GATE AI Tutor core modules (offline-safe & deterministic)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF

from config import DATA_DIR, GATE_SUBJECTS, SUGGESTED_QUESTIONS, AppSettings
from features.chat_service import answer_chat_query, answer_solver_query
from features.mcq_generator import generate_mcqs, parse_mcq_json
from llm.base import BaseLLMClient, LLMResponse, ModelStatus
from llm.ollama_client import OllamaClient
from prompts.gate_prompts import MODE_GATE_EXAM, MODE_SIMPLE_LEARNING
from rag.extracted_loader import load_local_data_folder
from rag.pdf_processor import extract_and_chunk_pdf
from rag.vector_store import LocalKnowledgeBase


class MockMinistralClient(BaseLLMClient):
    """Deterministic test double simulating a connected Ministral 3 3B client."""

    def __init__(self, response_text: str = "Mock Ministral response") -> None:
        self.response_text = response_text
        self.last_messages: List[Dict[str, str]] = []

    def list_models(self) -> List[str]:
        return ["ministral-3:3b"]

    def check_status(self) -> ModelStatus:
        return ModelStatus(
            provider="ollama",
            intended_model="Ministral 3 3B",
            configured_model="ministral-3:3b",
            active_model="ministral-3:3b",
            ollama_connected=True,
            model_connected=True,
            ai_available=True,
            available_models=["ministral-3:3b"],
            status_message="Connected to ministral-3:3b.",
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.25,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        self.last_messages = messages
        return LLMResponse(
            success=True,
            content=self.response_text,
            model_used="ministral-3:3b",
            is_fallback=False,
            latency_ms=42,
        )


def _make_sample_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestGateAITutor(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="gate_tutor_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_subjects_and_default_model(self) -> None:
        settings = AppSettings()
        self.assertEqual(settings.model_id, "ministral-3:3b")
        self.assertNotIn("llama", settings.model_id.lower())
        self.assertEqual(len(GATE_SUBJECTS), 8)
        self.assertEqual(len(SUGGESTED_QUESTIONS), 5)

    def test_ollama_blocks_llama_default_and_handles_unreachable(self) -> None:
        llama_client = OllamaClient(model_id="llama3:8b")
        status = llama_client.check_status()
        self.assertFalse(status.ai_available)
        self.assertIn("blocked", status.status_message.lower())

        offline_client = OllamaClient(
            base_url="http://127.0.0.1:59999",
            model_id="ministral-3:3b",
        )
        off_status = offline_client.check_status()
        self.assertFalse(off_status.ollama_connected)
        self.assertFalse(off_status.ai_available)

    def test_pdf_extraction_and_error_handling(self) -> None:
        empty_res = extract_and_chunk_pdf("empty.pdf", b"")
        self.assertFalse(empty_res.success)

        bad_res = extract_and_chunk_pdf("corrupt.pdf", b"not a real pdf")
        self.assertFalse(bad_res.success)

        sample_pdf = _make_sample_pdf_bytes(
            "Bayes theorem relates posterior probability P(A|B) to likelihood P(B|A) and prior P(A)."
        )
        ok_res = extract_and_chunk_pdf("bayes_notes.pdf", sample_pdf)
        self.assertTrue(ok_res.success)
        self.assertEqual(ok_res.page_count, 1)
        self.assertGreaterEqual(len(ok_res.chunks), 1)
        self.assertEqual(ok_res.chunks[0]["source"], "bayes_notes.pdf")

    def test_rag_knowledge_base_persistence_and_retrieval(self) -> None:
        kb = LocalKnowledgeBase(storage_dir=self.temp_dir / "kb")
        # Force TF-IDF in unit test so test never depends on external HuggingFace download
        kb.embedder._load_attempted = True
        kb.embedder._backend_mode = "tfidf_fallback"
        kb.embedder._status_note = "TF-IDF Test Mode"

        pdf_bytes = _make_sample_pdf_bytes(
            "Principal Component Analysis (PCA) is unsupervised and maximizes variance. "
            "Linear Discriminant Analysis (LDA) is supervised and maximizes class separability."
        )
        add_res = kb.add_pdf("ml_dim_reduction.pdf", pdf_bytes)
        self.assertTrue(add_res.success)
        self.assertEqual(kb.get_status().document_count, 1)

        hits = kb.search("difference between PCA and LDA", top_k=2, min_score=0.05)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["source"], "ml_dim_reduction.pdf")

    def test_chat_and_solver_grounding_honesty(self) -> None:
        kb = LocalKnowledgeBase(storage_dir=self.temp_dir / "kb2")
        kb.embedder._load_attempted = True
        kb.embedder._backend_mode = "tfidf_fallback"
        kb.embedder._status_note = "TF-IDF Test Mode"

        client = MockMinistralClient("Detailed answer.")

        # Without uploaded documents -> rag_used must be False
        res_general = answer_chat_query(
            client=client,
            kb=kb,
            query="Explain eigenvalues",
            subject="Linear Algebra",
            mode=MODE_SIMPLE_LEARNING,
            use_rag=True,
        )
        self.assertTrue(res_general.success)
        self.assertFalse(res_general.rag_used)
        self.assertEqual(res_general.sources, [])
        self.assertIn("NO RELEVANT EXCERPTS WERE FOUND", client.last_messages[0]["content"])

        # With uploaded document matching query -> rag_used must be True with exact source
        kb.add_pdf(
            "la_notes.pdf",
            _make_sample_pdf_bytes("Eigenvalues satisfy det(A - lambda I) = 0 for square matrix A."),
        )
        res_grounded = answer_solver_query(
            client=client,
            kb=kb,
            problem_text="Find eigenvalues using det(A - lambda I) = 0",
            subject="Linear Algebra",
            use_rag=True,
            min_score=0.05,
        )
        self.assertTrue(res_grounded.success)
        self.assertTrue(res_grounded.rag_used)
        self.assertEqual(res_grounded.sources[0]["source"], "la_notes.pdf")
        self.assertIn("GATE Exam Mode" if False else "STEP-BY-STEP", client.last_messages[0]["content"])

    def test_mcq_parsing_and_generation(self) -> None:
        sample_json = """```json
        {
          "questions": [
            {
              "question": "What is the time complexity of binary search on a sorted array of size n?",
              "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
              "correct_index": "B",
              "explanation": "Each step halves the search interval."
            }
          ]
        }
        ```"""
        parsed = parse_mcq_json(sample_json)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].correct_index, 1)
        self.assertEqual(parsed[0].options[1], "O(log n)")

        client = MockMinistralClient(sample_json)
        mcq_res = generate_mcqs(
            client=client,
            kb=None,
            subject="Data Structures and Algorithms",
            topic="Binary Search",
        )
        self.assertTrue(mcq_res.success)
        self.assertEqual(len(mcq_res.questions), 1)

    def test_extracted_data_loader_and_live_faiss_retrieval(self) -> None:
        gz_path = DATA_DIR / "extracted_.json.gz"
        self.assertTrue(gz_path.exists(), "Expected data/extracted_.json.gz to exist")
        orig_size = gz_path.stat().st_size

        # 1. Verify read-only loader parses all 12 documents and chunks meaningfully
        corpus = load_local_data_folder(DATA_DIR)
        self.assertEqual(len(corpus.documents), 12)
        self.assertGreater(len(corpus.chunks), 1500)
        self.assertEqual(gz_path.stat().st_size, orig_size)

        # 2. Verify persisted local FAISS + SentenceTransformer knowledge base retrieval
        kb = LocalKnowledgeBase()
        status = kb.get_status()
        self.assertEqual(status.document_count, 12)
        self.assertGreater(status.chunk_count, 1500)
        self.assertIn("FAISS", status.index_backend)

        # Query 1: GATE DA Syllabus
        hits_syllabus = kb.search("Probability and Statistics Linear Algebra GATE DA syllabus", top_k=4, min_score=0.35)
        self.assertGreaterEqual(len(hits_syllabus), 1)
        self.assertTrue(any("GATE _DA_2025_Syllabus" in h["source"] for h in hits_syllabus))

        # Query 2: Python textbook
        hits_python = kb.search("How do Python dictionaries and lists work?", top_k=4, min_score=0.35)
        self.assertGreaterEqual(len(hits_python), 1)
        self.assertTrue(any("python" in h["source"].lower() for h in hits_python))

        # Query 3: Discrete structures textbook
        hits_discrete = kb.search("Explain the pigeonhole principle in discrete structures", top_k=4, min_score=0.35)
        self.assertGreaterEqual(len(hits_discrete), 1)
        self.assertTrue(any("discrete" in h["source"].lower() for h in hits_discrete))

        # Negative control: off-topic query should not cross 0.35 threshold
        hits_offtopic = kb.search("What is the recipe for chocolate chip cookies?", top_k=3, min_score=0.35)
        self.assertEqual(hits_offtopic, [])

    def test_streamlit_app_renders_without_ollama(self) -> None:
        from streamlit.testing.v1 import AppTest

        app_path = str(Path(__file__).resolve().parent.parent / "app.py")
        at = AppTest.from_file(app_path, default_timeout=20)
        at.run()
        self.assertFalse(at.exception)

        # Click a suggested question when Ollama is unavailable -> must not crash
        suggested_btns = [b for b in at.button if b.key and b.key.startswith("suggested_q_")]
        self.assertEqual(len(suggested_btns), 5)
        suggested_btns[0].click().run()
        self.assertFalse(at.exception)

        # Verify sidebar navigation pages can be switched cleanly without crashing
        for page_name in [
            "Learn Concepts",
            "Practice MCQs",
            "Step-by-Step Solver",
            "Upload Study Material",
            "Ask AI",
        ]:
            at.sidebar.radio("nav_radio").set_value(page_name).run()
            self.assertFalse(at.exception, f"Page {page_name} raised an exception")


if __name__ == "__main__":
    unittest.main()
