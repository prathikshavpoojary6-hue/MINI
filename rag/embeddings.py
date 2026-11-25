"""Embedding engine using Sentence Transformers with graceful TF-IDF fallback."""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class EmbeddingEngine:
    """Generates L2-normalized dense embeddings for chunks and queries.

    Uses SentenceTransformers (`all-MiniLM-L6-v2` by default) when available.
    Falls back cleanly to scikit-learn TF-IDF vectors if SentenceTransformers or
    local/online model weights are unavailable, so the app never crashes.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._st_model = None
        self._load_attempted = False
        self._backend_mode = "uninitialized"
        self._status_note = ""
        self._tfidf: TfidfVectorizer | None = None

    def _ensure_backend(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            # Prefer local cached model so runtime stays 100% local/offline
            try:
                self._st_model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self._st_model = SentenceTransformer(self.model_name)

            self._backend_mode = "sentence_transformers"
            self._status_note = f"SentenceTransformer ({self.model_name}, local)"
        except Exception as exc:
            self._st_model = None
            self._backend_mode = "tfidf_fallback"
            self._status_note = (
                f"TF-IDF Lexical Fallback (SentenceTransformer unavailable: {exc.__class__.__name__})"
            )

    @property
    def is_neural(self) -> bool:
        self._ensure_backend()
        return self._backend_mode == "sentence_transformers"

    @property
    def backend_label(self) -> str:
        self._ensure_backend()
        return self._status_note

    def fit_and_embed_corpus(self, texts: List[str]) -> np.ndarray:
        """Embed an entire list of corpus chunk strings as float32 L2-normalized vectors."""
        self._ensure_backend()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if self._backend_mode == "sentence_transformers" and self._st_model is not None:
            try:
                vecs = self._st_model.encode(
                    texts,
                    batch_size=64,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                return np.asarray(vecs, dtype=np.float32)
            except Exception as exc:
                self._backend_mode = "tfidf_fallback"
                self._status_note = f"TF-IDF Fallback (embedding error: {exc.__class__.__name__})"

        # TF-IDF fallback fits across corpus so query vectors share exact vocabulary dimension
        self._tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=1024,
        )
        matrix = self._tfidf.fit_transform(texts)
        dense = normalize(matrix, norm="l2").toarray().astype(np.float32)
        return dense

    def embed_query(self, query: str, corpus_texts: List[str] | None = None) -> np.ndarray:
        """Embed a single search query matching the corpus vector space."""
        self._ensure_backend()
        if self._backend_mode == "sentence_transformers" and self._st_model is not None:
            try:
                vec = self._st_model.encode(
                    [query],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                return np.asarray(vec, dtype=np.float32)
            except Exception:
                pass

        if self._tfidf is None:
            if not corpus_texts:
                return np.zeros((1, 1), dtype=np.float32)
            self._tfidf = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=1024,
            )
            self._tfidf.fit(corpus_texts)

        vec = self._tfidf.transform([query])
        return normalize(vec, norm="l2").toarray().astype(np.float32)
