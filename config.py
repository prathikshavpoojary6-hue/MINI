"""Central configuration and settings persistence for GATE AI Tutor."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"
SETTINGS_FILE = DATA_DIR / "settings.json"

INTENDED_MODEL_NAME = "Ministral 3 3B"
DEFAULT_OLLAMA_MODEL = "ministral-3:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GATE_SUBJECTS: List[str] = [
    "Linear Algebra",
    "Probability and Statistics",
    "Calculus and Optimization",
    "Programming",
    "Data Structures and Algorithms",
    "Database Management Systems",
    "Machine Learning",
    "Artificial Intelligence",
]

SUBJECT_TOPICS: Dict[str, List[str]] = {
    "Linear Algebra": [
        "Vector Spaces, Rank & Nullity",
        "Systems of Linear Equations",
        "Eigenvalues & Eigenvectors",
        "Singular Value Decomposition (SVD)",
        "Orthogonality & Projections",
    ],
    "Probability and Statistics": [
        "Bayes' Theorem & Conditional Probability",
        "Random Variables & Distributions",
        "Expectation, Variance & Covariance",
        "Central Limit Theorem & Hypothesis Testing",
        "Maximum Likelihood Estimation (MLE)",
    ],
    "Calculus and Optimization": [
        "Limits, Continuity & Differentiability",
        "Partial Derivatives & Gradients",
        "Taylor Series & Maxima/Minima",
        "Gradient Descent & Convexity",
        "Lagrange Multipliers",
    ],
    "Programming": [
        "Python Data Structures & Functions",
        "Recursion & Scope Rules",
        "Pointers, Memory & Arrays in C",
        "Time & Space Complexity Basics",
        "Object-Oriented Programming Concepts",
    ],
    "Data Structures and Algorithms": [
        "Asymptotic Complexity & Recurrences",
        "Binary Search & Divide and Conquer",
        "Trees, BST, AVL & Heaps",
        "Graphs: BFS, DFS & Shortest Paths",
        "Dynamic Programming & Greedy Algorithms",
    ],
    "Database Management Systems": [
        "ER Model & Relational Algebra",
        "SQL Queries, Joins & Aggregation",
        "Normalization (1NF to BCNF)",
        "Transactions, ACID & Concurrency",
        "Indexing & B+ Trees",
    ],
    "Machine Learning": [
        "Supervised vs Unsupervised Learning",
        "Bias-Variance Tradeoff & Regularization",
        "Linear & Logistic Regression",
        "PCA vs LDA (Dimensionality Reduction)",
        "SVM, Decision Trees & Ensemble Methods",
    ],
    "Artificial Intelligence": [
        "Uninformed Search (BFS, DFS, UCS)",
        "Informed Search (A*, Heuristics)",
        "Adversarial Search & Alpha-Beta Pruning",
        "Propositional & First-Order Logic",
        "Bayesian Networks & Probabilistic Reasoning",
    ],
}

SUGGESTED_QUESTIONS: List[str] = [
    "Explain Bayes' theorem with a GATE example.",
    "What is the difference between PCA and LDA?",
    "Explain gradient descent step by step.",
    "Generate 5 probability MCQs.",
    "Explain the time complexity of binary search.",
]

SOLVER_SAMPLES: Dict[str, str] = {
    "Linear Algebra": (
        "Find the eigenvalues and eigenvectors of the matrix A = [[4, 1], [2, 3]]."
    ),
    "Probability and Statistics": (
        "A bag contains 4 red and 6 blue balls. Two balls are drawn without replacement. "
        "Given that the second ball drawn is red, find the probability that the first ball was also red."
    ),
    "Calculus and Optimization": (
        "Perform two iterations of gradient descent starting from x0 = 4 with learning rate alpha = 0.25 "
        "to minimize f(x) = (x - 1)^2 + 3."
    ),
    "Programming": (
        "Trace the output and recursion depth of f(n): if n <= 1 return 1; else return f(n-1) + f(n-2) for n = 5."
    ),
    "Data Structures and Algorithms": (
        "Solve the recurrence T(n) = 2T(n/2) + n log n using the recursion tree or Master theorem variant."
    ),
    "Database Management Systems": (
        "Given relation R(A, B, C, D) with functional dependencies A -> B, B -> C, C -> D, "
        "find all candidate keys and identify the highest normal form."
    ),
    "Machine Learning": (
        "A binary classifier produces TP=45, FP=5, FN=10, TN=40 on a test set of 100 samples. "
        "Compute Precision, Recall, Specificity, and F1-score."
    ),
    "Artificial Intelligence": (
        "In an A* search problem, explain whether heuristic h(n) must be admissible or consistent "
        "to guarantee optimality in graph search, and trace f(n)=g(n)+h(n) on a small 4-node graph."
    ),
}


def load_dotenv_file(env_path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from a local .env file without extra dependencies."""
    target = env_path or (BASE_DIR / ".env")
    if not target.exists():
        return
    try:
        for raw_line in target.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


@dataclass
class AppSettings:
    """Configurable runtime settings for GATE AI Tutor."""

    llm_provider: str = "ollama"
    model_id: str = DEFAULT_OLLAMA_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_URL
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = ""
    allow_explicit_fallback: bool = False
    fallback_model_id: str = ""
    temperature: float = 0.25
    max_tokens: int = 1200
    request_timeout: int = 120
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = 750
    chunk_overlap: int = 120
    top_k_retrieval: int = 4
    min_similarity_score: float = 0.35
    lora_adapter_note: str = (
        "Future LoRA/QLoRA fine-tuned Ministral 3 3B adapters can be served via an "
        "Ollama Modelfile (ADAPTER ./adapter.gguf) or OpenAI-compatible vLLM endpoint."
    )

    @classmethod
    def load(cls) -> "AppSettings":
        """Load settings from environment variables and persisted settings.json."""
        load_dotenv_file()
        defaults = cls()
        if SETTINGS_FILE.exists():
            try:
                saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                for field_name in asdict(defaults):
                    if field_name in saved and saved[field_name] is not None:
                        setattr(defaults, field_name, saved[field_name])
            except (OSError, json.JSONDecodeError):
                pass

        # Environment variables override file defaults when explicitly set
        env_provider = os.getenv("GATE_LLM_PROVIDER")
        env_model = os.getenv("GATE_LLM_MODEL")
        env_ollama_url = os.getenv("OLLAMA_BASE_URL")
        env_openai_url = os.getenv("GATE_OPENAI_BASE_URL")
        env_openai_key = os.getenv("GATE_OPENAI_API_KEY")
        env_embed = os.getenv("GATE_EMBEDDING_MODEL")

        if env_provider:
            defaults.llm_provider = env_provider.strip()
        if env_model:
            defaults.model_id = env_model.strip()
        if env_ollama_url:
            defaults.ollama_base_url = env_ollama_url.strip().rstrip("/")
        if env_openai_url:
            defaults.openai_base_url = env_openai_url.strip().rstrip("/")
        if env_openai_key:
            defaults.openai_api_key = env_openai_key.strip()
        if env_embed:
            defaults.embedding_model = env_embed.strip()

        # Safety guard: never allow Llama as the configured default if accidentally empty
        if not defaults.model_id.strip():
            defaults.model_id = DEFAULT_OLLAMA_MODEL

        return defaults

    def save(self) -> None:
        """Persist settings to data/settings.json."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(
            json.dumps(asdict(self), indent=2),
            encoding="utf-8",
        )
