# ==============================================================================
# CHAINFLOW AI — app/core/config.py
# All configuration from environment variables with sensible defaults.
# ==============================================================================

from __future__ import annotations
import os
from pathlib import Path

# ── Artifact paths ─────────────────────────────────────────────────────────────
# Set ARTIFACTS_ROOT to wherever you extracted chainflow_artifacts.zip
ARTIFACTS_ROOT = Path(os.getenv("ARTIFACTS_ROOT", "./artifacts"))

MODEL_DIR  = Path(os.getenv("MODEL_DIR",  str(ARTIFACTS_ROOT / "models")))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(ARTIFACTS_ROOT / "chromadb")))
RESULTS_DIR= Path(os.getenv("RESULTS_DIR",str(ARTIFACTS_ROOT / "results")))
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", "/tmp/chainflow_memory"))

for d in [MODEL_DIR, CHROMA_DIR, RESULTS_DIR, MEMORY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LLM ────────────────────────────────────────────────────────────────────────
LLM_MODEL_ID  = os.getenv("LLM_MODEL_ID", "nvidia/Llama3-ChatQA-1.5-7B")
LLM_USE_4BIT  = os.getenv("LLM_USE_4BIT", "true").lower() == "true"
LLM_MAX_CTX   = int(os.getenv("LLM_MAX_CTX",  "4096"))
LLM_MAX_TOKENS= int(os.getenv("LLM_MAX_TOKENS","512"))

# ── Inference ──────────────────────────────────────────────────────────────────
SEQ_LEN = int(os.getenv("SEQ_LEN", "21"))

# ── RAG ────────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# Priority-ordered collections for retrieval.
# prediction_memory is checked first so fresh prediction context surfaces.
RAG_COLLECTIONS = [
    "chainflow_prediction_memory",
    "chainflow_final_results",
    "chainflow_unified",
    "chainflow_manual_ml",
    "chainflow_autogluon",
    "chainflow_deep_learning",
    "chainflow_diagnosis",
    "chainflow_production_model",
    "chainflow_plots",
    "supply_chain_engineered",
    "supply_chain_collated",
]

# ── Retraining thresholds ──────────────────────────────────────────────────────
MAPE_DRIFT_THRESHOLD    = float(os.getenv("MAPE_DRIFT_THRESHOLD", "2.0"))
MIN_NEW_ROWS_TO_RETRAIN = int(os.getenv("MIN_NEW_ROWS_TO_RETRAIN", "5000"))

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST    = os.getenv("API_HOST", "0.0.0.0")
API_PORT    = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

SYSTEM_PROMPT = (
    "You are ChainFlow AI's supply chain analytics assistant. "
    "You have detailed ML training results, model performance metrics, "
    "feature engineering analysis, diagnostic findings, and live prediction "
    "results for a demand forecasting system. "
    "Answer precisely using the provided context. "
    "Always cite specific numbers (R-squared, MAE, MAPE, demand units) when available. "
    "If a user has just uploaded prediction data, refer to the prediction results "
    "in your answer. If context is insufficient, say so clearly."
)
