# ==============================================================================
# CHAINFLOW AI — app/core/state.py
# Global model registry. Loaded once at startup, shared across all requests.
# ==============================================================================

from __future__ import annotations

import gc
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import numpy as np
import torch

from app.core.config import (
    MODEL_DIR, CHROMA_DIR, LLM_MODEL_ID, LLM_USE_4BIT,
)

logger = logging.getLogger("chainflow.state")

# ── Registry ──────────────────────────────────────────────────────────────────
R: Dict[str, Any] = {
    "lgb"             : None,
    "cnn_lstm"        : None,
    "scaler_X"        : None,
    "scaler_y"        : None,
    "features"        : None,
    "metadata"        : None,
    "llm"             : None,
    "tokenizer"       : None,
    "embedder"        : None,
    "chroma"          : None,
    "device"          : "cpu",
    "loaded_at"       : None,
    "inference_count" : 0,
    "retrain_status"  : "idle",
    "retrain_started" : None,
}

# ==============================================================================
# LOADER FUNCTIONS
# ==============================================================================

def _reset_chroma_singleton():
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        for attr in list(vars(SharedSystemClient)):
            val = getattr(SharedSystemClient, attr)
            if isinstance(val, dict) and not attr.startswith("__"):
                val.clear()
    except Exception:
        pass


def load_ml(device: torch.device):
    lgb_path  = MODEL_DIR / "lgb_model.pkl"
    feat_path = MODEL_DIR / "feature_names.json"
    meta_path = MODEL_DIR / "model_metadata.json"

    if lgb_path.exists():
        R["lgb"] = joblib.load(lgb_path)
        logger.info("LightGBM loaded")

    if feat_path.exists():
        with open(feat_path) as f:
            R["features"] = json.load(f)

    # CNN+LSTM
    pt_path = MODEL_DIR / "cnn_lstm_production.pt"
    if pt_path.exists() and R["features"]:
        from app.core.model_def import CNN_LSTM
        cnn = CNN_LSTM(n_feat=len(R["features"]))
        state = torch.load(pt_path, map_location=device)
        cnn.load_state_dict(state)
        cnn.eval().to(device)
        R["cnn_lstm"] = cnn
        logger.info("CNN+LSTM loaded")

    for key, fname in [("scaler_X", "scaler_X.pkl"), ("scaler_y", "scaler_y.pkl")]:
        p = MODEL_DIR / fname
        if p.exists():
            R[key] = joblib.load(p)

    if meta_path.exists():
        with open(meta_path) as f:
            R["metadata"] = json.load(f)

    logger.info(
        f"ML: lgb={'✓' if R['lgb'] else '✗'} | "
        f"cnn_lstm={'✓' if R['cnn_lstm'] else '✗'} | "
        f"features={len(R['features'] or [])} | "
        f"scalers={'✓' if R['scaler_X'] else '✗'}"
    )


def load_chromadb():
    try:
        import chromadb
        _reset_chroma_singleton()
        R["chroma"] = chromadb.PersistentClient(path=str(CHROMA_DIR))
        cols = [c.name for c in R["chroma"].list_collections()]
        logger.info(f"ChromaDB: {len(cols)} collections at {CHROMA_DIR}")
    except Exception as e:
        logger.error(f"ChromaDB: {e}")


def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        R["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedder: all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Embedder: {e}")


def load_llm(device: torch.device):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        logger.info(f"Loading LLM: {LLM_MODEL_ID}")
        R["tokenizer"] = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

        if str(device) == "cuda" and LLM_USE_4BIT:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            R["llm"] = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID,
                quantization_config=bnb,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            R["llm"] = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        R["llm"].eval()
        logger.info("LLM loaded ✓")
    except Exception as e:
        logger.error(f"LLM load failed: {e}")


async def load_all():
    logger.info("=" * 60)
    logger.info("CHAINFLOW AI — STARTUP")
    logger.info("=" * 60)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R["device"] = str(device)
    logger.info(f"Device: {device}")
    load_ml(device)
    load_chromadb()
    load_embedder()
    load_llm(device)
    R["loaded_at"] = datetime.now().isoformat()
    logger.info(f"Startup complete in {time.time()-t0:.1f}s")


def release():
    R["llm"]      = None
    R["cnn_lstm"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Resources released")


def readiness() -> Dict[str, bool]:
    return {
        "lgb"     : R["lgb"]      is not None,
        "cnn_lstm": R["cnn_lstm"] is not None,
        "llm"     : R["llm"]      is not None,
        "chromadb": R["chroma"]   is not None,
        "scalers" : R["scaler_X"] is not None,
        "features": R["features"] is not None,
    }
