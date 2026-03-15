# ==============================================================================
# CHAINFLOW AI — app/routers/system.py
# ==============================================================================

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from typing import Optional

import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.core import state

router = APIRouter(tags=["System"])


@router.get("/health")
def health():
    return {
        "status"         : "ok",
        "loaded_at"      : state.R["loaded_at"],
        "ready"          : state.readiness(),
        "device"         : state.R["device"],
        "inference_count": state.R["inference_count"],
    }


@router.get("/model/info")
def model_info():
    meta = state.R.get("metadata") or {}
    chroma = state.R.get("chroma")
    col_count = 0
    if chroma:
        try:
            col_count = len(chroma.list_collections())
        except Exception:
            pass
    return {
        "best_model"         : meta.get("best_model"),
        "r2"                 : meta.get("best_r2"),
        "mae"                : meta.get("best_mae"),
        "mape_pct"           : meta.get("best_mape"),
        "n_features"         : meta.get("n_features"),
        "trained_at"         : meta.get("timestamp"),
        "prd_r2_achieved"    : meta.get("prd_r2_achieved"),
        "prd_mape_achieved"  : meta.get("prd_mape_achieved"),
        "chromadb_collections": col_count,
        "inference_count"    : state.R["inference_count"],
    }


@router.get("/collections")
def list_collections():
    chroma = state.R.get("chroma")
    if not chroma:
        raise HTTPException(503, "ChromaDB not loaded")
    result = {}
    for c in chroma.list_collections():
        try:
            result[c.name] = chroma.get_collection(c.name).count()
        except Exception:
            result[c.name] = -1
    return {"collections": result, "total": len(result)}


class RetrainRequest(BaseModel):
    reason:    str           = Field("manual_trigger")
    data_path: Optional[str] = None


@router.post("/retrain/trigger")
async def trigger_retrain(req: RetrainRequest, bg: BackgroundTasks):
    if state.R["retrain_status"] == "running":
        raise HTTPException(409, "Retraining already in progress")
    state.R["retrain_status"]  = "running"
    state.R["retrain_started"] = datetime.now().isoformat()
    bg.add_task(_run_retrain, req.reason, req.data_path)
    return {"status": "queued", "reason": req.reason,
            "started": state.R["retrain_started"]}


@router.get("/retrain/status")
def retrain_status():
    return {
        "status" : state.R["retrain_status"],
        "started": state.R["retrain_started"],
    }


async def _run_retrain(reason: str, data_path: Optional[str]):
    import logging
    logger = logging.getLogger("chainflow.retrain")
    logger.info(f"Retraining: {reason}")
    try:
        cmd = ["python", "scripts/train_pipeline.py", "--mode", "retrain"]
        if data_path:
            cmd += ["--data_path", data_path]
        subprocess.run(cmd, check=True)
        state.R["retrain_status"] = "completed"
        await state.load_all()   # hot-reload artifacts
        logger.info("Retrain complete — artifacts reloaded")
    except Exception as e:
        state.R["retrain_status"] = f"failed: {e}"
        logger.error(f"Retrain error: {e}")


@router.get("/metrics")
def metrics():
    gpu_mem = 0.0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
    return {
        "chainflow_inference_total" : state.R["inference_count"],
        "chainflow_gpu_memory_gb"   : round(gpu_mem, 3),
        "chainflow_llm_loaded"      : int(state.R["llm"] is not None),
        "chainflow_retrain_status"  : state.R["retrain_status"],
    }
