# ==============================================================================
# CHAINFLOW AI — app/routers/predict.py
# ==============================================================================

from __future__ import annotations

import io
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core import state
from app.core.config import SEQ_LEN
from app.services.prediction import engineer_features, run_prediction

router = APIRouter(prefix="/predict", tags=["Prediction"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRow(BaseModel):
    sku:            str   = Field(..., example="SKU-001")
    location:       str   = Field(..., example="LOC-01")
    date:           str   = Field(..., example="2026-03-20")
    inventory:      float = Field(..., example=450.0)
    lead_time_days: int   = Field(..., example=7)
    unit_cost:      float = Field(..., example=45.0)
    promo_flag:     int   = Field(0)
    holiday_flag:   int   = Field(0)
    model:          str   = Field("lgb")


class BatchRequest(BaseModel):
    rows:       List[PredictRow]
    session_id: str = Field("api_batch")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_row_df(req: PredictRow) -> pd.DataFrame:
    return pd.DataFrame([{
        "date"          : req.date,
        "sku"           : req.sku,
        "location"      : req.location,
        "inventory"     : req.inventory,
        "lead_time_days": req.lead_time_days,
        "unit_cost"     : req.unit_cost,
        "promo_flag"    : req.promo_flag,
        "holiday_flag"  : req.holiday_flag,
    }])


def _lgb_predict(X_scaled: np.ndarray) -> np.ndarray:
    R = state.R
    if not R["lgb"]:
        raise HTTPException(503, "LightGBM not loaded")
    return np.clip(np.round(R["lgb"].predict(X_scaled)), 0, None).astype(int)


def _cnn_predict(X_scaled: np.ndarray) -> float:
    R = state.R
    seq    = np.tile(X_scaled, (SEQ_LEN, 1))
    tensor = torch.tensor(seq).unsqueeze(0).to(R["device"])
    with torch.no_grad():
        ps = R["cnn_lstm"](tensor).cpu().numpy()
    return float(R["scaler_y"].inverse_transform(ps.reshape(-1,1))[0,0])


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/single")
def predict_single(req: PredictRow):
    """Single SKU demand forecast from JSON body."""
    R  = state.R
    t0 = time.time()
    df = _build_row_df(req)
    try:
        df_eng = engineer_features(df)
    except ValueError as e:
        raise HTTPException(422, str(e))

    X = df_eng[R["features"]].values.astype("float32")
    X_sc = R["scaler_X"].transform(X)

    if req.model == "cnn_lstm" and R["cnn_lstm"]:
        pred = max(0, round(_cnn_predict(X_sc)))
    else:
        pred = int(_lgb_predict(X_sc)[0])

    R["inference_count"] += 1
    return {
        "sku"             : req.sku,
        "location"        : req.location,
        "date"            : req.date,
        "demand_forecast" : pred,
        "model_used"      : req.model,
        "latency_ms"      : round((time.time()-t0)*1000, 2),
    }


@router.post("/batch")
def predict_batch(req: BatchRequest):
    """Batch forecast from JSON list. Returns predictions + LLM memory update."""
    R  = state.R
    t0 = time.time()

    rows = [_build_row_df(r) for r in req.rows]
    import pandas as _pd
    df_all = _pd.concat(rows, ignore_index=True)

    try:
        result = run_prediction(df_all, session_id=req.session_id)
    except Exception as e:
        raise HTTPException(500, str(e))

    R["inference_count"] += len(req.rows)
    return {
        "session_id"     : req.session_id,
        "count"          : result["stats"]["n_rows"],
        "stats"          : result["stats"],
        "sku_summary"    : result["sku_summary"],
        "loc_summary"    : result["loc_summary"],
        "output_file"    : result["output_file"],
        "llm_memory"     : "injected",
        "latency_ms"     : round((time.time()-t0)*1000, 2),
    }


@router.post("/upload")
async def predict_from_upload(
    file:       UploadFile = File(..., description="CSV with supply chain data"),
    session_id: str        = Form("upload_session"),
):
    """
    User uploads a CSV file → model predicts demand → results injected into
    LLM memory → user can chat about the predictions immediately.

    Required CSV columns: date, sku, location, inventory, lead_time_days, unit_cost
    Optional: promo_flag, holiday_flag
    """
    R = state.R
    if not R["lgb"]:
        raise HTTPException(503, "Model not loaded")
    if not file.filename.endswith(".csv"):
        raise HTTPException(422, "Only CSV files are accepted")

    t0 = time.time()

    # Read uploaded bytes
    contents = await file.read()
    import io as _io
    try:
        df_raw = pd.read_csv(_io.BytesIO(contents), low_memory=False)
    except Exception as e:
        raise HTTPException(422, f"Could not parse CSV: {e}")

    if len(df_raw) == 0:
        raise HTTPException(422, "Uploaded file is empty")

    try:
        result = run_prediction(df_raw, session_id=session_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

    R["inference_count"] += result["stats"]["n_rows"]

    return {
        "session_id"  : session_id,
        "filename"    : file.filename,
        "rows_predicted": result["stats"]["n_rows"],
        "stats"       : result["stats"],
        "sku_summary" : result["sku_summary"],
        "loc_summary" : result["loc_summary"],
        "output_file" : result["output_file"],
        "llm_memory"  : "injected — ask the LLM about these results",
        "latency_ms"  : round((time.time()-t0)*1000, 2),
        "message"     : (
            f"Predictions ready. {result['stats']['n_rows']} rows processed. "
            f"Total forecast demand: {result['stats']['total_demand']:,} units. "
            f"Now chat with the LLM at /llm/query or /llm/chat to explore results."
        ),
    }
