# ==============================================================================
# CHAINFLOW AI — app/services/prediction.py
# Prediction pipeline: CSV upload → features → inference → LLM memory.
# Mirrors Cell 14 logic but runs on the production server (no Colab).
# ==============================================================================

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.core import state
from app.core.config import CHROMA_DIR, RESULTS_DIR, MEMORY_DIR


# ── Feature list (loaded from artifacts at startup) ───────────────────────────
def _feats() -> List[str]:
    return state.R.get("features") or []


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same 40-feature engineering as Cell 10.
    Handles partial history gracefully via fillna.
    """
    df = df.copy()
    required = ["date", "sku", "location", "inventory",
                "lead_time_days", "unit_cost"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku", "location", "date"]).reset_index(drop=True)

    df["sku_enc"] = df["sku"].astype("category").cat.codes.astype(np.int16)
    df["loc_enc"] = df["location"].astype("category").cat.codes.astype(np.int8)

    df["year"]         = df["date"].dt.year.astype(np.int16)
    df["month"]        = df["date"].dt.month.astype(np.int8)
    df["day"]          = df["date"].dt.day.astype(np.int8)
    df["dow"]          = df["date"].dt.dayofweek.astype(np.int8)
    df["doy"]          = df["date"].dt.dayofyear.astype(np.int16)
    df["week"]         = df["date"].dt.isocalendar().week.astype(np.int8)
    df["quarter"]      = df["date"].dt.quarter.astype(np.int8)
    df["is_weekend"]   = (df["dow"] >= 5).astype(np.int8)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(np.int8)
    df["month_sin"]    = np.sin(2*np.pi*df["month"]/12).astype(np.float32)
    df["month_cos"]    = np.cos(2*np.pi*df["month"]/12).astype(np.float32)
    df["dow_sin"]      = np.sin(2*np.pi*df["dow"]/7).astype(np.float32)
    df["dow_cos"]      = np.cos(2*np.pi*df["dow"]/7).astype(np.float32)
    df["doy_sin"]      = np.sin(2*np.pi*df["doy"]/365).astype(np.float32)
    df["doy_cos"]      = np.cos(2*np.pi*df["doy"]/365).astype(np.float32)

    if "promo_flag"   not in df.columns: df["promo_flag"]   = 0
    if "holiday_flag" not in df.columns: df["holiday_flag"] = 0

    grp = df.groupby(["sku", "location"])
    for lag in [1, 3, 7, 14]:
        df[f"inv_lag{lag}"] = grp["inventory"].shift(lag).fillna(df["inventory"]).astype(np.float32)
        df[f"lt_lag{lag}"]  = grp["lead_time_days"].shift(lag).fillna(df["lead_time_days"]).astype(np.float32)
    for win in [7, 14, 30]:
        df[f"inv_ma{win}"]  = grp["inventory"].transform(
            lambda x: x.shift(1).rolling(win, min_periods=1).mean()
        ).fillna(df["inventory"]).astype(np.float32)
        df[f"inv_std{win}"] = grp["inventory"].transform(
            lambda x: x.shift(1).rolling(win, min_periods=1).std()
        ).fillna(0).astype(np.float32)

    df["inv_change"]   = grp["inventory"].diff().fillna(0).astype(np.float32)
    df["inv_chg_pct"]  = grp["inventory"].pct_change().fillna(0).clip(-5,5).astype(np.float32)
    df["cost_ma7"]     = grp["unit_cost"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    ).fillna(df["unit_cost"]).astype(np.float32)

    sku_inv = df.groupby("sku_enc")["inventory"].transform("mean").astype(np.float32)
    loc_inv = df.groupby("loc_enc")["inventory"].transform("mean").astype(np.float32)
    df["sku_inv_ratio"] = (df["inventory"] / (sku_inv + 1e-6)).astype(np.float32)
    df["loc_inv_ratio"] = (df["inventory"] / (loc_inv + 1e-6)).astype(np.float32)

    # Ensure all expected features exist
    for feat in _feats():
        if feat not in df.columns:
            df[feat] = 0.0

    return df


def run_prediction(df_input: pd.DataFrame, session_id: str) -> Dict:
    """
    Core prediction: engineer → scale → predict → summarise → inject memory.
    Returns the full result dict (also saved to RESULTS_DIR).
    """
    R = state.R
    t0 = time.time()

    df_eng   = engineer_features(df_input)
    feats    = _feats()
    X        = df_eng[feats].values.astype(np.float32)
    X_scaled = R["scaler_X"].transform(X)
    preds_r  = R["lgb"].predict(X_scaled)
    preds    = np.clip(np.round(preds_r), 0, None).astype(int)
    elapsed  = time.time() - t0

    df_eng["predicted_demand"] = preds
    df_eng["session_id"]       = session_id
    df_eng["predicted_at"]     = datetime.now().isoformat()

    stats = {
        "n_rows"      : int(len(preds)),
        "mean_demand" : round(float(preds.mean()), 2),
        "std_demand"  : round(float(preds.std()),  2),
        "min_demand"  : int(preds.min()),
        "max_demand"  : int(preds.max()),
        "total_demand": int(preds.sum()),
        "latency_ms"  : round(elapsed * 1000, 1),
    }

    sku_summary = loc_summary = None
    if "sku" in df_eng.columns:
        sku_summary = (
            df_eng.groupby("sku")["predicted_demand"]
            .agg(["mean","min","max","sum"]).round(1).reset_index()
            .rename(columns={"mean":"avg","sum":"total"})
            .to_dict("records")
        )
    if "location" in df_eng.columns:
        loc_summary = (
            df_eng.groupby("location")["predicted_demand"]
            .agg(["mean","min","max","sum"]).round(1).reset_index()
            .rename(columns={"mean":"avg","sum":"total"})
            .to_dict("records")
        )

    # Save CSV
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"predictions_{session_id}_{ts}.csv"
    out_cols = ["date","sku","location","inventory","lead_time_days",
                "unit_cost","predicted_demand"]
    df_eng[[c for c in out_cols if c in df_eng.columns]].to_csv(out_path, index=False)

    result = {
        "session_id" : session_id,
        "timestamp"  : datetime.now().isoformat(),
        "output_file": str(out_path),
        "stats"      : stats,
        "sku_summary": sku_summary,
        "loc_summary": loc_summary,
    }

    # Inject into LLM memory (ChromaDB)
    _inject_memory(result)
    return result


def _inject_memory(result: Dict):
    """
    Embeds prediction result as a ChromaDB document so the LLM
    can reference it in follow-up conversations.
    """
    R     = state.R
    stats = result["stats"]
    meta  = R.get("metadata") or {}

    sku_lines = ""
    if result["sku_summary"]:
        top5 = sorted(result["sku_summary"], key=lambda x: -x["total"])[:5]
        sku_lines = ("Top 5 SKUs by forecast demand: "
                     + "; ".join(f"{r['sku']}={int(r['total'])} units (avg {r['avg']}/day)"
                                 for r in top5) + ". ")

    loc_lines = ""
    if result["loc_summary"]:
        loc_lines = ("Demand by location: "
                     + "; ".join(f"{r['location']}={int(r['total'])} units"
                                 for r in result["loc_summary"]) + ". ")

    doc = (
        f"[PREDICTION RESULT — Session {result['session_id']}] "
        f"Processed at: {result['timestamp']}. "
        f"Rows predicted: {stats['n_rows']:,}. "
        f"Model: {meta.get('best_model','LightGBM')} "
        f"(R²={meta.get('best_r2','?')}, MAE={meta.get('best_mae','?')}). "
        f"Demand stats: mean={stats['mean_demand']}, std={stats['std_demand']}, "
        f"min={stats['min_demand']}, max={stats['max_demand']}, "
        f"total={stats['total_demand']:,} units. "
        f"{sku_lines}{loc_lines}"
        f"Latency: {stats['latency_ms']}ms. "
        f"Full CSV: {result['output_file']}."
    )

    if not R["chroma"] or not R["embedder"]:
        # Fallback: JSON file
        mem_path = MEMORY_DIR / f"mem_{result['session_id']}_{datetime.now().strftime('%H%M%S')}.json"
        mem_path.write_text(json.dumps({"doc": doc, "result": result}, indent=2))
        return

    try:
        try:
            col = R["chroma"].get_collection("chainflow_prediction_memory")
        except Exception:
            col = R["chroma"].create_collection("chainflow_prediction_memory")

        emb = R["embedder"].encode([doc], show_progress_bar=False).tolist()
        doc_id = f"pred_{result['session_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        col.upsert(
            documents=[doc],
            metadatas=[{
                "session_id"  : result["session_id"],
                "n_rows"      : str(stats["n_rows"]),
                "total_demand": str(stats["total_demand"]),
                "mean_demand" : str(stats["mean_demand"]),
                "output_file" : result["output_file"],
                "timestamp"   : result["timestamp"],
                "type"        : "prediction_result",
            }],
            embeddings=emb,
            ids=[doc_id],
        )
    except Exception as e:
        import logging
        logging.getLogger("chainflow.prediction").warning(f"Memory injection failed: {e}")
