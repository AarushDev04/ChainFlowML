# ==============================================================================
# CHAINFLOW AI — scripts/train_pipeline.py
# Standalone training orchestrator. Runs independently of Colab.
# Reads data, engineers features, trains LightGBM, evaluates, embeds to ChromaDB.
#
# Usage:
#   python scripts/train_pipeline.py --mode auto
#   python scripts/train_pipeline.py --mode full
#   python scripts/train_pipeline.py --mode retrain --data_path /data/new.csv
#   python scripts/train_pipeline.py --mode predict
#   python scripts/train_pipeline.py --mode embed
# ==============================================================================

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("chainflow.pipeline")

# ── Paths (env overrides for Docker) ─────────────────────────────────────────
ARTIFACTS_ROOT = Path(os.getenv("ARTIFACTS_ROOT", "./artifacts"))
MODEL_DIR      = Path(os.getenv("MODEL_DIR",   str(ARTIFACTS_ROOT / "models")))
DATA_DIR       = Path(os.getenv("DATA_DIR",    "./data"))
CHROMA_DIR     = Path(os.getenv("CHROMA_DIR",  str(ARTIFACTS_ROOT / "chromadb")))
RESULTS_DIR    = Path(os.getenv("RESULTS_DIR", str(ARTIFACTS_ROOT / "results")))
RAW_DIR        = DATA_DIR / "raw"
PROC_DIR       = DATA_DIR / "processed"

for d in [MODEL_DIR, DATA_DIR, CHROMA_DIR, RESULTS_DIR, RAW_DIR, PROC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MAPE_DRIFT_THRESHOLD    = float(os.getenv("MAPE_DRIFT_THRESHOLD", "2.0"))
MIN_NEW_ROWS_TO_RETRAIN = int(os.getenv("MIN_NEW_ROWS_TO_RETRAIN", "5000"))

# ==============================================================================
# DECISION ENGINE
# ==============================================================================

def should_retrain(data_path: Optional[str] = None) -> Tuple[bool, str]:
    lgb_path  = MODEL_DIR / "lgb_model.pkl"
    meta_path = MODEL_DIR / "model_metadata.json"

    if not lgb_path.exists() or not meta_path.exists():
        return True, "no_saved_model"

    if data_path and Path(data_path).exists():
        try:
            n = sum(1 for _ in open(data_path)) - 1
            if n >= MIN_NEW_ROWS_TO_RETRAIN:
                return True, f"new_data ({n:,} rows)"
        except Exception:
            pass

    leaderboard = RESULTS_DIR / "ultimate_model_leaderboard.csv"
    if leaderboard.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            df_r = pd.read_csv(leaderboard)
            drift = float(df_r["MAPE"].min()) - float(meta.get("best_mape", 0))
            if drift > MAPE_DRIFT_THRESHOLD:
                return True, f"mape_drift (+{drift:.2f}pp)"
        except Exception:
            pass

    return False, "saved_model_ok"

# ==============================================================================
# STEP 1 — DATA GENERATION (synthetic, used when no real data supplied)
# ==============================================================================

def step_data_generation() -> Path:
    out = RAW_DIR / "supply_chain_enriched.csv"
    if out.exists():
        n = sum(1 for _ in open(out)) - 1
        logger.info(f"Data already exists ({n:,} rows) — skipping regen")
        return out

    logger.info("Generating synthetic supply chain data...")
    np.random.seed(42)
    NUM_SKUS, NUM_LOCS, NUM_DAYS = 50, 8, 730
    CHUNK = 50_000

    date_range = pd.date_range("2022-01-01", periods=NUM_DAYS, freq="D")
    skus       = np.array([f"SKU-{str(i).zfill(3)}" for i in range(1, NUM_SKUS+1)])
    locs       = np.array([f"LOC-{str(i).zfill(2)}" for i in range(1, NUM_LOCS+1)])
    sku_base   = np.random.randint(40, 160, NUM_SKUS)
    sku_trend  = np.random.uniform(-0.02, 0.05, NUM_SKUS)
    sku_price  = np.random.uniform(10, 100, NUM_SKUS)
    loc_factor = np.random.uniform(0.7, 1.4, NUM_LOCS)
    total      = NUM_SKUS * NUM_LOCS * NUM_DAYS

    first = True
    for ci in range((total // CHUNK) + 1):
        s = ci * CHUNK; e = min(s + CHUNK, total)
        if s >= e: break
        idx   = np.arange(s, e)
        sku_i = (idx // (NUM_LOCS * NUM_DAYS)) % NUM_SKUS
        loc_i = (idx // NUM_DAYS) % NUM_LOCS
        day_i = idx % NUM_DAYS
        dates = date_range[day_i]
        doy, dow, month = dates.dayofyear.values, dates.dayofweek.values, dates.month.values
        wk    = (day_i // 7).astype(float)

        base       = sku_base[sku_i].astype(float)
        seasonal   = 1.0 + 0.35 * np.sin(2*np.pi*doy/365.25)
        weekend    = np.where((dow==5)|(dow==6), 1.25, 1.0)
        trend_mult = 1.0 + sku_trend[sku_i] * wk / 52
        hol        = 1.0 + 0.5*((month==12)&(doy>=340)) + 0.4*((month==11)&(doy>=325)&(doy<=332)) + 0.2*((month>=6)&(month<=8))
        promo      = (np.random.rand(len(idx)) < 0.05).astype(float)
        promo_lift = 1.0 + promo * np.random.uniform(0.2, 0.8, len(idx))

        demand   = np.clip(base*seasonal*weekend*trend_mult*loc_factor[loc_i]*hol*promo_lift*np.random.normal(1,0.08,len(idx)), 5, 500).astype(np.int16)
        lead_t   = np.random.randint(3, 15, len(idx)).astype(np.int8)
        inventory= np.clip(demand*lead_t*0.8 + base*0.5, 50, 2000).astype(np.int16)
        cost     = (sku_price[sku_i]*np.random.uniform(0.9,1.1,len(idx))).astype(np.float32)

        chunk = pd.DataFrame({
            "date":dates, "sku":skus[sku_i], "location":locs[loc_i],
            "demand":demand, "inventory":inventory, "lead_time_days":lead_t,
            "unit_cost":cost, "promo_flag":promo.astype(np.int8),
            "holiday_flag":(hol>1.0).astype(np.int8),
        })
        chunk.to_csv(out, mode="w" if first else "a", header=first, index=False)
        first = False
        del chunk; gc.collect()

    logger.info(f"Data written → {out} ({total:,} rows)")
    return out

# ==============================================================================
# STEP 2 — FEATURE ENGINEERING
# ==============================================================================

def step_feature_engineering(raw_path: Path) -> Tuple[Dict, list]:
    logger.info("Feature engineering...")
    df = pd.read_csv(raw_path, parse_dates=["date"], low_memory=False)
    df = df.sort_values(["sku","location","date"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} rows")

    df["sku_enc"] = df["sku"].astype("category").cat.codes.astype(np.int16)
    df["loc_enc"] = df["location"].astype("category").cat.codes.astype(np.int8)
    df["year"]    = df["date"].dt.year.astype(np.int16)
    df["month"]   = df["date"].dt.month.astype(np.int8)
    df["day"]     = df["date"].dt.day.astype(np.int8)
    df["dow"]     = df["date"].dt.dayofweek.astype(np.int8)
    df["doy"]     = df["date"].dt.dayofyear.astype(np.int16)
    df["week"]    = df["date"].dt.isocalendar().week.astype(np.int8)
    df["quarter"] = df["date"].dt.quarter.astype(np.int8)
    df["is_weekend"]   = (df["dow"]>=5).astype(np.int8)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(np.int8)
    for col, period in [("month",12),("dow",7),("doy",365)]:
        df[f"{col}_sin"] = np.sin(2*np.pi*df[col]/period).astype(np.float32)
        df[f"{col}_cos"] = np.cos(2*np.pi*df[col]/period).astype(np.float32)

    if "promo_flag"   not in df.columns: df["promo_flag"]   = 0
    if "holiday_flag" not in df.columns: df["holiday_flag"] = 0

    grp = df.groupby(["sku","location"])
    for lag in [1,3,7,14]:
        df[f"inv_lag{lag}"] = grp["inventory"].shift(lag).astype(np.float32)
        df[f"lt_lag{lag}"]  = grp["lead_time_days"].shift(lag).astype(np.float32)
    for win in [7,14,30]:
        df[f"inv_ma{win}"]  = grp["inventory"].transform(lambda x: x.shift(1).rolling(win,min_periods=1).mean()).astype(np.float32)
        df[f"inv_std{win}"] = grp["inventory"].transform(lambda x: x.shift(1).rolling(win,min_periods=1).std()).fillna(0).astype(np.float32)
    df["inv_change"]   = grp["inventory"].diff().astype(np.float32)
    df["inv_chg_pct"]  = grp["inventory"].pct_change().clip(-5,5).astype(np.float32)
    df["cost_ma7"]     = grp["unit_cost"].transform(lambda x: x.shift(1).rolling(7,min_periods=1).mean()).astype(np.float32)

    sku_inv = df.groupby("sku_enc")["inventory"].transform("mean").astype(np.float32)
    loc_inv = df.groupby("loc_enc")["inventory"].transform("mean").astype(np.float32)
    df["sku_inv_ratio"] = (df["inventory"]/(sku_inv+1e-6)).astype(np.float32)
    df["loc_inv_ratio"] = (df["inventory"]/(loc_inv+1e-6)).astype(np.float32)

    df = df.dropna().reset_index(drop=True)

    FEATURES = [
        "sku_enc","loc_enc","year","month","day","dow","doy","week","quarter",
        "is_weekend","is_month_end","month_sin","month_cos","dow_sin","dow_cos",
        "doy_sin","doy_cos","inventory","lead_time_days","unit_cost",
        "promo_flag","holiday_flag","inv_lag1","inv_lag3","inv_lag7","inv_lag14",
        "lt_lag1","lt_lag3","lt_lag7","inv_ma7","inv_ma14","inv_ma30",
        "inv_std7","inv_std14","inv_std30","inv_change","inv_chg_pct",
        "cost_ma7","sku_inv_ratio","loc_inv_ratio",
    ]
    TARGET = "demand"
    assert not any("demand" in f for f in FEATURES), "LEAKAGE"

    df = df.sort_values("date").reset_index(drop=True)
    n  = len(df); t1 = int(n*0.70); t2 = int(n*0.85)

    splits = {"train":df.iloc[:t1], "val":df.iloc[t1:t2], "test":df.iloc[t2:]}
    result = {}
    for split, sdf in splits.items():
        result[f"X_{split}"] = sdf[FEATURES].values.astype(np.float32)
        result[f"y_{split}"] = sdf[TARGET].values.astype(np.float32)

    logger.info(f"Features: {len(FEATURES)} | Train:{t1:,} Val:{t2-t1:,} Test:{n-t2:,}")
    return result, FEATURES

# ==============================================================================
# STEP 3 — TRAIN
# ==============================================================================

def step_train(result: Dict, features: list) -> Dict[str, Any]:
    logger.info("Training LightGBM...")
    import joblib
    import lightgbm as lgb
    from sklearn.preprocessing import RobustScaler

    X_tr, y_tr = result["X_train"], result["y_train"]
    X_vl, y_vl = result["X_val"],   result["y_val"]

    sx = RobustScaler(); sy = RobustScaler()
    X_tr_s = sx.fit_transform(X_tr)
    X_vl_s = sx.transform(X_vl)

    ds_tr = lgb.Dataset(X_tr_s, y_tr)
    ds_vl = lgb.Dataset(X_vl_s, y_vl, reference=ds_tr)

    params = {
        "objective":"regression","metric":"mae","n_estimators":2000,
        "learning_rate":0.03,"num_leaves":127,"min_child_samples":20,
        "feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,
        "reg_alpha":0.1,"reg_lambda":0.1,"verbose":-1,"n_jobs":-1,
    }
    t0 = time.time()
    model = lgb.train(params, ds_tr, valid_sets=[ds_vl],
                      callbacks=[lgb.early_stopping(50,verbose=False),
                                 lgb.log_evaluation(-1)])
    logger.info(f"LightGBM done in {time.time()-t0:.1f}s | best_iter={model.best_iteration}")

    joblib.dump(model,  MODEL_DIR/"lgb_model.pkl",  compress=3)
    joblib.dump(sx,     MODEL_DIR/"scaler_X.pkl",   compress=3)
    joblib.dump(sy,     MODEL_DIR/"scaler_y.pkl",   compress=3)
    model.save_model(str(MODEL_DIR/"lgb_final.txt"))
    with open(MODEL_DIR/"feature_names.json","w") as f:
        json.dump(features, f, indent=2)

    return {"model":model,"scaler_X":sx,"scaler_y":sy}

# ==============================================================================
# STEP 4 — EVALUATE
# ==============================================================================

def step_evaluate(trained: Dict, result: Dict) -> Dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
    from scipy.stats import pearsonr

    X_t   = trained["scaler_X"].transform(result["X_test"])
    y_t   = result["y_test"]
    preds = trained["model"].predict(X_t)

    mae  = mean_absolute_error(y_t, preds)
    rmse = float(np.sqrt(mean_squared_error(y_t, preds)))
    r2   = r2_score(y_t, preds)
    mape = mean_absolute_percentage_error(y_t, preds)*100
    smape= 100*np.mean(2*np.abs(preds-y_t)/(np.abs(y_t)+np.abs(preds)+1e-8))
    corr,_=pearsonr(y_t,preds)

    metrics = dict(model="LightGBM",MAE=mae,RMSE=rmse,R2=r2,
                   MAPE=mape,SMAPE=smape,Pearson_r=corr,
                   VarRatio=float(np.std(preds)/np.std(y_t)),
                   Coverage90=float(np.mean(np.abs(y_t-preds)<=1.645*np.std(y_t-preds))*100))

    pd.DataFrame([metrics]).to_csv(RESULTS_DIR/"ultimate_model_leaderboard.csv",index=False)

    meta = {
        "timestamp":datetime.now().isoformat(),"best_model":"LightGBM",
        "best_r2":round(r2,6),"best_mae":round(mae,4),"best_mape":round(mape,4),
        "best_rmse":round(rmse,4),"best_smape":round(smape,4),
        "best_pearson_r":round(float(corr),4),
        "n_features":len(result["X_train"][0]),
        "prd_r2_target":0.75,"prd_mape_target":8.0,
        "prd_r2_achieved":bool(r2>=0.75),"prd_mape_achieved":bool(mape<=8.0),
        "all_models":[metrics],
        "lgb_path":str(MODEL_DIR/"lgb_model.pkl"),
        "scaler_X_path":str(MODEL_DIR/"scaler_X.pkl"),
        "scaler_y_path":str(MODEL_DIR/"scaler_y.pkl"),
        "feature_names_path":str(MODEL_DIR/"feature_names.json"),
        "chromadb_path":str(CHROMA_DIR),
    }
    with open(MODEL_DIR/"model_metadata.json","w") as f:
        json.dump(meta,f,indent=2)

    logger.info(f"R²={r2:.4f}  MAE={mae:.3f}  MAPE={mape:.2f}%  "
                f"PRD_R2={'✓' if r2>=0.75 else '✗'}  PRD_MAPE={'✓' if mape<=8 else '✗'}")
    return meta

# ==============================================================================
# STEP 5 — EMBED
# ==============================================================================

def step_embed(meta: Dict, features: list):
    logger.info("Embedding results into ChromaDB...")
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        try:
            from chromadb.api.shared_system_client import SharedSystemClient
            for a in list(vars(SharedSystemClient)):
                v = getattr(SharedSystemClient,a)
                if isinstance(v,dict) and not a.startswith("__"): v.clear()
        except Exception: pass

        chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        emb    = SentenceTransformer("all-MiniLM-L6-v2")

        def upsert(name, docs, metas, ids):
            try: chroma.delete_collection(name)
            except Exception: pass
            col = chroma.create_collection(name)
            col.upsert(documents=docs,
                       embeddings=emb.encode(docs,show_progress_bar=False).tolist(),
                       metadatas=metas, ids=ids)

        summary_doc = (
            f"ChainFlow AI training run at {meta['timestamp']}. "
            f"Best model: {meta['best_model']}. R2={meta['best_r2']:.4f}. "
            f"MAE={meta['best_mae']:.4f}. MAPE={meta['best_mape']:.2f}%. "
            f"PRD R2 (>=0.75): {'achieved' if meta['prd_r2_achieved'] else 'not achieved'}. "
            f"PRD MAPE (<=8%): {'achieved' if meta['prd_mape_achieved'] else 'not achieved'}. "
            f"Features: {len(features)} (zero target leakage)."
        )
        upsert("chainflow_unified",
               [summary_doc],
               [{"type":"training_run","timestamp":meta["timestamp"]}],
               ["latest_run"])

        feat_doc = (
            f"Production model uses {len(features)} leak-free features: "
            + ", ".join(features) + "."
        )
        upsert("chainflow_production_model",
               [feat_doc],
               [{"type":"feature_list","n_features":str(len(features)),
                 "timestamp":meta["timestamp"]}],
               ["prod_feature_list"])

        logger.info(f"ChromaDB updated — {len(chroma.list_collections())} collections")
    except Exception as e:
        logger.warning(f"ChromaDB embed failed: {e}")

# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

def run(mode: str, data_path: Optional[str] = None):
    t0 = time.time()
    logger.info("="*60)
    logger.info(f"CHAINFLOW AI PIPELINE — mode={mode.upper()}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)

    if mode == "auto":
        needs, reason = should_retrain(data_path)
        logger.info(f"Auto-decision: retrain={needs} ({reason})")
        mode = "full" if needs else "predict"

    if mode == "full":
        raw = step_data_generation()
        res, feats = step_feature_engineering(raw)
        trained = step_train(res, feats)
        meta = step_evaluate(trained, res)
        step_embed(meta, feats)

    elif mode == "retrain":
        raw = Path(data_path) if data_path and Path(data_path).exists() \
              else RAW_DIR/"supply_chain_enriched.csv"
        if not raw.exists():
            logger.error(f"Data not found: {raw} — switching to full")
            raw = step_data_generation()
        res, feats = step_feature_engineering(raw)
        trained    = step_train(res, feats)
        meta       = step_evaluate(trained, res)
        step_embed(meta, feats)

    elif mode == "predict":
        import joblib
        lgb_path = MODEL_DIR/"lgb_model.pkl"
        if not lgb_path.exists():
            logger.error("No saved model — run --mode full first")
            sys.exit(1)
        model = joblib.load(lgb_path)
        with open(MODEL_DIR/"model_metadata.json") as f:
            meta = json.load(f)
        logger.info(f"Model loaded — R²={meta['best_r2']:.4f} MAPE={meta['best_mape']:.2f}%")

    elif mode == "embed":
        for p in [MODEL_DIR/"model_metadata.json", MODEL_DIR/"feature_names.json"]:
            if not p.exists():
                logger.error(f"Missing {p} — run training first")
                sys.exit(1)
        with open(MODEL_DIR/"model_metadata.json") as f: meta = json.load(f)
        with open(MODEL_DIR/"feature_names.json")   as f: feats= json.load(f)
        step_embed(meta, feats)

    logger.info(f"Pipeline done in {time.time()-t0:.1f}s — mode={mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChainFlow AI Training Orchestrator")
    parser.add_argument("--mode", default="auto",
                        choices=["auto","full","retrain","predict","embed"])
    parser.add_argument("--data_path", default=None)
    args = parser.parse_args()
    run(args.mode, args.data_path)
