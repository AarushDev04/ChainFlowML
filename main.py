# ==============================================================================
# CHAINFLOW AI — app/main.py
# Production FastAPI entry point.
# Completely independent of Colab — reads artifacts from ARTIFACTS_ROOT.
# ==============================================================================
# Start:
#   uvicorn app.main:app --host 0.0.0.0 --port 8000
#   (or via docker-compose)
#
# Endpoints:
#   GET  /health
#   GET  /model/info
#   GET  /collections
#   POST /predict/single        JSON body → single SKU forecast
#   POST /predict/batch         JSON list → bulk forecast + LLM memory
#   POST /predict/upload        CSV file upload → forecast + LLM memory
#   POST /llm/query             Single-turn RAG query
#   POST /llm/chat              Multi-turn conversation
#   POST /retrain/trigger       Background retraining
#   GET  /retrain/status
#   GET  /metrics               Prometheus plaintext
#   GET  /docs                  Swagger UI (auto-generated)
# ==============================================================================

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import state
from app.routers import predict, llm, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    await state.load_all()
    yield
    state.release()


app = FastAPI(
    title="ChainFlow AI — Demand Forecasting + LLM Analytics",
    version="2.0.0",
    description=(
        "Supply chain demand forecasting with LightGBM + CNN-LSTM inference "
        "and Llama3-ChatQA-1.5-7B RAG analytics. "
        "Upload new data → get predictions → chat with LLM about results."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(predict.router)
app.include_router(llm.router)
