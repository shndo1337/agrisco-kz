"""
AgriScore KZ — FastAPI scoring endpoint v3.
Запуск: uvicorn api.main:app --reload
"""
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import load_model, score_applicants, explain_single  # noqa: E402
from src.features import FEATURE_COLS  # noqa: E402

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agrisco")

# ─── API Key auth ───
API_KEY = os.getenv("AGRISCO_API_KEY", "dev-key-123")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key is None or api_key != API_KEY:
        logger.warning("Unauthorized request: invalid or missing API key")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


models, encoders, oblast_stats = None, None, None


def startup() -> None:
    """Загружает артефакты модели в глобальное состояние."""
    global models, encoders, oblast_stats
    logger.info("Loading model artifacts...")
    models, encoders, oblast_stats = load_model()
    logger.info(f"Models loaded: {list(models.keys())}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield


app = FastAPI(
    title="AgriScore KZ API",
    description="API для merit-based скоринга заявок на субсидии сельхозпроизводителей",
    version="0.3.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration:.3f}s)")
    return response


class ApplicantIn(BaseModel):
    oblast: str
    district: str = "Неизвестно"
    direction: str
    subsidy_name: str
    normative: float
    amount: float
    month: int = 1
    hour: int = 12


class ScoreOut(BaseModel):
    score: float
    ml_score: float
    rule_score: float
    probability: float
    top_factors: list[dict]


@app.post("/score", response_model=ScoreOut)
def score_one(applicant: ApplicantIn, _: str = Depends(verify_api_key)):
    data = applicant.model_dump()
    data["animals_count"] = data["amount"] / data["normative"] if data["normative"] > 0 else 0

    row = pd.DataFrame([{
        "num": 0, "date": None, "app_number": "api",
        "akimat": "", "status": "Новая", "approved": np.nan,
        "day_of_year": data["month"] * 30,
        **data,
    }])

    scored, shap_vals, _ = score_applicants(row, models, encoders, oblast_stats)
    expl = explain_single(shap_vals[0], FEATURE_COLS, top_n=5)
    r = scored.iloc[0]

    logger.info(f"Scored: oblast={data['oblast']}, score={r['score']:.1f}")

    return ScoreOut(
        score=float(r["score"]),
        ml_score=float(r["ml_score"]),
        rule_score=float(r["rule_score"]),
        probability=float(r["probability"]),
        top_factors=expl,
    )


@app.post("/score/batch")
def score_batch(applicants: list[ApplicantIn], _: str = Depends(verify_api_key)):
    rows = []
    for a in applicants:
        d = a.model_dump()
        d["animals_count"] = d["amount"] / d["normative"] if d["normative"] > 0 else 0
        d.update({"num": 0, "date": None, "app_number": "api",
                  "akimat": "", "status": "Новая", "approved": np.nan,
                  "day_of_year": d["month"] * 30})
        rows.append(d)

    df = pd.DataFrame(rows)
    scored, _, _ = score_applicants(df, models, encoders, oblast_stats)

    logger.info(f"Batch scored: {len(applicants)} applicants")

    return scored[["oblast", "direction", "amount", "normative",
                    "ml_score", "rule_score", "score"]].to_dict("records")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": models is not None,
            "models": list(models.keys()) if models else []}
