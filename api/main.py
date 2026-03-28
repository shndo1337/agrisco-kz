"""
AgriScore KZ — FastAPI scoring endpoint.
Запуск: uvicorn api.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import load_model, score_applicants, explain_single
from src.features import FEATURE_COLS

app = FastAPI(
    title="AgriScore KZ API",
    description="API для скоринга заявок на субсидии сельхозпроизводителей",
    version="0.1.0",
)

model, encoders = None, None


@app.on_event("startup")
def startup():
    global model, encoders
    model, encoders = load_model()


class ApplicantIn(BaseModel):
    oblast: str
    district: str
    direction: str
    subsidy_name: str
    normative: float
    amount: float
    month: int = 1
    hour: int = 12


class ScoreOut(BaseModel):
    score: float
    probability: float
    top_factors: list[dict]


@app.post("/score", response_model=ScoreOut)
def score_one(applicant: ApplicantIn):
    row = pd.DataFrame([{
        "num": 0,
        "date": None,
        "app_number": "new",
        "akimat": "",
        "status": "Новая",
        "approved": None,
        **applicant.model_dump(),
    }])

    scored, shap_vals, X = score_applicants(row, model, encoders)
    expl = explain_single(shap_vals[0], FEATURE_COLS, top_n=5)

    return ScoreOut(
        score=float(scored["score"].iloc[0]),
        probability=float(scored["probability"].iloc[0]),
        top_factors=expl,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
