"""
AgriScore KZ — FastAPI scoring endpoint v2.
Запуск: uvicorn api.main:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import load_model, score_applicants, explain_single
from src.features import FEATURE_COLS

app = FastAPI(
    title="AgriScore KZ API",
    description="API для merit-based скоринга заявок на субсидии сельхозпроизводителей",
    version="0.2.0",
)

models, encoders, oblast_stats = None, None, None


@app.on_event("startup")
def startup():
    global models, encoders, oblast_stats
    models, encoders, oblast_stats = load_model()


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
def score_one(applicant: ApplicantIn):
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

    return ScoreOut(
        score=float(r["score"]),
        ml_score=float(r["ml_score"]),
        rule_score=float(r["rule_score"]),
        probability=float(r["probability"]),
        top_factors=expl,
    )


@app.post("/score/batch")
def score_batch(applicants: list[ApplicantIn]):
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

    return scored[["oblast", "direction", "amount", "normative",
                    "ml_score", "rule_score", "score"]].to_dict("records")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": models is not None,
            "models": list(models.keys()) if models else []}
