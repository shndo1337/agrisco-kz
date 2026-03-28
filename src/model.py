"""
XGBoost scoring model + SHAP explainer.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from src.preprocessing import load_and_clean
from src.features import (
    engineer,
    get_feature_matrix,
    FEATURE_COLS,
    FEATURE_NAMES_RU,
)

ARTIFACTS = Path("data/processed")
MODEL_PATH = ARTIFACTS / "model.joblib"
ENCODERS_PATH = ARTIFACTS / "encoders.joblib"


def train(df: pd.DataFrame | None = None, save: bool = True):
    if df is None:
        df = load_and_clean()

    labeled = df[df["approved"].notna()].copy()
    print(f"Labeled samples: {len(labeled)}  "
          f"(approved={int(labeled['approved'].sum())}, "
          f"rejected={int((labeled['approved'] == 0).sum())})")

    df_feat, encoders = engineer(labeled, fit=True)
    X = get_feature_matrix(df_feat)
    y = df_feat["approved"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC on test: {auc:.4f}")
    print(classification_report(y_test, (y_proba >= 0.5).astype(int),
                                target_names=["Отклонена", "Одобрена"]))

    if save:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoders, ENCODERS_PATH)
        print(f"Artifacts saved to {ARTIFACTS}/")

    return model, encoders, auc


def load_model():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, encoders


def score_applicants(
    df: pd.DataFrame,
    model: xgb.XGBClassifier,
    encoders: dict,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    df_feat, _ = engineer(df, encoders=encoders, fit=False)
    X = get_feature_matrix(df_feat)

    proba = model.predict_proba(X)[:, 1]
    score = (proba * 100).round(1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    result = df.copy()
    result["score"] = score
    result["probability"] = proba

    return result, shap_values, X


def explain_single(
    shap_vals_row: np.ndarray,
    feature_names: list[str] = FEATURE_COLS,
    top_n: int = 5,
) -> list[dict]:
    pairs = sorted(
        zip(feature_names, shap_vals_row),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    explanations = []
    for feat, val in pairs:
        name_ru = FEATURE_NAMES_RU.get(feat, feat)
        direction = "повышает" if val > 0 else "понижает"
        explanations.append({
            "feature": feat,
            "name_ru": name_ru,
            "shap_value": round(float(val), 4),
            "direction": direction,
            "text": f"{name_ru} {direction} балл ({val:+.3f})",
        })
    return explanations


def generate_text_explanation(explanations: list[dict], score: float) -> str:
    lines = [f"Итоговый балл: {score:.1f}/100\n"]
    positives = [e for e in explanations if e["shap_value"] > 0]
    negatives = [e for e in explanations if e["shap_value"] < 0]

    if positives:
        lines.append("Положительные факторы:")
        for e in positives:
            lines.append(f"  + {e['name_ru']} ({e['shap_value']:+.3f})")

    if negatives:
        lines.append("Факторы риска:")
        for e in negatives:
            lines.append(f"  - {e['name_ru']} ({e['shap_value']:+.3f})")

    return "\n".join(lines)
