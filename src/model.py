"""
Ensemble scoring model (XGBoost + LightGBM) + SHAP + composite scoring.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

from src.preprocessing import load_and_clean
from src.features import (
    engineer,
    get_feature_matrix,
    FEATURE_COLS,
    FEATURE_NAMES_RU,
)
from src.rules import rule_score_batch, compute_oblast_stats, composite_score

ARTIFACTS = Path("data/processed")
MODEL_PATH = ARTIFACTS / "model.joblib"
ENCODERS_PATH = ARTIFACTS / "encoders.joblib"
STATS_PATH = ARTIFACTS / "oblast_stats.joblib"

# Try to import LightGBM (optional, falls back to XGBoost-only)
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def train(df: pd.DataFrame | None = None, save: bool = True):
    if df is None:
        df = load_and_clean()

    # Compute oblast stats for rule-based scoring
    oblast_stats = compute_oblast_stats(df)

    labeled = df[df["approved"].notna()].copy()
    print(f"Labeled samples: {len(labeled)}  "
          f"(approved={int(labeled['approved'].sum())}, "
          f"rejected={int((labeled['approved'] == 0).sum())})")

    df_feat, encoders = engineer(labeled, fit=True)
    X = get_feature_matrix(df_feat)
    y = df_feat["approved"].astype(int)

    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = neg / pos

    # ─── XGBoost ───
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring="roc_auc")
    print(f"XGBoost CV ROC-AUC: {xgb_cv_scores.mean():.4f} +/- {xgb_cv_scores.std():.4f}")

    xgb_model.fit(X, y, verbose=False)

    models = {"xgb": xgb_model}
    weights = {"xgb": 1.0}

    # ─── LightGBM (if available) ───
    if HAS_LGB:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_cv_scores = cross_val_score(lgb_model, X, y, cv=cv, scoring="roc_auc")
        print(f"LightGBM CV ROC-AUC: {lgb_cv_scores.mean():.4f} +/- {lgb_cv_scores.std():.4f}")

        lgb_model.fit(X, y)
        models["lgb"] = lgb_model
        weights = {"xgb": 0.55, "lgb": 0.45}

    # ─── Ensemble prediction on full data ───
    proba = _ensemble_predict(models, weights, X)
    auc = roc_auc_score(y, proba)
    print(f"\nEnsemble ROC-AUC (full): {auc:.4f}")
    print(classification_report(
        y, (proba >= 0.5).astype(int),
        target_names=["Отклонена", "Одобрена"],
    ))

    if save:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, MODEL_PATH)
        joblib.dump(encoders, ENCODERS_PATH)
        joblib.dump(oblast_stats, STATS_PATH)
        print(f"Artifacts saved to {ARTIFACTS}/")

    return models, encoders, oblast_stats, auc


def _ensemble_predict(models: dict, weights: dict, X: pd.DataFrame) -> np.ndarray:
    """Weighted ensemble prediction across model dict."""
    total_w = sum(weights.values())
    proba = np.zeros(len(X))
    for name, model in models.items():
        proba += model.predict_proba(X)[:, 1] * weights.get(name, 0.0)
    return proba / total_w


def load_model():
    models = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    oblast_stats = joblib.load(STATS_PATH) if STATS_PATH.exists() else None
    return models, encoders, oblast_stats


def score_applicants(
    df: pd.DataFrame,
    models: dict,
    encoders: dict,
    oblast_stats: dict | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    df_feat, _ = engineer(df, encoders=encoders, fit=False)
    X = get_feature_matrix(df_feat)

    weights = {"xgb": 0.55, "lgb": 0.45} if "lgb" in models else {"xgb": 1.0}
    proba = _ensemble_predict(models, weights, X)
    ml_score = pd.Series((proba * 100).round(1), index=df.index)

    # Rule-based scoring
    rule_scores_df = rule_score_batch(df, oblast_stats)
    rule_s = rule_scores_df["rule_score"]

    # Composite
    final_score = composite_score(ml_score, rule_s, ml_weight=0.6, rule_weight=0.4)

    # SHAP (use XGBoost for explainability)
    explainer = shap.TreeExplainer(models["xgb"])
    shap_values = explainer.shap_values(X)

    result = df.copy()
    result["ml_score"] = ml_score
    result["rule_score"] = rule_s
    result["score"] = final_score
    result["probability"] = proba

    # Add rule score components
    for col in rule_scores_df.columns:
        if col != "rule_score":
            result[col] = rule_scores_df[col]

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


def generate_text_explanation(
    explanations: list[dict],
    score: float,
    ml_score: float | None = None,
    rule_score: float | None = None,
) -> str:
    lines = [f"Итоговый балл: {score:.1f}/100"]
    if ml_score is not None and rule_score is not None:
        lines.append(f"  ML-компонент: {ml_score:.1f}  |  Rule-компонент: {rule_score:.1f}")
        lines.append(f"  (60% ML + 40% Rules)")
    lines.append("")

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
