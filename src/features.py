"""
Feature engineering для скоринговой модели.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


CATEGORICAL_COLS = ["oblast", "direction", "district"]

FEATURE_COLS = [
    "log_amount",
    "log_normative",
    "log_animals_count",
    "normative_tier",
    "is_import",
    "is_breeding",
    "is_milk",
    "is_meat",
    "is_purchase",
    "is_feed",
    "is_poultry",
    "oblast_enc",
    "direction_enc",
    "district_enc",
    "subsidy_enc",
    "month",
    "hour",
]

FEATURE_NAMES_RU = {
    "log_amount": "Сумма субсидии",
    "log_normative": "Норматив на голову",
    "log_animals_count": "Количество животных",
    "normative_tier": "Класс норматива",
    "is_import": "Импортное поголовье",
    "is_breeding": "Племенная работа",
    "is_milk": "Молочное направление",
    "is_meat": "Мясное направление",
    "is_purchase": "Приобретение поголовья",
    "is_feed": "Субсидия на корма",
    "is_poultry": "Птицеводство",
    "oblast_enc": "Область",
    "direction_enc": "Направление животноводства",
    "district_enc": "Район хозяйства",
    "subsidy_enc": "Тип субсидии",
    "month": "Месяц подачи",
    "hour": "Час подачи",
}


def _safe_label_encode(series: pd.Series, le: LabelEncoder | None, fit: bool):
    filled = series.fillna("__UNKNOWN__").astype(str)
    if fit:
        le = LabelEncoder()
        encoded = le.fit_transform(filled)
    else:
        mapping = {c: i for i, c in enumerate(le.classes_)}
        encoded = filled.map(mapping).fillna(-1).astype(int)
    return encoded, le


def engineer(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    if encoders is None:
        encoders = {}

    # --- log transforms ---
    for col in ["amount", "normative", "animals_count"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # --- text flags from subsidy_name ---
    name_lower = df["subsidy_name"].str.lower()
    df["is_import"] = name_lower.str.contains("импорт", na=False).astype(int)
    df["is_breeding"] = name_lower.str.contains("племен", na=False).astype(int)
    df["is_milk"] = name_lower.str.contains("молок", na=False).astype(int)
    df["is_meat"] = name_lower.str.contains("мяс", na=False).astype(int)
    df["is_purchase"] = name_lower.str.contains("приобретен", na=False).astype(int)
    df["is_feed"] = name_lower.str.contains("корм", na=False).astype(int)
    df["is_poultry"] = name_lower.str.contains("птиц", na=False).astype(int)

    # --- normative tier ---
    bins = [0, 1_000, 20_000, 100_000, 200_000, float("inf")]
    labels = [0, 1, 2, 3, 4]
    df["normative_tier"] = pd.cut(
        df["normative"].fillna(0), bins=bins, labels=labels, include_lowest=True
    ).astype(int)

    # --- categorical encoding ---
    for col in CATEGORICAL_COLS:
        df[f"{col}_enc"], encoders[col] = _safe_label_encode(
            df[col], encoders.get(col), fit=fit
        )

    df["subsidy_enc"], encoders["subsidy_name"] = _safe_label_encode(
        df["subsidy_name"], encoders.get("subsidy_name"), fit=fit
    )

    return df, encoders


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLS].fillna(0).astype(float)
