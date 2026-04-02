"""
Feature engineering для скоринговой модели.
v2: добавлены региональные агрегаты, ratios, interaction features.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


CATEGORICAL_COLS = ["oblast", "direction", "district"]

FEATURE_COLS = [
    # Base numeric (log-scaled)
    "log_amount",
    "log_normative",
    "log_animals_count",
    # Tiers
    "normative_tier",
    # Text-derived flags
    "is_import",
    "is_breeding",
    "is_selection",
    "is_milk",
    "is_meat",
    "is_purchase",
    "is_feed",
    "is_poultry",
    "is_young",
    # Categorical encoded
    "oblast_enc",
    "direction_enc",
    "district_enc",
    "subsidy_enc",
    # Temporal
    "month",
    "hour",
    "is_q1",
    # Regional aggregates
    "oblast_avg_amount",
    "oblast_approval_rate",
    "oblast_app_count",
    "district_avg_amount",
    # Ratios
    "amount_to_oblast_avg",
    "normative_to_direction_avg",
    "animals_to_oblast_avg",
]

FEATURE_NAMES_RU = {
    "log_amount": "Сумма субсидии",
    "log_normative": "Норматив на голову",
    "log_animals_count": "Количество животных",
    "normative_tier": "Класс норматива",
    "is_import": "Импортное поголовье",
    "is_breeding": "Племенная работа",
    "is_selection": "Селекционная работа",
    "is_milk": "Молочное направление",
    "is_meat": "Мясное направление",
    "is_purchase": "Приобретение поголовья",
    "is_feed": "Субсидия на корма",
    "is_poultry": "Птицеводство",
    "is_young": "Молодняк",
    "oblast_enc": "Область",
    "direction_enc": "Направление животноводства",
    "district_enc": "Район хозяйства",
    "subsidy_enc": "Тип субсидии",
    "month": "Месяц подачи",
    "hour": "Час подачи",
    "is_q1": "Подача в Q1 (янв-март)",
    "oblast_avg_amount": "Средняя сумма по области",
    "oblast_approval_rate": "Доля одобренных по области",
    "oblast_app_count": "Кол-во заявок в области",
    "district_avg_amount": "Средняя сумма по району",
    "amount_to_oblast_avg": "Сумма / средняя по области",
    "normative_to_direction_avg": "Норматив / средний по направлению",
    "animals_to_oblast_avg": "Кол-во голов / среднее по области",
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


def _compute_aggregates(df: pd.DataFrame, fit: bool, agg_stats: dict | None):
    """Compute regional and directional aggregate features."""
    if fit:
        agg_stats = {}
        # Oblast-level
        oblast_agg = df.groupby("oblast").agg(
            oblast_avg_amount=("amount", "mean"),
            oblast_app_count=("amount", "count"),
            oblast_avg_animals=("animals_count", "mean"),
        )
        if "approved" in df.columns:
            approval = df[df["approved"].notna()].groupby("oblast")["approved"].mean()
            oblast_agg["oblast_approval_rate"] = approval
        else:
            oblast_agg["oblast_approval_rate"] = 0.5
        oblast_agg = oblast_agg.fillna(0.5)
        agg_stats["oblast"] = oblast_agg.to_dict("index")

        # District-level
        district_agg = df.groupby("district")["amount"].mean()
        agg_stats["district_avg_amount"] = district_agg.to_dict()

        # Direction-level
        direction_agg = df.groupby("direction")["normative"].mean()
        agg_stats["direction_avg_normative"] = direction_agg.to_dict()

    # Map aggregates to rows
    oblast_data = agg_stats["oblast"]
    df["oblast_avg_amount"] = df["oblast"].map(
        {k: v["oblast_avg_amount"] for k, v in oblast_data.items()}
    ).fillna(0)
    df["oblast_approval_rate"] = df["oblast"].map(
        {k: v["oblast_approval_rate"] for k, v in oblast_data.items()}
    ).fillna(0.5)
    df["oblast_app_count"] = df["oblast"].map(
        {k: v["oblast_app_count"] for k, v in oblast_data.items()}
    ).fillna(0)
    df["oblast_avg_animals"] = df["oblast"].map(
        {k: v.get("oblast_avg_animals", 0) for k, v in oblast_data.items()}
    ).fillna(0)

    df["district_avg_amount"] = df["district"].map(
        agg_stats["district_avg_amount"]
    ).fillna(0)

    direction_avg_norm = agg_stats["direction_avg_normative"]
    df["direction_avg_normative"] = df["direction"].map(direction_avg_norm).fillna(1)

    # Ratios (relative to region/direction averages)
    df["amount_to_oblast_avg"] = np.where(
        df["oblast_avg_amount"] > 0,
        df["amount"] / df["oblast_avg_amount"],
        1.0,
    )
    df["normative_to_direction_avg"] = np.where(
        df["direction_avg_normative"] > 0,
        df["normative"] / df["direction_avg_normative"],
        1.0,
    )
    df["animals_to_oblast_avg"] = np.where(
        df["oblast_avg_animals"] > 0,
        df["animals_count"] / df["oblast_avg_animals"],
        1.0,
    )

    # Log-scale the ratios to avoid extreme values
    for col in ["amount_to_oblast_avg", "normative_to_direction_avg", "animals_to_oblast_avg"]:
        df[col] = np.log1p(df[col].clip(0, 100))

    # Log-scale counts
    df["oblast_app_count"] = np.log1p(df["oblast_app_count"])
    df["oblast_avg_amount"] = np.log1p(df["oblast_avg_amount"])
    df["district_avg_amount"] = np.log1p(df["district_avg_amount"])

    return df, agg_stats


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
    df["is_selection"] = name_lower.str.contains("селекцион", na=False).astype(int)
    df["is_milk"] = name_lower.str.contains("молок", na=False).astype(int)
    df["is_meat"] = name_lower.str.contains("мяс", na=False).astype(int)
    df["is_purchase"] = name_lower.str.contains("приобретен", na=False).astype(int)
    df["is_feed"] = name_lower.str.contains("корм", na=False).astype(int)
    df["is_poultry"] = name_lower.str.contains("птиц", na=False).astype(int)
    df["is_young"] = name_lower.str.contains("молодняк", na=False).astype(int)

    # --- normative tier ---
    bins = [0, 1_000, 20_000, 100_000, 200_000, float("inf")]
    labels = [0, 1, 2, 3, 4]
    df["normative_tier"] = pd.cut(
        df["normative"].fillna(0), bins=bins, labels=labels, include_lowest=True
    ).astype(int)

    # --- temporal ---
    df["is_q1"] = (df["month"].between(1, 3)).astype(int)

    # --- categorical encoding ---
    for col in CATEGORICAL_COLS:
        df[f"{col}_enc"], encoders[col] = _safe_label_encode(
            df[col], encoders.get(col), fit=fit
        )

    df["subsidy_enc"], encoders["subsidy_name"] = _safe_label_encode(
        df["subsidy_name"], encoders.get("subsidy_name"), fit=fit
    )

    # --- regional aggregates & ratios ---
    df, agg_stats = _compute_aggregates(df, fit=fit, agg_stats=encoders.get("_agg_stats"))
    if fit:
        encoders["_agg_stats"] = agg_stats

    return df, encoders


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLS].fillna(0).astype(float)
