"""
Загрузка и очистка датасета субсидий.
"""
import pandas as pd
import numpy as np


DATA_PATH = "data/raw/Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx"

APPROVED_STATUSES = {"Исполнена", "Одобрена"}
REJECTED_STATUSES = {"Отклонена"}
EXCLUDE_STATUSES = {"Отозвано", "Сформировано поручение", "Получена"}


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, header=4)
    df.columns = [
        "num", "date", "_col3", "_col4",
        "oblast", "akimat", "app_number",
        "direction", "subsidy_name", "status",
        "normative", "amount", "district",
    ]
    df = df.drop(columns=["_col3", "_col4"])
    df = df.dropna(subset=["num"])
    df = df[df["num"] != "№ п/п"].copy()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- numeric ---
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["normative"] = pd.to_numeric(df["normative"], errors="coerce").fillna(0)

    # --- datetime ---
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df["month"] = df["date"].dt.month.fillna(0).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear.fillna(0).astype(int)
    df["hour"] = df["date"].dt.hour.fillna(0).astype(int)

    # --- derived ---
    df["animals_count"] = np.where(
        df["normative"] > 0,
        df["amount"] / df["normative"],
        0,
    )

    # --- strip strings ---
    for col in ["oblast", "direction", "subsidy_name", "status", "district"]:
        df[col] = df[col].astype(str).str.strip()

    # --- target ---
    def _label(s):
        if s in APPROVED_STATUSES:
            return 1
        if s in REJECTED_STATUSES:
            return 0
        return np.nan

    df["approved"] = df["status"].apply(_label)

    return df


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))
