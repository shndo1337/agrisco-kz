"""
Загрузка, валидация и очистка датасета субсидий.
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


def data_quality_report(df: pd.DataFrame) -> dict:
    """Формирует отчёт о качестве данных: пропуски, дубликаты, выбросы, типы."""
    report = {}

    # 1. Размер
    report["shape"] = df.shape

    # 2. Пропуски
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing"] = pd.DataFrame({
        "пропусков": missing,
        "процент": missing_pct,
    }).sort_values("пропусков", ascending=False)

    # 3. Дубликаты
    dup_full = df.duplicated().sum()
    dup_key = 0
    key_cols = ["oblast", "district", "direction", "subsidy_name", "amount", "normative"]
    existing_keys = [c for c in key_cols if c in df.columns]
    if existing_keys:
        dup_key = df.duplicated(subset=existing_keys).sum()
    report["duplicates_full"] = dup_full
    report["duplicates_key"] = dup_key
    report["duplicate_key_cols"] = existing_keys

    # 4. Типы данных
    report["dtypes"] = df.dtypes

    # 5. Выбросы (IQR) для числовых колонок
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((s < lower) | (s > upper)).sum()
        if n_out > 0:
            outliers[col] = {
                "count": int(n_out),
                "percent": round(n_out / len(s) * 100, 2),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "min": round(s.min(), 2),
                "max": round(s.max(), 2),
            }
    report["outliers"] = outliers

    # 6. Аномалии (бизнес-правила)
    anomalies = {}
    if "amount" in df.columns:
        neg_amount = (pd.to_numeric(df["amount"], errors="coerce") < 0).sum()
        zero_amount = (pd.to_numeric(df["amount"], errors="coerce") == 0).sum()
        anomalies["amount_negative"] = int(neg_amount)
        anomalies["amount_zero"] = int(zero_amount)
    if "normative" in df.columns:
        neg_norm = (pd.to_numeric(df["normative"], errors="coerce") < 0).sum()
        anomalies["normative_negative"] = int(neg_norm)
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        future = (dates > pd.Timestamp.now()).sum()
        anomalies["date_future"] = int(future)
        very_old = (dates < pd.Timestamp("2020-01-01")).sum()
        anomalies["date_before_2020"] = int(very_old)
    report["anomalies"] = anomalies

    return report


def print_quality_report(report: dict) -> None:
    """Выводит отчёт о качестве данных в консоль."""
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    rows, cols = report["shape"]
    print(f"\nРазмер: {rows:,} строк × {cols} столбцов")

    print("\n--- Пропуски ---")
    miss = report["missing"]
    has_miss = miss[miss["пропусков"] > 0]
    if len(has_miss) == 0:
        print("  Пропусков нет.")
    else:
        for col, row in has_miss.iterrows():
            print(f"  {col:25s} {int(row['пропусков']):>6,}  ({row['процент']:.1f}%)")

    print("\n--- Дубликаты ---")
    print(f"  Полные дубликаты: {report['duplicates_full']:,}")
    print(f"  По ключевым полям ({', '.join(report['duplicate_key_cols'])}): {report['duplicates_key']:,}")

    print("\n--- Выбросы (IQR) ---")
    if not report["outliers"]:
        print("  Выбросов не обнаружено.")
    else:
        for col, info in report["outliers"].items():
            print(f"  {col:25s} {info['count']:>6,} ({info['percent']:.1f}%)  "
                  f"[{info['lower_bound']:,.0f} .. {info['upper_bound']:,.0f}]  "
                  f"min={info['min']:,.0f}  max={info['max']:,.0f}")

    print("\n--- Аномалии ---")
    for key, val in report["anomalies"].items():
        print(f"  {key:30s} {val:,}")

    print("\n" + "=" * 60)


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Валидация + очистка. Возвращает (очищенный df, отчёт)."""
    report_before = data_quality_report(df)

    df = df.copy()
    cleaning_log = {"initial_rows": len(df), "steps": []}

    # --- numeric ---
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["normative"] = pd.to_numeric(df["normative"], errors="coerce").fillna(0)

    # --- datetime ---
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df["month"] = df["date"].dt.month.fillna(0).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear.fillna(0).astype(int)
    df["hour"] = df["date"].dt.hour.fillna(0).astype(int)

    # --- удаление полных дубликатов ---
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        cleaning_log["steps"].append(f"Удалено полных дубликатов: {n_dropped}")

    # --- аномалии: отрицательные суммы ---
    neg_mask = df["amount"] < 0
    if neg_mask.any():
        df.loc[neg_mask, "amount"] = 0
        cleaning_log["steps"].append(f"Обнулено отрицательных сумм: {neg_mask.sum()}")

    neg_norm = df["normative"] < 0
    if neg_norm.any():
        df.loc[neg_norm, "normative"] = 0
        cleaning_log["steps"].append(f"Обнулено отрицательных нормативов: {neg_norm.sum()}")

    # --- выбросы: cap amount по 99.5 перцентилю ---
    cap_amount = df["amount"].quantile(0.995)
    n_capped = (df["amount"] > cap_amount).sum()
    if n_capped > 0:
        df["amount_capped"] = df["amount"].clip(upper=cap_amount)
        cleaning_log["steps"].append(
            f"Capped amount > {cap_amount:,.0f} ₸ (99.5 перцентиль): {n_capped} строк"
        )

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

    cleaning_log["final_rows"] = len(df)

    return df, {"quality_before": report_before, "cleaning_log": cleaning_log}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка без валидационного отчёта (обратная совместимость)."""
    cleaned, _ = validate_and_clean(df)
    return cleaned


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def load_with_report(path: str = DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Загрузка + валидация + очистка с отчётом (для EDA)."""
    raw = load_raw(path)
    return validate_and_clean(raw)
