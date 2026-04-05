"""Тесты для preprocessing: загрузка, валидация, очистка."""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    data_quality_report,
    validate_and_clean,
    clean,
    APPROVED_STATUSES,
    REJECTED_STATUSES,
)


@pytest.fixture
def sample_df():
    """Минимальный DataFrame, имитирующий сырые данные."""
    return pd.DataFrame({
        "num": [1, 2, 3, 4, 5],
        "date": ["01.03.2025 10:00:00"] * 5,
        "oblast": ["Алматинская"] * 3 + ["Костанайская"] * 2,
        "akimat": ["Акимат"] * 5,
        "app_number": ["A1", "A2", "A3", "A4", "A5"],
        "direction": ["Скотоводство"] * 5,
        "subsidy_name": ["Приобретение племенного поголовья"] * 5,
        "status": ["Исполнена", "Одобрена", "Отклонена", "Исполнена", "Отозвано"],
        "normative": [50000, 30000, 10000, 0, -100],
        "amount": [500000, 300000, 100000, 0, -5000],
        "district": ["Район1"] * 5,
    })


def test_data_quality_report_structure(sample_df):
    report = data_quality_report(sample_df)
    assert "shape" in report
    assert "missing" in report
    assert "duplicates_full" in report
    assert "duplicates_key" in report
    assert "outliers" in report
    assert "anomalies" in report
    assert report["shape"] == (5, 11)


def test_data_quality_missing(sample_df):
    sample_df.loc[0, "oblast"] = None
    report = data_quality_report(sample_df)
    assert report["missing"].loc["oblast", "пропусков"] == 1


def test_data_quality_duplicates(sample_df):
    df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    report = data_quality_report(df)
    assert report["duplicates_full"] >= 1


def test_data_quality_anomalies(sample_df):
    report = data_quality_report(sample_df)
    assert report["anomalies"]["amount_negative"] == 1
    assert report["anomalies"]["normative_negative"] == 1


def test_validate_and_clean_removes_negative(sample_df):
    cleaned, info = validate_and_clean(sample_df)
    assert (cleaned["amount"] >= 0).all()
    assert (cleaned["normative"] >= 0).all()


def test_validate_and_clean_creates_columns(sample_df):
    cleaned, _ = validate_and_clean(sample_df)
    assert "month" in cleaned.columns
    assert "hour" in cleaned.columns
    assert "day_of_year" in cleaned.columns
    assert "animals_count" in cleaned.columns
    assert "approved" in cleaned.columns


def test_clean_approved_labels(sample_df):
    cleaned = clean(sample_df)
    assert cleaned.loc[0, "approved"] == 1  # Исполнена
    assert cleaned.loc[2, "approved"] == 0  # Отклонена
    assert pd.isna(cleaned.loc[4, "approved"])  # Отозвано


def test_animals_count_derived(sample_df):
    cleaned = clean(sample_df)
    row = cleaned.iloc[0]
    expected = 500000 / 50000
    assert abs(row["animals_count"] - expected) < 0.01


def test_animals_count_zero_normative(sample_df):
    cleaned = clean(sample_df)
    row = cleaned[cleaned["normative"] == 0].iloc[0]
    assert row["animals_count"] == 0
