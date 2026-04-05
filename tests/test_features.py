"""Тесты для feature engineering."""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import clean
from src.features import engineer, get_feature_matrix, FEATURE_COLS


@pytest.fixture
def labeled_df():
    """Минимальный размеченный DataFrame."""
    df = pd.DataFrame({
        "num": range(1, 21),
        "date": ["01.03.2025 10:00:00"] * 20,
        "oblast": ["Алматинская"] * 10 + ["Костанайская"] * 10,
        "akimat": ["Акимат"] * 20,
        "app_number": [f"A{i}" for i in range(20)],
        "direction": ["Скотоводство"] * 10 + ["Птицеводство"] * 10,
        "subsidy_name": ["Приобретение импортного племенного поголовья"] * 10 +
                        ["Селекционная работа птицеводства"] * 10,
        "status": ["Исполнена"] * 15 + ["Отклонена"] * 5,
        "normative": [50000] * 20,
        "amount": [500000] * 20,
        "district": ["Район1"] * 10 + ["Район2"] * 10,
    })
    return clean(df)


def test_engineer_output_shape(labeled_df):
    df_feat, encoders = engineer(labeled_df, fit=True)
    assert len(df_feat) == len(labeled_df)
    for col in FEATURE_COLS:
        assert col in df_feat.columns, f"Missing feature: {col}"


def test_feature_matrix_no_nans(labeled_df):
    df_feat, _ = engineer(labeled_df, fit=True)
    X = get_feature_matrix(df_feat)
    assert X.isna().sum().sum() == 0


def test_feature_matrix_shape(labeled_df):
    df_feat, _ = engineer(labeled_df, fit=True)
    X = get_feature_matrix(df_feat)
    assert X.shape == (len(labeled_df), len(FEATURE_COLS))


def test_text_flags(labeled_df):
    df_feat, _ = engineer(labeled_df, fit=True)
    # First 10 rows have "импортного" in subsidy_name
    assert df_feat.iloc[0]["is_import"] == 1
    # Last 10 have "Селекционная"
    assert df_feat.iloc[15]["is_selection"] == 1


def test_engineer_inference_mode(labeled_df):
    df_feat, encoders = engineer(labeled_df, fit=True)
    # Re-run in inference mode
    df_feat2, _ = engineer(labeled_df, encoders=encoders, fit=False)
    assert len(df_feat2) == len(labeled_df)
    for col in FEATURE_COLS:
        assert col in df_feat2.columns


def test_log_transforms_positive(labeled_df):
    df_feat, _ = engineer(labeled_df, fit=True)
    assert (df_feat["log_amount"] >= 0).all()
    assert (df_feat["log_normative"] >= 0).all()
