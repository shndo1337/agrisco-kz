"""Тесты для rule-based scoring engine."""
import pytest
import pandas as pd
import numpy as np
from src.rules import rule_score_batch, compute_oblast_stats, composite_score


@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        "num": range(1, 6),
        "date": ["01.03.2025 10:00:00"] * 5,
        "oblast": ["Алматинская", "Костанайская", "Акмолинская", "Алматинская", "Алматинская"],
        "akimat": ["Акимат"] * 5,
        "app_number": [f"A{i}" for i in range(5)],
        "direction": ["Скотоводство"] * 3 + ["Птицеводство", "Коневодство"],
        "subsidy_name": [
            "Приобретение импортного племенного поголовья",
            "Селекционная и племенная работа",
            "Удешевление корма",
            "Удешевление мяса птицы",
            "Приобретение отечественного поголовья",
        ],
        "status": ["Исполнена"] * 3 + ["Отклонена"] * 2,
        "normative": [150000, 50000, 5000, 30000, 80000],
        "amount": [1500000, 500000, 50000, 300000, 800000],
        "district": ["Район1"] * 5,
        "month": [3, 5, 7, 9, 4],
        "animals_count": [10, 10, 10, 10, 10],
        "approved": [1, 1, 1, 0, 0],
    })
    return df


@pytest.fixture
def oblast_stats(sample_df):
    return compute_oblast_stats(sample_df)


def test_rule_score_batch_returns_dataframe(sample_df, oblast_stats):
    result = rule_score_batch(sample_df, oblast_stats)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_df)
    assert "rule_score" in result.columns


def test_rule_score_range(sample_df, oblast_stats):
    result = rule_score_batch(sample_df, oblast_stats)
    assert (result["rule_score"] >= 0).all()
    assert (result["rule_score"] <= 100).all()


def test_rule_score_components_exist(sample_df, oblast_stats):
    result = rule_score_batch(sample_df, oblast_stats)
    expected_cols = [
        "subsidy_type_score", "normative_score", "herd_size_score",
        "direction_score", "regional_score", "seasonal_score",
        "mortality_score", "pasture_score",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing component: {col}"


def test_import_scores_higher_than_feed(sample_df, oblast_stats):
    result = rule_score_batch(sample_df, oblast_stats)
    import_score = result.iloc[0]["subsidy_type_score"]  # импортное племенное
    feed_score = result.iloc[2]["subsidy_type_score"]  # корма
    assert import_score > feed_score


def test_composite_score_weighted():
    ml = pd.Series([80.0, 50.0, 20.0])
    rule = pd.Series([60.0, 70.0, 90.0])
    comp = composite_score(ml, rule, ml_weight=0.6, rule_weight=0.4)
    expected = 0.6 * ml + 0.4 * rule
    assert np.allclose(comp, expected)


def test_composite_score_range():
    ml = pd.Series([0.0, 50.0, 100.0])
    rule = pd.Series([0.0, 50.0, 100.0])
    comp = composite_score(ml, rule)
    assert (comp >= 0).all()
    assert (comp <= 100).all()


def test_compute_oblast_stats(sample_df):
    stats = compute_oblast_stats(sample_df)
    assert isinstance(stats, dict)
    assert "Алматинская" in stats
    assert "avg_amount" in stats["Алматинская"]
    assert "approval_rate" in stats["Алматинская"]
