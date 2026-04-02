"""
Rule-based scoring component.
Основан на "Правилах субсидирования развития племенного животноводства"
(Приказ МСХ РК от 15.03.2019 №108).

Каждая заявка получает rule-score от 0 до 100,
который потом комбинируется с ML-score.
"""
import numpy as np
import pandas as pd


# ─── Приоритеты направлений ───
# Племенное животноводство — основная цель субсидирования (п. 1-1 Правил).
# Приоритет: племенная работа > приобретение поголовья > удешевление продукции > корма
DIRECTION_PRIORITY = {
    "Субсидирование в скотоводстве": 1.0,
    "Субсидирование в овцеводстве": 0.95,
    "Субсидирование в коневодстве": 0.90,
    "Субсидирование в верблюдоводстве": 0.90,
    "Субсидирование в козоводстве": 0.85,
    "Субсидирование в птицеводстве": 0.80,
    "Субсидирование в свиноводстве": 0.75,
    "Субсидирование в пчеловодстве": 0.70,
    "Субсидирование затрат по искусственному осеменению": 0.85,
}

# ─── Классификация типа субсидии по приоритету ───
# Из Правил: племенная работа (селекция) > приобретение племенного поголовья >
# удешевление стоимости продукции > удешевление кормов
SUBSIDY_TYPE_KEYWORDS = {
    "селекционной и племенной работы": 30,   # Высший приоритет — развитие генетики
    "приобретен": 25,                         # Приобретение поголовья
    "выращивани": 22,                         # Выращивание племенного молодняка
    "искусственн": 20,                        # Искусственное осеменение
    "импорт": 28,                             # Импорт — повышенный норматив по Правилам
    "удешевлени": 15,                         # Удешевление стоимости продукции
    "корм": 12,                               # Удешевление кормов
    "содержан": 10,                           # Содержание поголовья
}


def _subsidy_type_score(subsidy_name: str) -> float:
    """Балл за тип субсидии (0–30)."""
    name = subsidy_name.lower()
    best = 10  # default
    for keyword, score in SUBSIDY_TYPE_KEYWORDS.items():
        if keyword in name:
            best = max(best, score)
    return best


def _normative_score(normative: float) -> float:
    """
    Балл за норматив (0–25).
    Высокий норматив → дорогое племенное поголовье → выше приоритет.
    Из Правил: нормативы от 20 тг/кг (мёд) до 400 000 тг/гол (импорт КРС).
    """
    if normative <= 0:
        return 0
    log_norm = np.log1p(normative)
    # Scale: log(20)≈3 → 0pts, log(400000)≈13 → 25pts
    score = np.clip((log_norm - 3) / 10 * 25, 0, 25)
    return round(score, 2)


def _herd_size_score(animals_count: float) -> float:
    """
    Балл за размер стада (0–15).
    Крупные хозяйства — больше вклад в отрасль.
    Но не линейно: от 1 до 1000+ голов.
    """
    if animals_count <= 0:
        return 0
    log_count = np.log1p(animals_count)
    # log(1)=0.7 → ~0, log(1000)=6.9 → 15
    score = np.clip((log_count - 0.7) / 6.2 * 15, 0, 15)
    return round(score, 2)


def _direction_score(direction: str) -> float:
    """Балл за направление животноводства (0–10)."""
    priority = DIRECTION_PRIORITY.get(direction, 0.5)
    return round(priority * 10, 2)


def _regional_demand_score(
    oblast: str,
    oblast_stats: dict[str, dict] | None = None,
) -> float:
    """
    Балл за региональный спрос (0–10).
    Регионы с меньшей долей заявок (недообеспечены) получают бонус.
    """
    if oblast_stats is None:
        return 5.0  # нейтральный, если нет статистики
    stats = oblast_stats.get(oblast, {})
    demand_ratio = stats.get("demand_ratio", 0.5)
    # demand_ratio < 0.5 → недообеспечен → бонус
    score = np.clip((1 - demand_ratio) * 10, 0, 10)
    return round(score, 2)


def _seasonal_score(month: int, direction: str) -> float:
    """
    Балл за сезонность подачи (0–10).
    Из Правил: для некоторых видов есть оптимальные сроки подачи.
    Случной сезон, заготовка кормов и т.д.
    """
    if month == 0:
        return 5.0

    # Скотоводство: оптимально начало года (январь-март) до случного сезона
    # Птицеводство: круглогодично
    # Корма: осень (после заготовки)
    cattle_optimal = {1: 10, 2: 10, 3: 9, 4: 8, 5: 7, 6: 6, 7: 5, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9}
    poultry_optimal = {m: 7 for m in range(1, 13)}
    default_optimal = {m: 6 for m in range(1, 13)}

    if "скотовод" in direction.lower() or "овцевод" in direction.lower():
        return cattle_optimal.get(month, 5)
    elif "птицевод" in direction.lower():
        return poultry_optimal.get(month, 5)
    return default_optimal.get(month, 5)


def compute_oblast_stats(df: pd.DataFrame) -> dict[str, dict]:
    """Рассчитать статистику по областям для regional scoring."""
    total = len(df)
    stats = {}
    for oblast, group in df.groupby("oblast"):
        count = len(group)
        approved = (group["status"].isin({"Исполнена", "Одобрена"})).sum()
        avg_amount = group["amount"].mean()
        stats[oblast] = {
            "count": count,
            "share": count / total,
            "demand_ratio": count / total * 18,  # нормализация на 18 областей
            "approval_rate": approved / count if count > 0 else 0,
            "avg_amount": avg_amount,
        }
    return stats


def rule_score_single(
    row: pd.Series,
    oblast_stats: dict[str, dict] | None = None,
) -> dict:
    """
    Вычислить rule-based score для одной заявки.
    Возвращает dict с компонентами и итоговым баллом.
    """
    subsidy_s = _subsidy_type_score(str(row.get("subsidy_name", "")))
    normative_s = _normative_score(float(row.get("normative", 0)))
    herd_s = _herd_size_score(float(row.get("animals_count", 0)))
    direction_s = _direction_score(str(row.get("direction", "")))
    regional_s = _regional_demand_score(str(row.get("oblast", "")), oblast_stats)
    seasonal_s = _seasonal_score(int(row.get("month", 0)), str(row.get("direction", "")))

    total = subsidy_s + normative_s + herd_s + direction_s + regional_s + seasonal_s
    # Нормализация к 0–100
    max_possible = 30 + 25 + 15 + 10 + 10 + 10  # = 100
    score_100 = round(total / max_possible * 100, 1)

    return {
        "rule_score": score_100,
        "subsidy_type_score": subsidy_s,
        "normative_score": normative_s,
        "herd_size_score": herd_s,
        "direction_score": direction_s,
        "regional_score": regional_s,
        "seasonal_score": seasonal_s,
    }


def rule_score_batch(
    df: pd.DataFrame,
    oblast_stats: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Вычислить rule-based scores для всего датафрейма."""
    results = df.apply(lambda row: rule_score_single(row, oblast_stats), axis=1)
    scores_df = pd.DataFrame(results.tolist(), index=df.index)
    return scores_df


def composite_score(
    ml_score: pd.Series,
    rule_score: pd.Series,
    ml_weight: float = 0.6,
    rule_weight: float = 0.4,
) -> pd.Series:
    """
    Финальный композитный балл = взвешенная комбинация ML и rule-based.
    ML-компонент ловит паттерны из данных.
    Rule-компонент обеспечивает соответствие нормативным требованиям.
    """
    return (ml_score * ml_weight + rule_score * rule_weight).round(1)
