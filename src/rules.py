"""
Rule-based scoring component.
Основан на трёх НПА Министерства сельского хозяйства РК:
1. Правила субсидирования развития племенного животноводства (Приказ МСХ РК №108,
   ред. №332 от 18.09.2023, ред. №428 от 18.11.2025)
2. Нормы естественной убыли (падежа) с/х животных (Приказ МСХ РК №3-3/1061)
3. Предельно допустимая норма нагрузки на площадь пастбищ (Приказ МСХ РК №3-3/332,
   ред. №394 от 09.12.2024)

Каждая заявка получает rule-score от 0 до 100,
который потом комбинируется с ML-score.
"""
import numpy as np
import pandas as pd


# ─── Приоритеты направлений ───
# п.1-1 Правил: основная цель — развитие племенного животноводства.
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

# ─── Детализированные нормы падежа (Приказ №3-3/1061) ───
# Ключ: (направление_keyword, контекст_keyword) → % падежа.
# Более специфичные правила проверяются первыми.
MORTALITY_DETAILED = [
    # КРС — импорт (повышенный риск перевозки)
    ("скотовод", "импорт",         "авиа",         2.5),
    ("скотовод", "импорт",         "автомобил",    5.0),
    ("скотовод", "импорт",         "морск",        7.5),
    ("скотовод", "импорт",         None,           5.0),   # импорт без уточнения
    # КРС — молодняк (повышенный риск)
    ("скотовод", "молодняк",       None,           2.0),
    ("скотовод", "выращивани",     None,           2.0),
    ("скотовод", "телен",          None,           2.0),
    # КРС молочное — маточное поголовье
    ("скотовод", "молоч",          None,           3.0),
    ("скотовод", "молок",          None,           3.0),
    # КРС мясное — маточное поголовье (базовый)
    ("скотовод", None,             None,           2.0),
    # Овцеводство
    ("овцевод",  "ягнят",          None,           5.0),
    ("овцевод",  "молодняк",       None,           2.0),
    ("овцевод",  None,             None,           3.0),
    # Козоводство
    ("козовод",  "козлят",         None,           5.0),
    ("козовод",  None,             None,           3.0),
    # Коневодство
    ("коневод",  "жеребят",        None,           2.3),
    ("коневод",  "молодняк",       None,           2.7),
    ("коневод",  "табун",          None,           2.3),
    ("коневод",  "конюшен",        None,           0.3),
    ("коневод",  None,             None,           2.1),
    # Верблюдоводство
    ("верблюд",  "верблюжат",      None,           6.0),
    ("верблюд",  None,             None,           1.0),
    # Свиноводство
    ("свиновод", "поросят",        None,          12.5),
    ("свиновод", "доращиван",      None,           5.2),
    ("свиновод", "откорм",         None,           1.0),
    ("свиновод", None,             None,           5.2),
    # Птицеводство
    ("птицевод", "суточн",         None,           6.0),
    ("птицевод", "бройлер",        None,           6.0),
    ("птицевод", "мяс",            None,           7.5),
    ("птицевод", "яичн",           None,           7.5),
    ("птицевод", None,             None,           7.5),
    # Пчеловодство
    ("пчеловод", None,             None,          20.0),
    # Искусственное осеменение (привязано к скотоводству)
    ("осеменен", None,             None,           2.5),
]

# ─── Нормативы субсидий из Приложения 1 Правил (ред. №428 от 18.11.2025) ───
# Используются для валидации: норматив заявки должен соответствовать НПА.
# Ключ: keyword в subsidy_name → норматив тг/единицу
SUBSIDY_NORMATIVES = {
    # Мясное скотоводство
    "быка-производителя мясного":     260_000,
    "маточного поголовья крупного рогатого скота": 260_000,  # отечественный мясной
    "импортированн":                  525_000,  # из дальнего зарубежья (max)
    "выращивании племенного молодняка крупного": 15_000,
    # Молочное скотоводство
    "маточного поголовья крупного рогатого скота молочн": 350_000,
    "производства молока":            45,       # max (от 600 голов)
    "эмбрионов крупного рогатого":    80_000,
    # Осеменение
    "осеменению маточного поголовья": 5_000,
    "семени племенного":              10_000,   # однополое (max)
    # Птицеводство
    "суточного молодняка":            600,
    "производства мяса курицы":       80,       # max (от 15000 тонн)
    # Овцеводство
    "отечественных племенных овец":   26_000,
    "импортированных племенных маточных овец": 52_000,
    "баранов-производителей":         260_000,
    "молодняка мелкого рогатого":     4_000,
    "шерсти":                         200,      # тонкая от 60 кач.
    # Коневодство
    "селекционной и племенной работы": 20_000,
    "жеребцов-производителей продуктивного": 175_000,
    # Верблюдоводство
    "верблюдов-производителей":       175_000,
    # Свиноводство
    "племенных свиней":               100_000,
    "стоимости свиней":               2_000,
    # Прочее
    "маточного поголовья коз":        70_000,
    "кобыльего молока":               60,
    "верблюжьего молока":             55,
    "производства меда":              200,
}

# ─── Нормы нагрузки на пастбища (га/голову КРС) ───
# Приказ МСХ РК №3-3/332 от 14.04.2015 (ред. №394 от 09.12.2024)
# Средняя норма га на 1 голову по восстановленным пастбищам.
PASTURE_NORMS = {
    "Абай":                   {"cattle": 10.0, "sheep": 2.0, "horse": 12.0, "camel": 14.0},
    "Акмолинская":            {"cattle": 8.5,  "sheep": 1.7, "horse": 10.2, "camel": 11.9},
    "Актюбинская":            {"cattle": 10.0, "sheep": 2.0, "horse": 12.0, "camel": 14.0},
    "Алматинская":            {"cattle": 12.0, "sheep": 2.4, "horse": 14.4, "camel": 16.8},
    "Атырауская":             {"cattle": 14.0, "sheep": 2.8, "horse": 16.8, "camel": 19.6},
    "Восточно-Казахстанская": {"cattle": 7.0,  "sheep": 1.4, "horse": 8.4,  "camel": 9.8},
    "Жамбылская":             {"cattle": 11.0, "sheep": 2.2, "horse": 13.2, "camel": 15.4},
    "Жетісу":                 {"cattle": 8.0,  "sheep": 1.6, "horse": 9.6,  "camel": 11.2},
    "Западно-Казахстанская":  {"cattle": 9.0,  "sheep": 1.8, "horse": 10.8, "camel": 12.6},
    "Карагандинская":         {"cattle": 12.0, "sheep": 2.4, "horse": 14.4, "camel": 16.8},
    "Костанайская":           {"cattle": 8.0,  "sheep": 1.6, "horse": 9.6,  "camel": 11.2},
    "Кызылординская":         {"cattle": 15.0, "sheep": 3.0, "horse": 18.0, "camel": 21.0},
    "Мангистауская":          {"cattle": 18.0, "sheep": 3.6, "horse": 21.6, "camel": 25.2},
    "Павлодарская":           {"cattle": 9.5,  "sheep": 1.9, "horse": 11.4, "camel": 13.3},
    "Северо-Казахстанская":   {"cattle": 6.5,  "sheep": 1.3, "horse": 7.8,  "camel": 9.1},
    "Туркестанская":          {"cattle": 13.0, "sheep": 2.6, "horse": 15.6, "camel": 18.2},
    "Ұлытау":                 {"cattle": 14.0, "sheep": 2.8, "horse": 16.8, "camel": 19.6},
    "Астана":                 {"cattle": 8.0,  "sheep": 1.6, "horse": 9.6,  "camel": 11.2},
}

# ─── Классификация типа субсидии по приоритету ───
# Из Правил (Приложение 1, ред. №428): племенная работа (селекция) >
# приобретение племенного поголовья > удешевление стоимости продукции > корма.
# Субсидии на корма — только при ЧС/аномальных погодных (п.6 Прил.1 примечание).
SUBSIDY_TYPE_KEYWORDS = {
    "селекционной и племенной работы": 30,   # Высший — развитие генетики
    "приобретен":   25,                      # Приобретение поголовья
    "выращивани":   22,                      # Выращивание племенного молодняка
    "искусственн":  20,                      # Искусственное осеменение
    "импорт":       28,                      # Импорт — повышенный норматив
    "семени":       18,                      # Приобретение семени
    "эмбрион":      20,                      # Приобретение эмбрионов
    "удешевлени":   15,                      # Удешевление стоимости продукции
    "реализован":   14,                      # Удешевление стоимости реализации
    "шерст":        13,                      # Удешевление шерсти
    "молок":        16,                      # Удешевление молока
    "мяс":          15,                      # Удешевление мяса
    "мед":          12,                      # Удешевление мёда
    "корм":          8,                      # Корма — только при ЧС, низкий приоритет
    "содержан":     10,                      # Содержание поголовья
}


def _subsidy_type_score(subsidy_name: str) -> float:
    """Балл за тип субсидии (0–30)."""
    name = subsidy_name.lower()
    best = 8  # default для неизвестных типов
    for keyword, score in SUBSIDY_TYPE_KEYWORDS.items():
        if keyword in name:
            best = max(best, score)
    return best


def _normative_score(normative: float) -> float:
    """
    Балл за норматив (0–25).
    Высокий норматив → дорогое племенное поголовье → выше приоритет.
    Из Приложения 1 Правил (ред. №428):
      60 тг/гол (суточный молодняк яичного) ... 700 000 тг/гол (импорт КРС молочного).
    """
    if normative <= 0:
        return 0
    log_norm = np.log1p(normative)
    # Scale: log(60)≈4.1 → 0pts, log(700000)≈13.5 → 25pts
    score = np.clip((log_norm - 4) / 9.5 * 25, 0, 25)
    return round(score, 2)


def _normative_compliance_score(normative: float, subsidy_name: str) -> float:
    """
    Балл за соответствие норматива заявки НПА (0–10).
    Приложение 1 Правил: каждому типу субсидии соответствует конкретный норматив.
    Субсидия до утверждённого норматива, но не более 50% стоимости приобретения.

    Если норматив заявки близок к утверждённому — высокий балл.
    Если сильно отклоняется (завышен/занижен) — снижение.
    """
    if normative <= 0:
        return 5.0  # нейтральный для нулевых

    name_lower = subsidy_name.lower()
    expected = None

    # Ищем наиболее специфичное совпадение
    best_match_len = 0
    for keyword, norm_val in SUBSIDY_NORMATIVES.items():
        if keyword in name_lower and len(keyword) > best_match_len:
            expected = norm_val
            best_match_len = len(keyword)

    if expected is None:
        return 5.0  # тип не найден — нейтральный

    # Отношение фактического норматива к ожидаемому
    ratio = normative / expected

    if 0.5 <= ratio <= 1.5:
        # В пределах нормы — высокий балл
        # Ближе к 1.0 — лучше
        score = 10 - abs(1 - ratio) * 8
    elif ratio < 0.5:
        # Сильно заниженный — подозрительно
        score = 3.0
    else:
        # Сильно завышенный (> 150% от нормы)
        score = 2.0

    return round(np.clip(score, 0, 10), 2)


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
        return 5.0
    stats = oblast_stats.get(oblast, {})
    demand_ratio = stats.get("demand_ratio", 0.5)
    score = np.clip((1 - demand_ratio) * 10, 0, 10)
    return round(score, 2)


def _mortality_risk_score(direction: str, subsidy_name: str) -> float:
    """
    Балл за устойчивость к падежу (0–10).
    Приказ МСХ РК №3-3/1061: нормы естественной убыли.

    Детализированный расчёт: учитывает не только направление, но и
    тип субсидии (импорт/местное, молодняк/маточное, тип перевозки).

    Логика: государству выгоднее субсидировать с низким риском потерь.
    Низкий падеж (<3%) → макс балл, высокий (>10%) → мин балл.
    """
    text = (direction + " " + subsidy_name).lower()

    mortality = 5.0  # default: средний риск

    for dir_kw, ctx1_kw, ctx2_kw, rate in MORTALITY_DETAILED:
        if dir_kw not in text:
            continue
        # Проверяем первый контекст (если задан)
        if ctx1_kw is not None and ctx1_kw not in text:
            continue
        # Проверяем второй контекст (если задан)
        if ctx2_kw is not None and ctx2_kw not in text:
            continue
        # Нашли наиболее специфичное совпадение
        mortality = rate
        break

    # Инверсия: низкий падеж → высокий балл
    # 0% → 10, 3% → 7, 7.5% → 2.5, 10% → 0, 20% → 0
    score = np.clip(10 - mortality, 0, 10)
    return round(score, 2)


def _pasture_capacity_score(oblast: str, direction: str) -> float:
    """
    Балл за обеспеченность пастбищами (0–10).
    Приказ МСХ РК №3-3/332 (ред. №394 от 09.12.2024).

    Логика: меньше га/голову = продуктивнее пастбища = ниже затраты
    фермера на корма = выше вероятность успешного использования субсидии.

    Для птицеводства/свиноводства/пчеловодства пастбища не релевантны →
    нейтральный балл.
    """
    dir_lower = direction.lower()

    # Направления без пастбищ — нейтральный балл
    if any(k in dir_lower for k in ("птицевод", "свиновод", "пчеловод")):
        return 5.0

    # Определяем тип животного для нормы
    if "коневод" in dir_lower:
        animal_key = "horse"
    elif "верблюд" in dir_lower:
        animal_key = "camel"
    elif "овцевод" in dir_lower or "козовод" in dir_lower:
        animal_key = "sheep"
    else:
        animal_key = "cattle"

    # Ищем область в справочнике (частичное совпадение)
    norms = None
    oblast_clean = oblast.strip()
    for region, data in PASTURE_NORMS.items():
        if region in oblast_clean or oblast_clean in region:
            norms = data
            break

    if norms is None:
        return 5.0

    ha_per_head = norms.get(animal_key, 10.0)

    # Шкала: 2 га/гол → 10 баллов (лучшие), 18+ га/гол → 1 балл (худшие)
    score = np.clip(10 - (ha_per_head - 2) * (9 / 16), 1, 10)
    return round(score, 2)


def _seasonal_score(month: int, direction: str) -> float:
    """
    Балл за сезонность подачи (0–10).
    Из Приложения 2 Правил: сроки подачи заявок — с 20 января по 20 декабря.
    Оптимальные периоды зависят от случного сезона и направления.
    """
    if month == 0:
        return 5.0

    # Скотоводство/овцеводство: оптимально Q1 (до случного сезона)
    cattle_optimal = {1: 10, 2: 10, 3: 9, 4: 8, 5: 7, 6: 6,
                      7: 5, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9}
    # Птицеводство: круглогодично
    poultry_optimal = {m: 7 for m in range(1, 13)}
    # Коневодство: случной сезон март-июнь
    horse_optimal = {1: 8, 2: 9, 3: 10, 4: 10, 5: 9, 6: 8,
                     7: 6, 8: 5, 9: 5, 10: 6, 11: 7, 12: 7}
    default_optimal = {m: 6 for m in range(1, 13)}

    dir_lower = direction.lower()
    if "скотовод" in dir_lower or "овцевод" in dir_lower or "козовод" in dir_lower:
        return cattle_optimal.get(month, 5)
    elif "птицевод" in dir_lower:
        return poultry_optimal.get(month, 5)
    elif "коневод" in dir_lower:
        return horse_optimal.get(month, 5)
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
            "demand_ratio": count / total * 18,
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
    9 компонентов из 3 НПА, максимум 135 сырых баллов → нормализация к 0–100.

    Правила субсидирования (Приказ №108, ред. №428):
      - Тип субсидии (0–30)
      - Норматив (0–25)
      - Соответствие норматива НПА (0–10) [NEW]
      - Размер стада (0–15)
      - Направление (0–10)
      - Сезонность (0–10)
    Нормы падежа (Приказ №3-3/1061):
      - Устойчивость к падежу — детализированная (0–10)
    Нормы пастбищ (Приказ №3-3/332, ред. №394):
      - Обеспеченность пастбищами (0–10)
    Региональная статистика:
      - Региональный спрос (0–10)
    """
    direction = str(row.get("direction", ""))
    subsidy_name = str(row.get("subsidy_name", ""))
    oblast = str(row.get("oblast", ""))
    normative_val = float(row.get("normative", 0))

    subsidy_s = _subsidy_type_score(subsidy_name)
    normative_s = _normative_score(normative_val)
    compliance_s = _normative_compliance_score(normative_val, subsidy_name)
    herd_s = _herd_size_score(float(row.get("animals_count", 0)))
    direction_s = _direction_score(direction)
    regional_s = _regional_demand_score(oblast, oblast_stats)
    seasonal_s = _seasonal_score(int(row.get("month", 0)), direction)
    mortality_s = _mortality_risk_score(direction, subsidy_name)
    pasture_s = _pasture_capacity_score(oblast, direction)

    total = (subsidy_s + normative_s + compliance_s + herd_s + direction_s
             + regional_s + seasonal_s + mortality_s + pasture_s)
    max_possible = 30 + 25 + 10 + 15 + 10 + 10 + 10 + 10 + 10  # = 130
    score_100 = round(total / max_possible * 100, 1)

    return {
        "rule_score": score_100,
        "subsidy_type_score": subsidy_s,
        "normative_score": normative_s,
        "compliance_score": compliance_s,
        "herd_size_score": herd_s,
        "direction_score": direction_s,
        "regional_score": regional_s,
        "seasonal_score": seasonal_s,
        "mortality_score": mortality_s,
        "pasture_score": pasture_s,
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
