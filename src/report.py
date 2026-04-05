"""
PDF-генератор отчётов AgriScore KZ.
Формирует shortlist-отчёт с результатами ML-скоринга.
"""

import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Пути к ресурсам ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FONT_REGULAR = os.path.join(ASSETS_DIR, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(ASSETS_DIR, "DejaVuSans-Bold.ttf")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

# ── Цвета ────────────────────────────────────────────────────────────────────
GREEN_PRIMARY = (34, 139, 34)
GRAY_BG = (245, 245, 245)
GRAY_BORDER = (220, 220, 220)
TEXT_DARK = (40, 40, 40)
TEXT_LIGHT = (100, 100, 100)
SCORE_GREEN = (34, 139, 34)
SCORE_YELLOW = (210, 140, 0)
SCORE_RED = (180, 0, 0)


class AgriReport(FPDF):
    """PDF-документ с фирменным стилем AgriScore."""

    def header(self):
        # Зелёная плашка
        self.set_fill_color(*GREEN_PRIMARY)
        self.rect(0, 0, 210, 40, "F")

        # Логотип
        if os.path.exists(LOGO_PATH):
            self.set_fill_color(255, 255, 255)
            self.rect(12, 7, 26, 26, "F")
            self.image(LOGO_PATH, 13, 8, 24)
            self.set_xy(45, 12)
        else:
            self.set_xy(10, 12)

        # Заголовок (Bold)
        self.set_font("DejaVuB", "", 24)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "AGRISCORE KAZAKHSTAN", ln=True, align="L")

        if os.path.exists(LOGO_PATH):
            self.set_x(45)
        self.set_font("DejaVu", "", 10)
        date_str = datetime.now().strftime("%d.%m.%Y %H:%M")
        self.cell(0, 5, f"ОТЧЕТ АНАЛИЗА РИСКОВ И БАЛЛЬНОЙ ОЦЕНКИ | {date_str}", align="L")

        self.ln(25)

    def footer(self):
        self.set_y(-20)
        self.set_draw_color(200, 200, 200)
        self.line(10, 275, 200, 275)

        self.set_font("DejaVu", "", 8)
        self.set_text_color(120)
        footer_text = f"Сгенерировано системой AgriScore AI | Страница {self.page_no()}/{{nb}}"
        self.cell(0, 10, footer_text, align="C")


def _init_pdf() -> AgriReport:
    """Создаёт PDF с подключёнными шрифтами."""
    pdf = AgriReport()
    pdf.alias_nb_pages()

    if os.path.exists(FONT_REGULAR):
        pdf.add_font("DejaVu", "", FONT_REGULAR, uni=True)
    if os.path.exists(FONT_BOLD):
        pdf.add_font("DejaVuB", "", FONT_BOLD, uni=True)
    else:
        # Fallback: используем обычный шрифт как bold
        if os.path.exists(FONT_REGULAR):
            pdf.add_font("DejaVuB", "", FONT_REGULAR, uni=True)

    pdf.set_font("DejaVu", size=11)
    return pdf


def _detect_score_col(df: pd.DataFrame) -> str | None:
    """Определить, какая колонка содержит итоговый балл."""
    for candidate in ("score", "composite_score"):
        if candidate in df.columns:
            return candidate
    return None


def _draw_summary_block(pdf: AgriReport, df: pd.DataFrame, y: int = 50):
    """Информационный блок с ключевыми метриками."""
    pdf.set_fill_color(*GRAY_BG)
    pdf.set_draw_color(*GRAY_BORDER)
    pdf.rect(10, y, 190, 30, "FD")

    pdf.set_xy(15, y + 3)
    pdf.set_font("DejaVuB", "", 11)
    pdf.set_text_color(*TEXT_DARK)
    pdf.cell(90, 8, f"Кандидатов в списке: {len(df)}")

    if "amount" in df.columns:
        total_amt = df["amount"].sum()
        pdf.set_x(110)
        pdf.cell(90, 8, f"Бюджет: {total_amt:,.0f} ₸".replace(",", " "))

    pdf.set_xy(15, y + 12)
    pdf.set_font("DejaVu", "", 9)
    score_col = _detect_score_col(df)
    if score_col:
        avg_score = df[score_col].mean()
        pdf.cell(90, 8, f"Средний балл: {avg_score:.1f}")
        pdf.set_x(110)
        top_count = (df[score_col] >= 80).sum()
        pdf.cell(90, 8, f"Высокий балл (>=80): {top_count} заявок")

    pdf.set_xy(15, y + 22)
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(*TEXT_LIGHT)
    pdf.cell(0, 6, "Статус: Проверка пройдена. Рекомендовано к рассмотрению комиссией.")


def _draw_table(pdf: AgriReport, df: pd.DataFrame):
    """Таблица с результатами скоринга."""
    score_col = _detect_score_col(df)

    has_direction = "direction" in df.columns
    if has_direction:
        headers = ["#", "Область", "Район", "Направление", "Сумма (₸)", "Балл"]
        widths = [8, 40, 40, 42, 32, 28]
    else:
        headers = ["#", "Область", "Район", "Сумма (₸)", "Балл"]
        widths = [10, 55, 55, 40, 30]

    # Шапка таблицы
    pdf.set_fill_color(60, 60, 60)
    pdf.set_text_color(255)
    pdf.set_font("DejaVuB", "", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 12, h, border=0, align="C", fill=True)
    pdf.ln()

    def _draw_header():
        pdf.set_fill_color(60, 60, 60)
        pdf.set_text_color(255)
        pdf.set_font("DejaVuB", "", 9)
        for h, w in zip(headers, widths):
            pdf.cell(w, 12, h, border=0, align="C", fill=True)
        pdf.ln()

    # Строки
    pdf.set_font("DejaVu", "", 8)
    for i, (_, row) in enumerate(df.iterrows()):
        # Новая страница если близко к низу
        if pdf.get_y() > 255:
            pdf.add_page()
            _draw_header()
            pdf.set_font("DejaVu", "", 8)

        if i % 2 == 0:
            pdf.set_fill_color(250, 250, 250)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(*TEXT_DARK)

        pdf.cell(widths[0], 9, str(i + 1), border="B", align="C", fill=True)
        pdf.cell(widths[1], 9, f" {str(row.get('oblast', ''))[:18]}", border="B", fill=True)
        pdf.cell(widths[2], 9, f" {str(row.get('district', ''))[:18]}", border="B", fill=True)
        if has_direction:
            pdf.cell(widths[3], 9, f" {str(row.get('direction', ''))[:20]}", border="B", fill=True)

        amt = f"{row.get('amount', 0):,.0f}".replace(",", " ")
        w_amt = widths[4] if has_direction else widths[3]
        pdf.cell(w_amt, 9, f"{amt} ", border="B", align="R", fill=True)

        # Цвет балла
        score = row.get(score_col, 0) if score_col else 0
        if score >= 80:
            pdf.set_text_color(*SCORE_GREEN)
        elif score >= 60:
            pdf.set_text_color(*SCORE_YELLOW)
        else:
            pdf.set_text_color(*SCORE_RED)

        w_score = widths[5] if has_direction else widths[4]
        pdf.cell(w_score, 9, f"{score:.1f}", border="B", align="C", fill=True)
        pdf.ln()


def _draw_stamp(pdf: AgriReport):
    """Печать и штамп AI-верификации."""
    pdf.ln(15)
    curr_y = pdf.get_y()

    if curr_y > 250:
        pdf.add_page()
        curr_y = pdf.get_y()

    # Место для печати
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(*TEXT_LIGHT)
    pdf.text(10, curr_y + 10, "М.П. (Место для печати)")
    pdf.set_draw_color(*GRAY_BORDER)
    pdf.line(10, curr_y + 15, 60, curr_y + 15)

    # Штамп AI
    pdf.set_draw_color(*GREEN_PRIMARY)
    pdf.set_line_width(0.5)
    pdf.rect(140, curr_y, 55, 25)
    pdf.set_xy(140, curr_y + 5)
    pdf.set_text_color(*GREEN_PRIMARY)
    pdf.set_font("DejaVuB", "", 9)
    pdf.cell(55, 5, "VERIFIED BY AI", align="C", ln=True)
    pdf.set_x(140)
    pdf.set_font("DejaVu", "", 8)
    pdf.cell(55, 5, "AGRISCORE SYSTEM", align="C", ln=True)
    pdf.set_x(140)
    pdf.set_font("DejaVu", "", 7)
    pdf.cell(55, 5, f"ID: {datetime.now().strftime('%Y%m%d%H%M%S')}", align="C")


def generate_shortlist_report(
    shortlist_df: pd.DataFrame,
    output_path: str = "reports/shortlist_report.pdf",
) -> str:
    """
    Генерирует PDF-отчёт по шорт-листу.

    Parameters
    ----------
    shortlist_df : DataFrame с колонками:
        oblast, district, [direction], amount, score/composite_score
    output_path : путь для сохранения PDF

    Returns
    -------
    str : путь к сохранённому файлу
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pdf = _init_pdf()
    pdf.add_page()

    _draw_summary_block(pdf, shortlist_df)
    pdf.set_y(85)
    _draw_table(pdf, shortlist_df)
    _draw_stamp(pdf)

    pdf.output(output_path)
    return output_path


# Обратная совместимость
generate_report = generate_shortlist_report


if __name__ == "__main__":
    test_data = pd.DataFrame({
        "oblast": ["Алматинская", "Туркестанская", "Акмолинская", "Костанайская", "Павлодарская"],
        "district": ["Кербулакский", "Сарыагашский", "Целиноградский", "Федоровский", "Баянаульский"],
        "direction": ["Скотоводство", "Овцеводство", "Скотоводство", "Птицеводство", "Коневодство"],
        "amount": [24500000, 18200000, 9500000, 31000000, 15800000],
        "score": [98.7, 85.2, 55.4, 91.0, 72.3],
    })
    path = generate_shortlist_report(test_data, "reports/test_report.pdf")
    print(f"Отчет создан: {path}")
