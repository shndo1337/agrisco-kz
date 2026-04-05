"""
AgriScore KZ — Streamlit Dashboard v2
Запуск: streamlit run app/dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import load_and_clean
from src.features import FEATURE_COLS, FEATURE_NAMES_RU
from src.model import load_model, score_applicants, explain_single, generate_text_explanation
from src.report import generate_shortlist_report

# ─── Page config ───
st.set_page_config(
    page_title="AgriScore KZ",
    page_icon="🌾",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 4rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f1923 0%, #1a2836 100%);
        border: 1px solid #2d4a5c;
        border-radius: 10px;
        padding: 12px 16px;
    }
    /* Tabs as visible buttons */
    button[data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border: 1px solid #4a90d9 !important;
        border-radius: 8px !important;
        margin-right: 6px !important;
        background: #1a2836 !important;
        color: #e0e0e0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #3b82f6 !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
    }
    div[data-baseweb="tab-list"] {
        gap: 6px;
        overflow-x: auto;
    }
    /* Hide the default tab underline */
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }
    div[data-baseweb="tab-border"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load data & model ───
@st.cache_resource
def init():
    models, encoders, oblast_stats = load_model()
    df = load_and_clean()
    scored, shap_vals, X = score_applicants(df, models, encoders, oblast_stats)
    return models, encoders, oblast_stats, scored, shap_vals, X


try:
    models, encoders, oblast_stats, df_scored, shap_values, X_matrix = init()
except Exception as e:
    st.error(f"Сначала обучите модель: `python train.py`\n\nОшибка: {e}")
    st.stop()


# ─── Sidebar ───
st.sidebar.title("🌾 AgriScore KZ")
st.sidebar.caption("Merit-based скоринг сельхозпроизводителей")
st.sidebar.divider()

oblasts = ["Все"] + sorted(df_scored["oblast"].unique().tolist())
sel_oblast = st.sidebar.selectbox("Область", oblasts)

directions = ["Все"] + sorted(df_scored["direction"].unique().tolist())
sel_direction = st.sidebar.selectbox("Направление", directions)

statuses = ["Все"] + sorted(df_scored["status"].unique().tolist())
sel_status = st.sidebar.selectbox("Статус заявки", statuses)

score_range = st.sidebar.slider("Диапазон балла", 0, 100, (0, 100))

# ─── Tech Stack (collapsible) ───
st.sidebar.divider()
with st.sidebar.expander("⚙️ О системе"):
    st.markdown("""
**ML Pipeline**
- Ensemble: XGBoost (55%) + LightGBM (45%)
- 30 engineered features, 5-fold CV
- ROC-AUC: **0.94**
- SHAP-объяснения каждой заявки

**Rule Engine (3 НПА)**
- 9 компонентов из 3 приказов МСХ РК
- 34 нормы падежа, 18 областей пастбищ
- Composite: 60% ML + 40% Rules

**Infrastructure**
- 🐳 Docker + docker-compose
- 🧪 26 unit tests (pytest)
- 🔄 CI/CD (GitHub Actions)
- 🔐 API auth (X-API-Key)
- 📝 Structured logging
- 📄 PDF-отчёты (fpdf2)

**API**: `POST /score`, `POST /score/batch`
""")
    st.caption("v0.3.0 | Python 3.12")

# Apply filters
mask = pd.Series(True, index=df_scored.index)
if sel_oblast != "Все":
    mask &= df_scored["oblast"] == sel_oblast
if sel_direction != "Все":
    mask &= df_scored["direction"] == sel_direction
if sel_status != "Все":
    mask &= df_scored["status"] == sel_status
mask &= df_scored["score"].between(score_range[0], score_range[1])

df_view = df_scored[mask].copy()


# ─── Onboarding ───
if "onboarding_dismissed" not in st.session_state:
    st.session_state.onboarding_dismissed = False

if not st.session_state.onboarding_dismissed:
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a3a2a 0%, #1a2836 100%);
                    border: 1px solid #2d5a3c; border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;">
        <h3 style="color: #64FFDA; margin-top: 0;">Добро пожаловать в AgriScore KZ</h3>
        <p style="color: #CDD6F4; font-size: 15px; margin-bottom: 12px;">
        AI-система merit-based скоринга заявок на субсидии. Каждая заявка получает балл от 0 до 100
        на основе <b>ML-модели</b> (60%) и <b>экспертных правил из 3 НПА МСХ РК</b> (40%).
        </p>
        <table style="color: #CDD6F4; font-size: 14px; border-collapse: collapse; width: 100%;">
        <tr><td style="padding: 4px 12px 4px 0;"><b>📊 Рейтинг</b></td><td>Все заявки с фильтрами по области, направлению, статусу</td></tr>
        <tr><td style="padding: 4px 12px 4px 0;"><b>📋 Шорт-лист</b></td><td>Топ-N кандидатов для комиссии + экспорт CSV/PDF</td></tr>
        <tr><td style="padding: 4px 12px 4px 0;"><b>📈 Аналитика</b></td><td>Распределения, SHAP feature importance, сравнение ML vs Rules</td></tr>
        <tr><td style="padding: 4px 12px 4px 0;"><b>🔍 Детали</b></td><td>SHAP-объяснение конкретной заявки + разбивка Rule-компонентов</td></tr>
        <tr><td style="padding: 4px 12px 4px 0;"><b>🎛 What-If</b></td><td>Симулятор: измените параметры → увидите как изменится балл</td></tr>
        <tr><td style="padding: 4px 12px 4px 0;"><b>📤 Загрузка</b></td><td>Загрузите CSV с новыми заявками → получите ранжированный список</td></tr>
        </table>
        <p style="color: #8892B0; font-size: 13px; margin-top: 12px; margin-bottom: 0;">
        Используйте фильтры в боковой панели для навигации. AI помогает принимать решения, но финальное слово — за комиссией.
        </p>
        </div>
        """, unsafe_allow_html=True)
    if st.button("Понятно, начать работу", type="primary"):
        st.session_state.onboarding_dismissed = True
        st.rerun()


# ─── Tabs ───
tab_rank, tab_shortlist, tab_analytics, tab_detail, tab_whatif, tab_upload = st.tabs([
    "📊 Рейтинг", "📋 Шорт-лист", "📈 Аналитика",
    "🔍 Детали", "🎛 What-If", "📤 Загрузка",
])


# ═══════════ TAB 1: Ranking ═══════════
with tab_rank:
    st.header("Рейтинг заявок")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Всего заявок", f"{len(df_view):,}")
    c2.metric("Средний балл", f"{df_view['score'].mean():.1f}")
    c3.metric("ML-компонент", f"{df_view['ml_score'].mean():.1f}")
    c4.metric("Rule-компонент", f"{df_view['rule_score'].mean():.1f}")
    approved_pct = (df_view["status"].isin({"Исполнена", "Одобрена"})).mean() * 100
    c5.metric("% одобренных", f"{approved_pct:.1f}%")

    show_cols = ["num", "oblast", "district", "direction", "subsidy_name",
                 "status", "amount", "normative", "animals_count",
                 "ml_score", "rule_score", "score"]
    display = df_view[show_cols].sort_values("score", ascending=False).head(500)
    display.columns = [
        "№", "Область", "Район", "Направление", "Тип субсидии",
        "Статус", "Сумма (₸)", "Норматив", "Кол-во голов",
        "ML балл", "Rule балл", "Итог. балл"
    ]

    st.dataframe(
        display.style.background_gradient(subset=["Итог. балл"], cmap="RdYlGn", vmin=0, vmax=100),
        height=500,
    )


# ═══════════ TAB 2: Shortlist ═══════════
with tab_shortlist:
    st.header("Шорт-лист для комиссии")

    top_n = st.slider("Количество кандидатов", 10, 200, 50)
    shortlist = df_view.nlargest(top_n, "score")

    st.success(f"Топ-{top_n} заявителей с наивысшим баллом (композитный: 60% ML + 40% Rules)")

    sl_cols = ["num", "oblast", "district", "direction", "amount",
               "animals_count", "ml_score", "rule_score", "score"]
    sl_display = shortlist[sl_cols].copy()
    sl_display.columns = [
        "№", "Область", "Район", "Направление",
        "Сумма (₸)", "Кол-во голов", "ML балл", "Rule балл", "Итог. балл"
    ]

    st.dataframe(
        sl_display.style.background_gradient(subset=["Итог. балл"], cmap="RdYlGn", vmin=0, vmax=100),
    )

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv = sl_display.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Скачать шорт-лист (CSV)", csv, "shortlist.csv", "text/csv")

    with dl_col2:
        # PDF-отчёт
        pdf_df = shortlist[["oblast", "district", "direction", "amount", "score"]].copy()
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                generate_shortlist_report(pdf_df, tmp.name)
                tmp.seek(0)
                pdf_bytes = open(tmp.name, "rb").read()
            st.download_button(
                "📄 Скачать отчёт (PDF)", pdf_bytes,
                "shortlist_report.pdf", "application/pdf",
            )
        except Exception as e:
            st.warning(f"PDF недоступен: {e}")


# ═══════════ TAB 3: Analytics ═══════════
with tab_analytics:
    st.header("Аналитика")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Распределение баллов")
        fig_hist = px.histogram(
            df_view, x="score", nbins=50,
            color_discrete_sequence=["#3b82f6"],
            labels={"score": "Балл", "count": "Количество"},
        )
        fig_hist.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_hist, key="hist_scores")

    with col_b:
        st.subheader("Средний балл по областям")
        avg_by_oblast = (
            df_view.groupby("oblast")["score"]
            .mean().sort_values(ascending=True).reset_index()
        )
        fig_bar = px.bar(
            avg_by_oblast, x="score", y="oblast", orientation="h",
            color="score", color_continuous_scale="RdYlGn",
            labels={"score": "Средний балл", "oblast": ""},
        )
        fig_bar.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, key="bar_oblasts")

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("ML vs Rule-based компоненты")
        scatter_data = df_view.sample(min(2000, len(df_view)), random_state=42)
        fig_scatter = px.scatter(
            scatter_data, x="ml_score", y="rule_score",
            color="score", color_continuous_scale="RdYlGn",
            labels={"ml_score": "ML балл", "rule_score": "Rule балл", "score": "Итоговый"},
            opacity=0.5,
        )
        fig_scatter.update_layout(height=350)
        st.plotly_chart(fig_scatter, key="scatter_ml_rule")

    with col_d:
        st.subheader("По направлениям")
        avg_by_dir = (
            df_view.groupby("direction")["score"]
            .agg(["mean", "count"]).reset_index()
        )
        avg_by_dir.columns = ["direction", "avg_score", "count"]
        fig_dir = px.bar(
            avg_by_dir.sort_values("avg_score"),
            x="avg_score", y="direction", orientation="h",
            color="avg_score", color_continuous_scale="RdYlGn",
            labels={"avg_score": "Средний балл", "direction": ""},
        )
        fig_dir.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_dir, key="bar_directions")

    # Feature importance
    st.subheader("Важность признаков (SHAP)")
    mean_shap = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": mean_shap,
        "name_ru": [FEATURE_NAMES_RU.get(f, f) for f in FEATURE_COLS],
    }).sort_values("importance", ascending=True).tail(15)

    fig_imp = px.bar(
        feat_imp, x="importance", y="name_ru", orientation="h",
        color="importance", color_continuous_scale="Blues",
        labels={"importance": "Средний |SHAP|", "name_ru": ""},
    )
    fig_imp.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig_imp, key="feature_importance")


# ═══════════ TAB 4: Detail ═══════════
with tab_detail:
    st.header("Детальный анализ заявки")

    row_idx = st.number_input(
        "Введите № заявки (из таблицы рейтинга)",
        min_value=int(df_scored["num"].min()),
        max_value=int(df_scored["num"].max()),
        value=int(df_scored["num"].iloc[0]),
    )

    matches = df_scored[df_scored["num"] == row_idx]
    if matches.empty:
        st.warning("Заявка не найдена")
    else:
        row = matches.iloc[0]
        pos_in_df = matches.index[0]
        pos_in_scored = df_scored.index.get_loc(pos_in_df)

        col1, col2 = st.columns([1, 2])

        with col1:
            score_val = row["score"]
            color = "#22c55e" if score_val >= 70 else "#f59e0b" if score_val >= 40 else "#ef4444"
            st.markdown(
                f"### Балл: <span style='color:{color};font-size:2em'>{score_val:.1f}</span>/100",
                unsafe_allow_html=True,
            )
            st.markdown(f"**ML балл:** {row['ml_score']:.1f} | **Rule балл:** {row['rule_score']:.1f}")
            st.divider()
            st.markdown(f"**Область:** {row['oblast']}")
            st.markdown(f"**Район:** {row['district']}")
            st.markdown(f"**Направление:** {row['direction']}")
            st.markdown(f"**Статус:** {row['status']}")
            st.markdown(f"**Сумма:** {row['amount']:,.0f} ₸")
            st.markdown(f"**Норматив:** {row['normative']:,.0f} ₸/гол")
            st.markdown(f"**Кол-во голов:** {row['animals_count']:,.0f}")

        with col2:
            sv = shap_values[pos_in_scored]
            expl = explain_single(sv, FEATURE_COLS, top_n=12)

            names = [e["name_ru"] for e in expl]
            vals = [e["shap_value"] for e in expl]
            colors = ["#22c55e" if v > 0 else "#ef4444" for v in vals]

            fig_shap = go.Figure(go.Bar(
                x=vals, y=names, orientation="h",
                marker_color=colors,
            ))
            fig_shap.update_layout(
                title="SHAP — вклад факторов в ML-балл",
                xaxis_title="Влияние на балл",
                height=450,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_shap, key="shap_detail")

        # Rule score breakdown
        st.subheader("Разбивка Rule-based балла")
        rule_cols = {
            "subsidy_type_score": ("Тип субсидии", 30),
            "normative_score": ("Норматив", 25),
            "compliance_score": ("Соответствие НПА", 10),
            "herd_size_score": ("Размер стада", 15),
            "direction_score": ("Направление", 10),
            "regional_score": ("Регион", 10),
            "seasonal_score": ("Сезонность", 10),
            "mortality_score": ("Устойчивость к падежу", 10),
            "pasture_score": ("Пастбищная база", 10),
        }

        rule_data = []
        for col, (name, max_val) in rule_cols.items():
            val = row.get(col, 0)
            rule_data.append({"Критерий": name, "Балл": val, "Макс": max_val})

        rule_df = pd.DataFrame(rule_data)
        fig_rule = go.Figure()
        fig_rule.add_trace(go.Bar(
            x=rule_df["Балл"], y=rule_df["Критерий"], orientation="h",
            marker_color="#3b82f6", name="Балл",
        ))
        fig_rule.add_trace(go.Bar(
            x=rule_df["Макс"] - rule_df["Балл"], y=rule_df["Критерий"], orientation="h",
            marker_color="#1e293b", name="Остаток",
        ))
        fig_rule.update_layout(
            barmode="stack", height=300, showlegend=False,
            title="Rule-based компоненты",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_rule, key="rule_breakdown")

        st.subheader("Текстовое объяснение")
        text = generate_text_explanation(expl, score_val, row["ml_score"], row["rule_score"])
        st.code(text, language=None)


# ═══════════ TAB 5: What-If ═══════════
with tab_whatif:
    st.header("🎛 What-If симулятор")
    st.caption("Измените параметры заявки и увидите как изменится балл")

    wc1, wc2 = st.columns(2)

    with wc1:
        wi_oblast = st.selectbox("Область", sorted(df_scored["oblast"].unique()), key="wi_obl")
        wi_direction = st.selectbox("Направление", sorted(df_scored["direction"].unique()), key="wi_dir")
        wi_subsidy = st.selectbox(
            "Тип субсидии",
            sorted(df_scored["subsidy_name"].unique()),
            key="wi_sub",
        )

    with wc2:
        wi_normative = st.slider("Норматив (₸/гол)", 20, 400_000, 15_000, step=1000, key="wi_norm")
        wi_animals = st.slider("Количество голов", 1, 5000, 100, key="wi_anim")
        wi_month = st.slider("Месяц подачи", 1, 12, 3, key="wi_month")

    wi_amount = wi_normative * wi_animals

    st.markdown(f"**Расчётная сумма:** {wi_amount:,.0f} ₸")

    if st.button("Рассчитать балл", type="primary"):
        wi_row = pd.DataFrame([{
            "num": 0, "date": None, "app_number": "whatif",
            "akimat": "", "status": "Новая", "approved": None,
            "oblast": wi_oblast, "district": "Неизвестно",
            "direction": wi_direction, "subsidy_name": wi_subsidy,
            "normative": wi_normative, "amount": wi_amount,
            "animals_count": wi_animals,
            "month": wi_month, "hour": 12, "day_of_year": wi_month * 30,
        }])

        try:
            scored_wi, shap_wi, _ = score_applicants(wi_row, models, encoders, oblast_stats)
            r = scored_wi.iloc[0]

            rc1, rc2, rc3 = st.columns(3)
            s_color = "#22c55e" if r["score"] >= 70 else "#f59e0b" if r["score"] >= 40 else "#ef4444"
            rc1.markdown(f"### Итог: <span style='color:{s_color};font-size:1.5em'>{r['score']:.1f}</span>",
                         unsafe_allow_html=True)
            rc2.metric("ML балл", f"{r['ml_score']:.1f}")
            rc3.metric("Rule балл", f"{r['rule_score']:.1f}")

            expl_wi = explain_single(shap_wi[0], FEATURE_COLS, top_n=8)
            names_wi = [e["name_ru"] for e in expl_wi]
            vals_wi = [e["shap_value"] for e in expl_wi]
            colors_wi = ["#22c55e" if v > 0 else "#ef4444" for v in vals_wi]

            fig_wi = go.Figure(go.Bar(x=vals_wi, y=names_wi, orientation="h", marker_color=colors_wi))
            fig_wi.update_layout(title="SHAP факторы", height=350, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_wi, key="shap_whatif")
        except Exception as e:
            st.error(f"Ошибка: {e}")


# ═══════════ TAB 6: Upload ═══════════
with tab_upload:
    st.header("📤 Загрузка новых заявок")
    st.caption("Загрузите CSV с новыми заявками для скоринга")

    st.markdown("""
    **Формат CSV** (колонки):
    `oblast, district, direction, subsidy_name, normative, amount, month`
    """)

    uploaded = st.file_uploader("Загрузить CSV", type=["csv"])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"Загружено {len(new_df)} заявок")

            # Add missing columns
            for col in ["num", "date", "app_number", "akimat", "status", "approved",
                        "hour", "day_of_year", "animals_count"]:
                if col not in new_df.columns:
                    new_df[col] = 0 if col != "status" else "Новая"

            if "animals_count" not in new_df.columns or new_df["animals_count"].sum() == 0:
                new_df["animals_count"] = np.where(
                    new_df["normative"] > 0,
                    new_df["amount"] / new_df["normative"],
                    0,
                )

            scored_new, _, _ = score_applicants(new_df, models, encoders, oblast_stats)

            show = scored_new[["oblast", "district", "direction", "subsidy_name",
                               "amount", "normative", "ml_score", "rule_score", "score"]]
            show = show.sort_values("score", ascending=False)
            show.columns = ["Область", "Район", "Направление", "Тип субсидии",
                            "Сумма", "Норматив", "ML балл", "Rule балл", "Итог. балл"]

            st.dataframe(
                show.style.background_gradient(subset=["Итог. балл"], cmap="RdYlGn", vmin=0, vmax=100)
            )

            csv_out = show.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Скачать результат", csv_out, "scored_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Ошибка при обработке: {e}")
