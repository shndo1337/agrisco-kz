"""
AgriScore KZ — Streamlit Dashboard
Запуск: streamlit run app/dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import load_and_clean
from src.features import FEATURE_COLS, FEATURE_NAMES_RU
from src.model import load_model, score_applicants, explain_single, generate_text_explanation

# ─── Page config ───
st.set_page_config(
    page_title="AgriScore KZ",
    page_icon="🌾",
    layout="wide",
)

st.markdown("""
<style>
    .score-high { color: #22c55e; font-weight: bold; font-size: 1.2em; }
    .score-mid  { color: #f59e0b; font-weight: bold; font-size: 1.2em; }
    .score-low  { color: #ef4444; font-weight: bold; font-size: 1.2em; }
    .metric-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load data & model ───
@st.cache_resource
def init():
    model, encoders = load_model()
    df = load_and_clean()
    scored, shap_vals, X = score_applicants(df, model, encoders)
    return model, encoders, scored, shap_vals, X


try:
    model, encoders, df_scored, shap_values, X_matrix = init()
except Exception as e:
    st.error(f"Сначала обучите модель: `py train.py`\n\nОшибка: {e}")
    st.stop()


# ─── Sidebar filters ───
st.sidebar.title("🌾 AgriScore KZ")
st.sidebar.markdown("Скоринг сельхозпроизводителей")
st.sidebar.divider()

oblasts = ["Все"] + sorted(df_scored["oblast"].unique().tolist())
sel_oblast = st.sidebar.selectbox("Область", oblasts)

directions = ["Все"] + sorted(df_scored["direction"].unique().tolist())
sel_direction = st.sidebar.selectbox("Направление", directions)

statuses = ["Все"] + sorted(df_scored["status"].unique().tolist())
sel_status = st.sidebar.selectbox("Статус заявки", statuses)

score_range = st.sidebar.slider("Диапазон балла", 0, 100, (0, 100))

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

# ─── Tabs ───
tab_rank, tab_shortlist, tab_analytics, tab_detail = st.tabs(
    ["📊 Рейтинг", "📋 Шорт-лист", "📈 Аналитика", "🔍 Детали заявки"]
)

# ═══════════ TAB 1: Ranking ═══════════
with tab_rank:
    st.header("Рейтинг заявок")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего заявок", f"{len(df_view):,}")
    c2.metric("Средний балл", f"{df_view['score'].mean():.1f}")
    c3.metric("Медианный балл", f"{df_view['score'].median():.1f}")
    approved_pct = (df_view["status"].isin({"Исполнена", "Одобрена"})).mean() * 100
    c4.metric("% одобренных", f"{approved_pct:.1f}%")

    show_cols = ["num", "oblast", "district", "direction", "subsidy_name",
                 "status", "amount", "normative", "animals_count", "score"]
    display = df_view[show_cols].sort_values("score", ascending=False).head(500)
    display.columns = [
        "№", "Область", "Район", "Направление", "Тип субсидии",
        "Статус", "Сумма (₸)", "Норматив", "Кол-во голов", "Балл"
    ]

    st.dataframe(
        display.style.background_gradient(subset=["Балл"], cmap="RdYlGn", vmin=0, vmax=100),
        height=500,
    )

# ═══════════ TAB 2: Shortlist ═══════════
with tab_shortlist:
    st.header("Шорт-лист для комиссии")

    top_n = st.slider("Количество кандидатов", 10, 200, 50)
    shortlist = df_view.nlargest(top_n, "score")

    st.success(f"Топ-{top_n} заявителей с наивысшим баллом")

    sl_cols = ["num", "oblast", "district", "direction", "amount",
               "animals_count", "score"]
    sl_display = shortlist[sl_cols].copy()
    sl_display.columns = [
        "№", "Область", "Район", "Направление",
        "Сумма (₸)", "Кол-во голов", "Балл"
    ]

    st.dataframe(
        sl_display.style.background_gradient(subset=["Балл"], cmap="RdYlGn", vmin=0, vmax=100),
    )

    csv = sl_display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Скачать шорт-лист (CSV)",
        csv,
        "shortlist.csv",
        "text/csv",
    )

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
            .mean()
            .sort_values(ascending=True)
            .reset_index()
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
        st.subheader("По направлениям")
        avg_by_dir = (
            df_view.groupby("direction")["score"]
            .agg(["mean", "count"])
            .reset_index()
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

    with col_d:
        st.subheader("Статусы заявок")
        status_counts = df_view["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig_pie = px.pie(
            status_counts, values="count", names="status",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, key="pie_statuses")

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
            st.markdown(f"### Балл: <span style='color:{color};font-size:2em'>{score_val:.1f}</span>/100",
                        unsafe_allow_html=True)
            st.markdown(f"**Область:** {row['oblast']}")
            st.markdown(f"**Район:** {row['district']}")
            st.markdown(f"**Направление:** {row['direction']}")
            st.markdown(f"**Статус:** {row['status']}")
            st.markdown(f"**Сумма:** {row['amount']:,.0f} ₸")
            st.markdown(f"**Норматив:** {row['normative']:,.0f} ₸/гол")
            st.markdown(f"**Кол-во голов:** {row['animals_count']:,.0f}")

        with col2:
            sv = shap_values[pos_in_scored]
            expl = explain_single(sv, FEATURE_COLS, top_n=10)

            names = [e["name_ru"] for e in expl]
            vals = [e["shap_value"] for e in expl]
            colors = ["#22c55e" if v > 0 else "#ef4444" for v in vals]

            fig_shap = go.Figure(go.Bar(
                x=vals, y=names, orientation="h",
                marker_color=colors,
            ))
            fig_shap.update_layout(
                title="SHAP — вклад факторов в балл",
                xaxis_title="Влияние на балл",
                height=400,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_shap, key="shap_detail")

        st.subheader("Текстовое объяснение")
        text = generate_text_explanation(expl, score_val)
        st.code(text, language=None)
