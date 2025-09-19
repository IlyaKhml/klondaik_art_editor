import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from hashlib import md5

from constants import (
    IMMUNITY_COLS, CAP_COLS, RESTORE_COLS, UTILITY_COLS,
    compute_artifact_scores,
)

def render_char_heatmap_by_type(
    df: pd.DataFrame,
    char_groups: dict[str, list[str]],
    agg_mode: str = "sum"  # "sum" или "presence"
):  
    if agg_mode == "sum":
        st.subheader(f"📊 Heatmap Сумма характеристик")
    else:
        st.subheader(f"📊 Heatmap Наличие характеристик")

    if "type" not in df.columns:
        st.warning("Колонка 'type' отсутствует в данных")
        return

    df_filtered = df.dropna(subset=["type"])

    for idx, (group_name, selected_cols) in enumerate(char_groups.items()):
        selected_cols = [col for col in selected_cols if col in df_filtered.columns]
        if not selected_cols:
            continue

        # Уникальный ключ на основе имени группы
        group_key = md5(group_name.encode()).hexdigest()[:8]

        expanded = False if (st.session_state.selected_level or st.session_state.selected_type) else True

        with st.expander(f"▶ {group_name}", expanded=expanded):
            if agg_mode == "sum":
                pivot_df = df_filtered.groupby("type")[selected_cols].sum().T

                fig = px.imshow(
                    pivot_df,
                    labels=dict(x="Тип артефакта", y="Характеристика", color="Сумма"),
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    text_auto=".2f",
                    height=800
                )

            elif agg_mode == "presence":
                # Все типы артефактов и характеристики, которые должны быть на графике
                all_types = sorted(df_filtered["type"].dropna().unique())  # ось X
                all_chars = selected_cols                                   # ось Y

                # 1. Маска присутствия
                presence_df = (df_filtered[all_chars].notna()) & (df_filtered[all_chars] != 0)
                group = df_filtered["type"]

                # 2. Кол-во заполненных значений на каждую характеристику (по группам)
                sum_per_group = presence_df.groupby(group).sum()
                sum_per_group = sum_per_group.reindex(index=all_types, columns=all_chars, fill_value=0)

                # 3. Общее количество строк на каждую группу
                count_per_group = df_filtered.groupby(group).size()
                count_per_group = count_per_group.reindex(all_types, fill_value=0)

                # 4. Процент (с защитой от деления на 0)
                normalized_df = sum_per_group.div(count_per_group.replace(0, np.nan), axis=0).fillna(0)
                percentage_df = (normalized_df * 100).round(1)

                # 5. Аннотации
                annotations = pd.DataFrame(index=all_types, columns=all_chars)
                for row in all_types:
                    for col in all_chars:
                        present = int(sum_per_group.loc[row, col])
                        total = int(count_per_group[row])
                        percent = percentage_df.loc[row, col]
                        annotations.loc[row, col] = f"{present} из {total}<br>({percent}%)"

                # 6. График
                fig = go.Figure(
                    data=go.Heatmap(
                        z=percentage_df.T.values,  # характеристики — по Y
                        x=percentage_df.index,
                        y=percentage_df.columns,
                        text=annotations.T.values,
                        
                        hoverinfo="text",
                        texttemplate="%{text}",
                        colorscale="BuGn",
                        colorbar=dict(title="Процент"),
                        textfont=dict(size=12, color="black"),  # Можно кастомизировать
                    )
                )

                fig.update_layout(
                    xaxis_title="Тип артефакта",
                    yaxis_title="Характеристика",
                    height=800
                )

            else:
                st.error(f"Неподдерживаемый тип агрегации: {agg_mode}")
                continue
            
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{agg_mode}_{group_key}")


def render_correlation_heatmap(df: pd.DataFrame, char_groups: dict):
    st.subheader("📊 Корреляция между характеристиками")

    all_cols = set()
    for group_cols in char_groups.values():
        all_cols.update(group_cols)
    cols = [col for col in all_cols if col in df.columns]

    corr = df[cols].fillna(0).corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Корреляционная матрица"
    )
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)


def render_score_analysis(df: pd.DataFrame):
    st.subheader("🏅 Оценка артефактов (по характеристикам)")

    fig = px.bar(
        df.sort_values("total_score", ascending=False),
        x="name",
        y=["positive_score", "negative_score", "total_score"],
        title="Оценка артефактов по характеристикам (Отсортировано)",
        barmode="group",
        hover_data=["name", "level", "type"],
    )
    fig.update_layout(xaxis_tickangle=45, height=600)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        df,
        x="name",
        y=["total_score"],
        title="Оценка артефактов по характеристикам (по уровням)",
        barmode="group",
        color="level",
        hover_data=["name", "level", "type", "positive_score", "negative_score"],
    )
    fig.update_layout(xaxis_tickangle=45, height=600)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        df,
        x="name",
        y=["total_score"],
        title="Оценка артефактов по характеристикам (по типам)",
        barmode="group",
        color="type",
        hover_data=["name", "level", "type", "positive_score", "negative_score"],
    )
    fig.update_layout(xaxis_tickangle=45, height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 Влияние оценки на рейтинг, стоимость и tier")
    cols = st.columns(3)
    cols[0].plotly_chart(
        px.scatter(
            df, 
            x="total_score", 
            y="af_rank_main", 
            title="score vs af_rank_main", 
            hover_data=["name", "level", "type"],
            color="level",
            ), 
        use_container_width=True)
    cols[1].plotly_chart(px.scatter(df, x="total_score", y="cost_main", title="score vs cost_main", hover_data=["name", "level", "type"], color="level",), use_container_width=True)
    cols[2].plotly_chart(px.scatter(df, x="total_score", y="tier_main", title="score vs tier_main", hover_data=["name", "level", "type"], color="level",), use_container_width=True)


def compute_score_contributions(df: pd.DataFrame, weights: dict, abs_weigths: dict) -> pd.DataFrame:
    score_contributions = []
    for _, row in df.iterrows():
        contributions = {}
        for col, coef in weights.items():
            val = row.get(col, 0)
            if pd.notna(val):
                contributions[col] = val * coef
            else:
                contributions[col] = 0.0  # Обязательно добавляем колонку
        
        # Другая логика для abs_weights
        for col, coef in abs_weigths.items():
            val = row.get(col, 0)
            if pd.notna(val) and val != 0:
                if val > 0:
                    contributions[col] = val * coef
                else:
                    contributions[col] = ((val + 1) * -1) * coef
            else:
                contributions[col] = 0.0  # Обязательно добавляем колонку

        score_contributions.append(contributions)

    score_df = pd.DataFrame(score_contributions)

    # Гарантируем, что все столбцы из weights есть, даже если они пустые
    for col in weights.keys():
        if col not in score_df:
            score_df[col] = 0.0

    for col in abs_weigths.keys():
        if col not in score_df:
            score_df[col] = 0.0

    # Добавим метаинформацию
    score_df["name"] = df.get("name", None)
    score_df["type"] = df.get("type", None)
    score_df["level"] = df.get("level", None)

    # Итоговая сумма
    score_df["total_score"] = score_df[list(weights.keys())].sum(axis=1)

    return score_df


def render_score_analysis_2(df: pd.DataFrame):
    score_df = compute_score_contributions(df, st.session_state.STATS_WEIGHT, st.session_state.STAT_ABS_WEIGHT)

    # print(score_df['level'].unique())

    level_order = ["None", "1", "2", "3", "4", "5", "6", "Базовый", "Модификат", "Мезомодификат", "Гипермодификат", "Абсолют"]

    def transform_level_column(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def transform_level(val):
            val_str = str(val).strip()
            return f"Ур. {val_str}" if val_str.isdigit() else \
                    val_str

        df["level"] = df["level"].apply(transform_level)

        # Получим уникальные уровни и зададим порядок категорий
        level_order = list(dict.fromkeys(df["level"].dropna()))
        df["level"] = pd.Categorical(df["level"], categories=level_order, ordered=True)

        return df, level_order


    score_df_long = score_df.melt(
        id_vars=["name", "type", "level", "total_score"],
        var_name="parameter",
        value_name="score_contribution"
    )

    st.subheader("📊 Вклад характеристик в score артефактов (Отсортировано)")
    fig4 = px.bar(score_df_long.sort_values("total_score", ascending=False), x="name", y="score_contribution", color="parameter",
                  title="Вклад характеристик в итоговый score (Отсортировано)", barmode="stack")
    st.plotly_chart(fig4, use_container_width=True, key="score_bar2")

    st.subheader("📊 Вклад характеристик в score артефактов")
    fig1 = px.bar(score_df_long, x="name", y="score_contribution", color="parameter",
                  title="Вклад характеристик в итоговый score", barmode="stack")
    st.plotly_chart(fig1, use_container_width=True, key="score_bar")

    st.subheader("📦 Распределение score по типам")
    fig2 = px.box(
        score_df,
        x="type",
        y="total_score",
        points="all",
        color="level",  # Цвет по уровню
        title="Score по типу",
        hover_data=["name", "total_score", "level", "type"]
    )
    st.plotly_chart(fig2, use_container_width=True, key="score_box_type")

    score_df, level_order = transform_level_column(score_df)

    st.subheader("📶 Распределение score по уровням")
    fig3 = px.box(
        score_df,
        x="level",
        y="total_score",
        points="all",
        color="type",  # Цвет по типу
        title="Score по уровню",
        category_orders={"level": level_order},
        hover_data=["name", "total_score", "level", "type"]
    )
    st.plotly_chart(fig3, use_container_width=True, key="score_box_level")


def render_stats_count_distribution(df: pd.DataFrame):
    """Интерактивное распределение количества заполненных статов у артефактов"""
    total_stats = len(IMMUNITY_COLS + CAP_COLS + RESTORE_COLS + UTILITY_COLS)

    # Подсчёт заполненных статов
    counts = []
    for _, row in df.iterrows():
        filled = sum(
            1 for col in (IMMUNITY_COLS + CAP_COLS + RESTORE_COLS + UTILITY_COLS)
            if col in row and pd.notna(row[col]) and row[col] != 0
        )
        counts.append(filled)

    df_counts = df.copy()
    df_counts["filled_stats"] = counts

    # --- Базовое распределение ---
    fig = px.histogram(
        df_counts,
        x="filled_stats",
        nbins=total_stats,
        title="📊 Распределение количества заполненных статов у артефактов",
        labels={"filled_stats": "Заполненные статы", "count": "Количество артефактов"},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
    fig.update_xaxes(dtick=1)  # показываем каждое деление
    fig.update_layout(bargap=0.05)

    st.subheader("📊 Распределение количества заполненных статов у артефактов")
    st.plotly_chart(fig, use_container_width=True)

    # --- Распределение по type ---
    fig_type = px.histogram(
        df_counts,
        x="filled_stats",
        color="type",
        nbins=total_stats,
        barmode="overlay",
        title="📊 Заполненные статы по типам артефактов",
        labels={"filled_stats": "Заполненные статы", "count": "Количество артефактов"},
    )
    fig_type.update_traces(opacity=0.7, marker=dict(line=dict(width=0.5, color="black")))
    fig_type.update_xaxes(dtick=1)
    st.subheader("📊 Заполненные статы по типам артефактов")
    st.plotly_chart(fig_type, use_container_width=True)

    # --- Распределение по level ---
    fig_level = px.histogram(
        df_counts,
        x="filled_stats",
        color="level",
        nbins=total_stats,
        barmode="overlay",
        title="📊 Заполненные статы по уровням артефактов",
        labels={"filled_stats": "Заполненные статы", "count": "Количество артефактов"},
    )
    fig_level.update_traces(opacity=0.7, marker=dict(line=dict(width=0.5, color="black")))
    fig_level.update_xaxes(dtick=1)
    st.subheader("📊 Заполненные статы по уровням артефактов")
    st.plotly_chart(fig_level, use_container_width=True)

def render_stats_count_scatter(df: pd.DataFrame):
    """Интерактивный scatter-график: каждый артефакт как точка с числом заполненных статов"""
    total_stats = len(IMMUNITY_COLS + CAP_COLS + RESTORE_COLS + UTILITY_COLS)

    # Подсчёт заполненных статов
    counts = []
    for _, row in df.iterrows():
        filled = sum(
            1 for col in (IMMUNITY_COLS + CAP_COLS + RESTORE_COLS + UTILITY_COLS)
            if col in row and pd.notna(row[col]) and row[col] != 0
        )
        counts.append(filled)

    df_counts = df.copy()
    df_counts["filled_stats"] = counts

    # --- Scatter-график ---
    fig = px.scatter(
        df_counts,
        x="filled_stats",
        y="af_rank_main",  # можно заменить на cost_main или tier_main
        color="type",
        hover_data=["name", "type", "level", "filled_stats", "cost_main", "tier_main"],
        title="🔎 Артефакты и количество заполненных статов",
        labels={
            "filled_stats": "Заполненные статы",
            "af_rank_main": "Ранг артефакта"
        }
    )

    fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color="black")))
    fig.update_xaxes(dtick=1, range=[0, total_stats + 1])
    fig.update_layout(hovermode="closest")

    st.subheader("🔎 Артефакты и количество заполненных статов")
    st.plotly_chart(fig, use_container_width=True)


def render_description_length_distribution(df: pd.DataFrame):
    """Интерактивное распределение длины описаний (RU vs EN)"""
    ru_lengths = df["main_description"].fillna("").astype(str).apply(len)
    en_lengths = df["main_description_eng"].fillna("").astype(str).apply(len)

    df_lengths = pd.DataFrame({
        "artifact": df["name"],
        "RU": ru_lengths,
        "EN": en_lengths,
    })

    # --- Гистограмма RU vs EN ---
    fig = px.histogram(
        df_lengths.melt(id_vars="artifact", value_vars=["RU", "EN"], var_name="lang", value_name="length"),
        x="length",
        color="lang",
        barmode="overlay",
        nbins=40,
        title="📖 Распределение длины описаний (RU vs EN)",
        labels={"length": "Количество символов", "count": "Количество артефактов"},
        color_discrete_map={"RU": "royalblue", "EN": "orangered"}  # яркие контрастные цвета
    )
    fig.update_traces(opacity=0.6, marker=dict(line=dict(width=1, color="black")))
    st.subheader("📖 Распределение длины описаний (RU vs EN)")
    st.plotly_chart(fig, use_container_width=True)

    # --- Сравнение RU vs EN ---
    fig2 = px.scatter(
        df_lengths,
        x="RU",
        y="EN",
        color="RU",  # градиент по длине RU для читаемости
        text="artifact",
        title="📖 Сравнение длины описаний (RU vs EN)",
        labels={"RU": "Длина описания (RU)", "EN": "Длина описания (EN)"},
    )
    fig2.update_traces(
        textposition="top center",
        marker=dict(size=10, opacity=0.7, line=dict(width=0.5, color="black"))
    )
    fig2.update_layout(
        width=900, height=700  # увеличиваем размер для читаемости
    )
    # Добавляем штриховую линию x = y
    fig2.add_shape(
        type="line",
        x0=df_lengths["RU"].min(),
        y0=df_lengths["RU"].min(),
        x1=df_lengths["RU"].max(),
        y1=df_lengths["RU"].max(),
        line=dict(
            color="rgba(128, 128, 128, 0.5)",  # серый с прозрачностью 70%
            width=2,
            dash="dash"
        ),
        layer='below'  # чтобы линия была под точками
    )
    st.subheader("📖 Сравнение длины описаний (RU vs EN)")
    st.plotly_chart(fig2, use_container_width=True)


def render_stats_count_boxplots(df: pd.DataFrame):
    """Графики распределения количества заполненных статов у артефактов с детализацией по type и level"""

    # Подсчёт заполненных статов
    counts = []
    for _, row in df.iterrows():
        filled = sum(
            1 for col in (IMMUNITY_COLS + CAP_COLS + RESTORE_COLS + UTILITY_COLS)
            if col in row and pd.notna(row[col]) and row[col] != 0
        )
        counts.append(filled)

    df_counts = df.copy()
    df_counts["filled_stats"] = counts

    # --- Трансформация уровней ---
    def transform_level_column(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        df = df.copy()

        def transform_level(val):
            val_str = str(val).strip()
            return f"Ур. {val_str}" if val_str.isdigit() else val_str

        df["level"] = df["level"].apply(transform_level)

        # Сохраним порядок появления
        level_order = list(dict.fromkeys(df["level"].dropna()))
        df["level"] = pd.Categorical(df["level"], categories=level_order, ordered=True)

        return df, level_order

    df_counts, level_order = transform_level_column(df_counts)

    # --- График по type (цвет = level) ---
    st.subheader("📦 Количество статов по типам")
    fig_type = px.box(
        df_counts,
        x="type",
        y="filled_stats",
        points="all",
        color="level",
        title="Распределение количества статов по типам",
        hover_data=["name", "level", "type", "filled_stats"]
    )
    st.plotly_chart(fig_type, use_container_width=True, key="stats_box_type")

    # --- График по level (цвет = type) ---
    st.subheader("📶 Количество статов по уровням")
    fig_level = px.box(
        df_counts,
        x="level",
        y="filled_stats",
        points="all",
        color="type",
        title="Распределение количества статов по уровням",
        category_orders={"level": level_order},
        hover_data=["name", "level", "type", "filled_stats"]
    )
    st.plotly_chart(fig_level, use_container_width=True, key="stats_box_level")


def render_all_charts(df: pd.DataFrame, char_groups: dict):
    # 📊 Распределение характеристик по группам
    render_char_heatmap_by_type(df, char_groups, agg_mode="sum")
    render_char_heatmap_by_type(df, char_groups, agg_mode="presence")

    # 📊 Корреляционная матрица характеристик
    render_correlation_heatmap(df, char_groups)

    # Расчёт оценки артефактов
    scored_df = compute_artifact_scores(df, st.session_state.STATS_WEIGHT, st.session_state.STAT_ABS_WEIGHT)
    render_score_analysis(scored_df)
    render_score_analysis_2(df)

    render_stats_count_distribution(df)
    render_stats_count_boxplots(df)
    render_description_length_distribution(df)