import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from hashlib import md5

from itertools import combinations
from datetime import datetime
import json
import time

from graph_utils import render_all_charts

from constants import (
    IMMUNITY_COLS, CAP_COLS, RESTORE_COLS, UTILITY_COLS,
    compute_artifact_scores, get_stats_weights
)
from files_utils import finalize_data, build_artifact_file

st.set_page_config(layout="wide", page_title="🧪 Artifact Balance Manager", page_icon="🧪")

# Добавим CSS для улучшенного интерфейса
st.markdown("""
<style>
    .characteristic-group {
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .positive-value { color: green; font-weight: bold; }
    .negative-value { color: red; font-weight: bold; }
    .neutral-value { color: gray; }
    .stTabs [data-baseweb="tab-list"] button {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 10px 10px 0 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("df.csv")
    df = df.drop(["extra_text_type_eng", "extra_text_level_eng","extra_text_term_eng", 
             "extra_text_type", "extra_text_level", "extra_text_term", 
             "description_new_2", "description_new_eng_2"], axis=1)
    return df

@st.cache_data
def load_balance_notes():
    """Загрузка заметок по балансу"""
    try:
        with open("balance_notes.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"artifacts": {}, "types": {}, "general": ""}

def save_balance_notes(notes):
    """Сохранение заметок по балансу"""
    with open("balance_notes.json", "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

def count_filled_stats(artifact_row, groups):
    """Подсчёт заполненных характеристик для артефакта"""
    all_cols = []
    for g in groups:
        all_cols.extend(g)

    total = len(all_cols)
    filled = sum(
        1 for col in all_cols
        if col in artifact_row and pd.notna(artifact_row[col]) and artifact_row[col] != 0
    )
    return filled, total


def prepare_data_not_cached(raw_df):

    other_cols = [
        col for col in raw_df.columns
        if col not in IMMUNITY_COLS + CAP_COLS + RESTORE_COLS + UTILITY_COLS and raw_df[col].dtype != 'object'
        and col not in ["cost_main", "new_artefact"]
    ]

    # Приведение уровня к строкам и категории
    raw_df["level"] = raw_df["level"].fillna("None").astype(str)
    level_order = ["None", "0", "1", "2", "3", "4", "5", "6", "Абсолют", "Базовый", "Модификат", "Мезомодификат", "Гипермодификат"]
    raw_df["level"] = pd.Categorical(raw_df["level"], categories=level_order, ordered=True)

    # Сохраняем в session_state
    if "df_data" not in st.session_state:
        st.session_state.df_data = raw_df.copy()

    df = st.session_state.df_data

    if 'level_order' not in st.session_state:
        st.session_state.level_order = level_order

    char_groups = {
        "🛡️ Сопротивления (immunity)": IMMUNITY_COLS,
        "📊 Лимиты (cap)": CAP_COLS,
        "♻️ Восстановление (restore)": RESTORE_COLS,
        "🎒 Утилити": UTILITY_COLS,
        "🔧 Прочее": other_cols,
    }

    return df, char_groups


def display_score_contribution_chart(artifact_row, weights, abs_weights):
    """Показать график вклада характеристик в итоговый score артефакта"""

    contributions = []

    # === обычные веса ===
    for col, weight in weights.items():
        val = artifact_row.get(col, 0)
        if pd.isna(val) or val == 0:
            continue
        score = val * weight
        if score != 0:
            contributions.append((col, score))

    # === abs_weights ===
    for col, weight in abs_weights.items():
        val = artifact_row.get(col, 0)
        if pd.isna(val) or val == 0:
            continue
        if val > 0:
            score = val * weight
        else:
            score = ((val + 1) * -1) * weight
        if score != 0:
            contributions.append((col, score))

    if not contributions:
        st.info("Нет характеристик, влияющих на score.")
        return

    contrib_df = pd.DataFrame(contributions, columns=["stat", "score"])
    contrib_df = contrib_df.sort_values("score", ascending=True)  # для красивого порядка

    # Цвета
    colors = ["green" if x > 0 else "red" for x in contrib_df["score"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=contrib_df["score"],
        y=[s.replace("_main", "").replace("_", " ").title() for s in contrib_df["stat"]],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in contrib_df["score"]],
        textposition="auto",
    ))

    total_score = contrib_df["score"].sum()
    fig.add_vline(x=0, line_dash="dot", line_color="black")
    fig.update_layout(
        title=f"Вклад характеристик в итоговый score (Total = {total_score:.2f})",
        xaxis_title="Вклад в score",
        yaxis_title="Характеристика",
        height=200 + len(contrib_df) * 20,
        margin=dict(l=150, r=20, t=50, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

def get_level_order(artifact_type: str):
    """Вернёт порядок уровней для заданного типа артефакта"""
    if artifact_type == "Уникальный":
        return ["1", "2", "3", "4", "Абсолют"]
    elif artifact_type == "Синтетический":
        return ["Базовый", "Модификат", "Мезомодификат", "Гипермодификат", "Абсолют"]
    else:
        return ["1", "2", "3", "4", "5", "6", "Абсолют"]

def transform_level(val):
    val_str = str(val).strip()
    return f"Ур. {val_str}" if val_str.isdigit() else \
            val_str

def transform_level_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["level"] = df["level"].apply(transform_level)

    # Получим уникальные уровни и зададим порядок категорий
    level_order = list(dict.fromkeys(df["level"].dropna()))
    df["level"] = pd.Categorical(df["level"], categories=level_order, ordered=True)

    return df, level_order

def display_artifact_score_comparison(artifact_row, df, weights, abs_weights):
    scored_df = compute_artifact_scores(df, weights, abs_weights)

    artifact_id = artifact_row["artifact_id"]
    art_level = artifact_row["level"]
    art_type = artifact_row["type"]

    current_row = scored_df.loc[scored_df["artifact_id"] == artifact_id].iloc[0]

    # --- сравнение внутри уровня ---
    level_df = scored_df[scored_df["level"] == art_level]

    fig_level = go.Figure()
    for col, color in [("positive_score", "green"),
                       ("negative_score", "red"),
                       ("total_score", "blue")]:
        fig_level.add_trace(go.Box(
            y=level_df[col],
            name=col.replace("_", " ").title(),
            boxmean="sd",
            marker_color=color,
            opacity=0.6
        ))
        fig_level.add_trace(go.Scatter(
            x=[col.replace("_", " ").title()],
            y=[current_row[col]],
            mode="markers",
            marker=dict(color="black", size=14, symbol="star"),
            name=f"{col} (current)"
        ))
    fig_level.update_layout(
        title=f"📊 Сравнение с артефактами уровня {art_level}",
        yaxis_title="Score",
        boxmode="group",
        height=500
    )

    # --- сравнение внутри типа ---
    type_df = scored_df[scored_df["type"] == art_type].copy()

    # применяем твою функцию трансформации уровней
    type_df, level_order = transform_level_column(type_df)

    fig_type = px.scatter(
        type_df,
        x="level",
        y="total_score",
        color="level",
        size="positive_score",
        hover_data=["artifact_id", "positive_score", "negative_score"],
        category_orders={"level": level_order},  # теперь порядок фиксируется функцией
        title=f"📈 Score по уровням для типа {art_type}",
        height=500
    )

    # текущий артефакт
    fig_type.add_trace(go.Scatter(
        x=[transform_level(current_row["level"])],
        y=[current_row["total_score"]],
        mode="markers+text",
        text=["Текущий"],
        textposition="top center",
        marker=dict(color="black", size=16, symbol="star"),
        name="Current Artifact"
    ))

    # тренд по среднему score для каждого уровня
    mean_scores = (
        type_df.groupby("level", observed=True)["total_score"]
        .mean()
        .reindex(level_order)   # порядок берём из transform_level_column
        .dropna()
    )
    fig_type.add_trace(go.Scatter(
        x=mean_scores.index.astype(str),
        y=mean_scores.values,
        mode="lines+markers",
        line=dict(color="blue", dash="dash"),
        name="Средний score"
    ))


    # --- рендер в Streamlit ---
    with st.expander("📈 **Общая статистика по группе и уровню**", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(fig_level, use_container_width=True)
        with col2:
            st.plotly_chart(fig_type, use_container_width=True)

def display_type_stats_distribution(artifact_row, df, char_groups):
    """Показ распределения характеристик по артефактам того же типа."""

    art_type = artifact_row["type"]
    type_df = df[df["type"] == art_type].copy()

    # Все характеристики из групп
    all_stats = [stat for group in char_groups.values() for stat in group]
    
    stats_data = []
    for stat in all_stats:
        if stat not in type_df.columns:
            continue

        # значения по типу
        values = type_df[stat].dropna()
        if (values == 0).all():
            continue  # исключаем полностью пустые статы для типа

        sum_pos = values[values > 0].sum()
        sum_neg = values[values < 0].sum()
        current_val = artifact_row.get(stat, 0)

        stats_data.append({
            "stat": stat,
            "sum_pos": sum_pos,
            "sum_neg": sum_neg,
            "current_val": current_val
        })

    if not stats_data:
        st.info("Для данного типа артефактов нет активных характеристик.")
        return

    # === Строим график ===
    fig = go.Figure()

    # положительные суммы
    fig.add_trace(go.Bar(
        x=[d["stat"] for d in stats_data],
        y=[d["sum_pos"] for d in stats_data],
        name="Сумма +",
        marker_color="green"
    ))

    # отрицательные суммы
    fig.add_trace(go.Bar(
        x=[d["stat"] for d in stats_data],
        y=[d["sum_neg"] for d in stats_data],
        name="Сумма -",
        marker_color="red"
    ))

    # значения текущего артефакта
    fig.add_trace(go.Scatter(
        x=[d["stat"] for d in stats_data],
        y=[d["current_val"] for d in stats_data],
        mode="markers+text",
        text=[f"{d['current_val']:.2f}" for d in stats_data],
        textposition="top center",
        marker=dict(color="black", size=12, symbol="star"),
        name="Текущий артефакт"
    ))

    fig.update_layout(
        barmode="relative",
        title=f"📊 Распределение характеристик по типу ({art_type})",
        xaxis_title="Характеристика",
        yaxis_title="Сумма значений",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


def display_artifact_card(artifact_row, char_groups, df):
    """Отображение карточки артефакта с группировкой характеристик + ранги"""
    col1, col2, col3 = st.columns([2, 2, 2])

    # === Подсчёт score и рангов ===
    scored_df = compute_artifact_scores(df, st.session_state.STATS_WEIGHT, st.session_state.STAT_ABS_WEIGHT)
    artifact_id = artifact_row["artifact_id"]

    pos_score = scored_df.loc[scored_df["artifact_id"] == artifact_id, "positive_score"].iloc[0]
    neg_score = scored_df.loc[scored_df["artifact_id"] == artifact_id, "negative_score"].iloc[0]
    total_score = scored_df.loc[scored_df["artifact_id"] == artifact_id, "total_score"].iloc[0]

    # ранги
    def get_rank(series, aid):
        ranked = series.rank(method="min", ascending=False)
        return int(ranked.loc[scored_df["artifact_id"] == aid].iloc[0]), len(series)

    pos_rank, pos_total = get_rank(scored_df["positive_score"], artifact_id)
    neg_rank, neg_total = get_rank(scored_df["negative_score"], artifact_id)
    total_rank, total_total = get_rank(scored_df["total_score"], artifact_id)

    # === Функция окраски числа ===
    def colored_score(value: float) -> str:
        # нормируем от -100 до 100
        norm = max(-100, min(100, value))
        if norm >= 0:
            # зелёный градиент от серого к зелёному
            intensity = int(155 + (100 * norm / 100))
            color = f"rgb(0,{intensity},0)"
        else:
            # красный градиент от серого к красному
            intensity = int(155 + (100 * abs(norm) / 100))
            color = f"rgb({intensity},0,0)"
        return f"<span style='color:{color}; font-weight:bold'>{value:.2f}</span>"

    with col1:
        st.markdown(f"### {artifact_row['name']}")
        st.write(f"**English name:** {artifact_row['name_eng']}")
        st.write(f"**ID:** {artifact_row['artifact_id']}")
        st.write(f"**Тип:** {artifact_row['type']} (ID: {artifact_row['k_art_subtype']:.0f})")
        st.write(f"**Уровень:** {artifact_row['level']} (ID: {artifact_row['k_art_level']:.0f})")
        # === Добавляем стоимость + место в топе ===
        if "cost_main" in df.columns and pd.notna(artifact_row.get("cost_main", None)):
            costs = df["cost_main"].fillna(0)
            ranks = costs.rank(method="min", ascending=False)  # 1 = самая высокая цена
            rank = int(ranks.loc[artifact_row.name])
            total = len(costs)
            cost_val = int(artifact_row["cost_main"])
            st.write(f"**Стоимость:** {cost_val} ({rank} из {total})")
        else:
            st.write("**Стоимость:** N/A") 
    with col2:
        desc = str(artifact_row.get('main_description', ''))
        desc_eng = str(artifact_row.get('main_description_eng', ''))
        if desc != 'nan':
            st.write(f"📝 **Длина описания (RU):** {len(desc)} символов")
        if desc_eng != 'nan':
            st.write(f"📝 **Длина описания (EN):** {len(desc_eng)} символов")
        if desc != 'nan' and desc_eng != 'nan':
            st.write(f"📝 **RU vs. EN:** {len(desc) - len(desc_eng):+} символов")
        filled, total = count_filled_stats(
            artifact_row, 
            [IMMUNITY_COLS, CAP_COLS, RESTORE_COLS, UTILITY_COLS]
        )
        # Убираем 1 стат (inv_weight), так как он у всех есть
        st.write(f"**Кол-во статов:** {filled-1} из {total-1}")
        st.write("**Новый артефакт? :**", f"{artifact_row['new_artefact']}")
    with col3:
        if artifact_row.get('tier_main'):
            st.metric("Tier", f"{artifact_row['tier_main']:.0f}")
        if artifact_row.get('af_rank_main'):
            st.metric("Rank", f"{artifact_row['af_rank_main']:.0f}")

        # выводим с цветом
        st.markdown(f"**Positive score:** {colored_score(pos_score)} ({pos_rank}/{pos_total})", unsafe_allow_html=True)
        st.markdown(f"**Negative score:** {colored_score(-neg_score)} ({neg_rank}/{neg_total})", unsafe_allow_html=True)  # делаем отрицательным для цвета
        st.markdown(f"**Total score:** {colored_score(total_score)} ({total_rank}/{total_total})", unsafe_allow_html=True)

    st.markdown("#### 🧮 Вклад характеристик в итоговый score")
    display_score_contribution_chart(artifact_row, st.session_state.STATS_WEIGHT, st.session_state.STAT_ABS_WEIGHT)
    display_artifact_score_comparison(artifact_row, df, st.session_state.STATS_WEIGHT, st.session_state.STAT_ABS_WEIGHT)
    
    # Характеристики
    st.markdown("#### 📊 Характеристики:")

    weights_sign = {key: 1 if value >= 0 else -1 for key, value in st.session_state.STATS_WEIGHT.items()}
    
    cols_per_row = 3
    for group_name, group_cols in char_groups.items():
        active_chars = []
        for col in group_cols:
            if col in artifact_row.index:
                val = artifact_row[col]
                if pd.notna(val) and val != 0:
                    active_chars.append((col, val))
        
        if active_chars:
            with st.expander(f"{group_name} ({len(active_chars)} активных)", expanded=True):
                cols = st.columns(cols_per_row)
                for idx, (char, val) in enumerate(active_chars):
                    col_idx = idx % cols_per_row
                    with cols[col_idx]:
                        if val * weights_sign.get(char, 1)  > 0:
                            color_class = "positive-value"
                            symbol = "+" if val > 0 else ""
                        elif val * weights_sign.get(char, 1) < 0:
                            color_class = "negative-value"
                            symbol = "" if val < 0 else "+"
                        else:
                            color_class = "neutral-value"
                            symbol = ""

                        # Ранг по конкретной характеристике
                        series = df[char].dropna()
                        series = series[series != 0]
                        total = len(series)
                        if total > 0 and artifact_row.name in series.index:
                            rank = series.rank(method="min", ascending=False)
                            pos = int(rank.loc[artifact_row.name])
                            rank_str = f" ({pos}/{total})"
                        else:
                            rank_str = ""

                        char_display = char.replace("_main", "").replace("_", " ").title()
                        
                        st.markdown(f"""
                        <div style='padding: 5px; border-radius: 5px; margin: 2px;'>
                            <small>{char_display}</small><br>
                            <span class='{color_class}'>{symbol}{val:.3f}</span>
                            <small>{rank_str}</small>
                        </div>
                        """, unsafe_allow_html=True)


def artifact_editor_tab(df, char_groups):
    """Вкладка редактора артефактов"""
    st.header("📝 Редактор артефактов")
    
    # Фильтры
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_type = st.selectbox("🎯 Тип артефакта", options=[None] + sorted(df['type'].dropna().unique().tolist()), key="editor_type")
        st.session_state.selected_type = selected_type
    with col2:
        selected_level = st.selectbox("📊 Уровень", options=[None] + sorted(df['level'].dropna().unique().tolist()), key="editor_level")
        st.session_state.selected_level = selected_level
    with col3:
        search_name = st.text_input("🔍 Поиск по названию", key="editor_search")
    
    # Применяем фильтры
    filtered = df.copy()
    if selected_type:
        filtered = filtered[filtered["type"] == selected_type]
    if selected_level:
        filtered = filtered[filtered["level"] == selected_level]
    if search_name:
        filtered = filtered[filtered["name"].str.contains(search_name, case=False, na=False)]
    
    if len(filtered) == 0:
        st.warning("Артефакты не найдены")
        return
    
    # Выбор артефакта для редактирования
    artifact_options = [
        f"{row['artifact_id']} | {row['name']} | {row['type']} | {row['level']}"
        for _, row in filtered.iterrows()
    ]
    selected_option = st.selectbox("🎮 Выберите артефакт для редактирования", artifact_options, key="artifact_selector")

    if selected_option:
        # ИСПРАВЛЕНИЕ: Получаем индекс в отфильтрованном DataFrame
        filtered_idx = artifact_options.index(selected_option)
        artifact_row = filtered.iloc[filtered_idx]
        
        # ВАЖНО: Используем artifact_id для поиска в исходном DataFrame
        artifact_id = artifact_row['artifact_id']
        # Находим реальный индекс в исходном DataFrame по artifact_id
        original_idx = df[df['artifact_id'] == artifact_id].index[0]
        
        selected_art = artifact_row['name']

        # Отображаем карточку артефакта
        display_artifact_card(artifact_row, char_groups, df)

        # Редактирование характеристик
        st.markdown("### ✏️ Редактирование характеристик")

        # ИСПРАВЛЕНИЕ: Используем artifact_id в ключе формы для уникальности
        form_key = f"artifact_edit_form_{artifact_id}"
        
        with st.form(form_key):
            edited_values = {}
            
            for group_name, group_cols in char_groups.items():
                group_key = md5(group_name.encode()).hexdigest()[:8]
                with st.expander(group_name, expanded=False):
                    cols = st.columns(2)
                    for idx, col in enumerate(group_cols):
                        if col in df.columns:
                            col_idx = idx % 2
                            with cols[col_idx]:
                                current_val = artifact_row[col] if pd.notna(artifact_row[col]) else None
                                char_display = col.replace("_main", "").replace("_", " ").title()
                                # ИСПРАВЛЕНИЕ: Используем artifact_id в ключе
                                input_key = f"form_{artifact_id}_{col}_{group_key}"

                                new_val = st.number_input(
                                    char_display,
                                    value=float(current_val) if current_val is not None else 0.0,
                                    format="%.3f",
                                    key=input_key,
                                    step=1.0
                                )

                                if new_val == 0.0 and current_val is None:
                                    edited_values[col] = None
                                else:
                                    edited_values[col] = new_val

            # Мета-поля
            meta_fields = {
                "name": "Название (RU)",
                "name_eng": "Название (EN)",
                "type": "Тип",
                "level": "Уровень",
                "main_description": "Описание (RU)",
                "main_description_eng": "Описание (EN)",
                "cost_main": "Стоимость",
            }

            with st.expander("📁 Основные поля", expanded=False):
                edited_values["name"] = st.text_input(
                    meta_fields["name"],
                    value=artifact_row.get("name", ""),
                    key=f"meta_name_{artifact_id}"  # ИСПРАВЛЕНИЕ
                )
                edited_values["name_eng"] = st.text_input(
                    meta_fields["name_eng"],
                    value=artifact_row.get("name_eng", ""),
                    key=f"meta_name_eng_{artifact_id}"  # ИСПРАВЛЕНИЕ
                )

                current_val = artifact_row.get("cost_main", None)
                edited_values["cost_main"] = st.number_input(
                    meta_fields["cost_main"],
                    value=float(current_val) if current_val is not None else 0.0,
                    format="%.3f",
                    key=f"meta_cost_main_{artifact_id}",  # ИСПРАВЛЕНИЕ
                    step=1.0
                )

                type_options = sorted(df["type"].dropna().unique().tolist())
                level_options = sorted(df["level"].dropna().unique().tolist())

                edited_values["type"] = st.selectbox(
                    meta_fields["type"],
                    options=type_options,
                    index=type_options.index(artifact_row["type"]) if artifact_row["type"] in type_options else 0,
                    key=f"meta_type_{artifact_id}"  # ИСПРАВЛЕНИЕ
                )
                edited_values["level"] = st.selectbox(
                    meta_fields["level"],
                    options=level_options,
                    index=level_options.index(artifact_row["level"]) if artifact_row["level"] in level_options else 0,
                    key=f"meta_level_{artifact_id}"  # ИСПРАВЛЕНИЕ
                )

                edited_values["main_description"] = st.text_area(
                    meta_fields["main_description"],
                    value=str(artifact_row.get("main_description", "")),
                    height=150,
                    key=f"meta_desc_ru_{artifact_id}"  # ИСПРАВЛЕНИЕ
                )
                edited_values["main_description_eng"] = st.text_area(
                    meta_fields["main_description_eng"],
                    value=str(artifact_row.get("main_description_eng", "")),
                    height=150,
                    key=f"meta_desc_en_{artifact_id}"  # ИСПРАВЛЕНИЕ
                )

            submitted = st.form_submit_button("💾 Сохранить изменения")
            if submitted:
                # ИСПРАВЛЕНИЕ: Используем реальный индекс в исходном DataFrame
                for col, val in edited_values.items():
                    st.session_state.df_data.loc[original_idx, col] = val
                st.success(f"✅ Артефакт '{selected_art}' обновлен!")
                st.rerun()

    # === Дополнительное окно: редактирование таблицей ===
    st.markdown("## 📋 Редактируемая таблица артефактов")
    st.info("Изменения сохраняются только по кнопке «Сохранить изменения». Можно редактировать сразу несколько строк.")

    not_number_cols = [
        "artifact_id", "artifact", "name", "type", "level", "new_artefact",
        "description_old", "main_description", "extra_text",
        "name_eng", "main_description_eng", "extra_text_term", "extra_text_eng",
        "extra_text_term_eng", "description_new", "description_new_eng"
    ]

    # Фиксируем отфильтрованный df в session_state, чтобы сохранялись изменения
    st.session_state.filtered_df = filtered.copy()

    with st.form("artifact_table_edit_form", clear_on_submit=False):
        edited_df = st.data_editor(
            st.session_state.filtered_df.reset_index(drop=True),
            use_container_width=True,
            num_rows="dynamic",
            disabled=["artifact_id", "artifact", "name", "new_artefact"],
            column_config={
                col: st.column_config.NumberColumn(col, format="%.3f")
                for col in st.session_state.filtered_df.columns
                if col not in not_number_cols
            },
            hide_index=True,
            key="artifact_table_editor",
        )

        save_table = st.form_submit_button("💾 Сохранить изменения (таблица)")

    if save_table:
        # ИСПРАВЛЕНИЕ: Убедимся, что у нас есть artifact_id для корректного обновления
        if "artifact_id" not in edited_df.columns:
            st.error("Ошибка: отсутствует колонка artifact_id")
            return
            
        original = st.session_state.df_data.set_index("artifact_id")
        updated = edited_df.set_index("artifact_id")

        for aid in updated.index.intersection(original.index):
            diffs = updated.loc[aid] != original.loc[aid]
            if diffs.any():
                for col in diffs.index[diffs]:
                    # Обновляем по artifact_id, а не по позиционному индексу
                    st.session_state.df_data.loc[
                        st.session_state.df_data["artifact_id"] == aid, col
                    ] = updated.loc[aid, col]

        st.success("Изменения сохранены! 🎉")
        # Обновляем фильтрованный df
        df = st.session_state.df_data
        st.session_state.filtered_df = df.copy()
        st.rerun()


def balance_notes_tab(df):
    """Вкладка для заметок по балансу"""
    st.header("📔 Заметки по балансу")
    
    notes = load_balance_notes()
    
    tab1, tab2, tab3 = st.tabs(["🎯 По артефактам", "📦 По типам", "📋 Общие принципы"])
    
    with tab1:
        st.subheader("Заметки по конкретным артефактам")
        
        artifact_name = st.selectbox("Выберите артефакт", df['name'].tolist())
        current_note = notes["artifacts"].get(artifact_name, "")
        
        new_note = st.text_area(
            f"Заметки для {artifact_name}",
            value=current_note,
            height=150,
            key=f"note_artifact_{artifact_name}"
        )
        
        if st.button("💾 Сохранить заметку", key="save_artifact_note"):
            notes["artifacts"][artifact_name] = new_note
            save_balance_notes(notes)
            st.success("Заметка сохранена!")
    
    with tab2:
        st.subheader("Заметки по типам артефактов")
        
        artifact_type = st.selectbox("Выберите тип", df['type'].dropna().unique().tolist())
        current_note = notes["types"].get(artifact_type, "")
        
        new_note = st.text_area(
            f"Заметки для типа {artifact_type}",
            value=current_note,
            height=150,
            key=f"note_type_{artifact_type}"
        )
        
        if st.button("💾 Сохранить заметку", key="save_type_note"):
            notes["types"][artifact_type] = new_note
            save_balance_notes(notes)
            st.success("Заметка сохранена!")
    
    with tab3:
        st.subheader("Общие принципы баланса")
        
        # principles = """
        # ### Основные принципы баланса:
        
        # 1. **Специализация vs Универсальность** 
        #    - Узкоспециализированные артефакты должны быть сильнее в своей нише
        #    - Универсальные артефакты слабее, но гибче
        
        # 2. **Risk/Reward** 
        #    - Сильные положительные эффекты компенсируются негативными
        #    - Чем выше риск, тем выше награда
        
        # 3. **Синергия типов** 
        #    - Определенные типы должны дополнять друг друга
        #    - Избегать "must-have" комбинаций
        
        # 4. **Прогрессия уровней** 
        #    - Высокоуровневые артефакты заметно лучше
        #    - Каждый уровень должен быть значимым апгрейдом
        
        # 5. **Альтернативные билды** 
        #    - Несколько жизнеспособных стратегий сборки
        #    - Нет единственно правильного решения
        # """
        
        # st.markdown(principles)
        
        general_notes = st.text_area(
            "Ваши дополнительные заметки по балансу",
            value=notes.get("general", ""),
            height=200
        )
        
        if st.button("💾 Сохранить общие заметки", key="save_general_notes"):
            notes["general"] = general_notes
            save_balance_notes(notes)
            st.success("Заметки сохранены!")

def combination_optimizer_tab(df):
    """Вкладка оптимизатора комбинаций"""
    st.header("🎯 Генератор оптимальных комбинаций")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Фильтры для выбора доступных артефактов
        st.subheader("Выберите доступные артефакты")
        
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            filter_types = st.multiselect("Типы", df['type'].dropna().unique().tolist())
        with filter_col2:
            filter_levels = st.multiselect("Уровни", df['level'].dropna().unique().tolist())
        
        # Применяем фильтры
        available_df = df.copy()
        if filter_types:
            available_df = available_df[available_df['type'].isin(filter_types)]
        if filter_levels:
            available_df = available_df[available_df['level'].isin(filter_levels)]
        
        # Мультиселект для выбора конкретных артефактов
        selected_artifacts = st.multiselect(
            "Или выберите конкретные артефакты",
            available_df['name'].tolist(),
            help="Оставьте пустым для использования всех отфильтрованных"
        )
        
        if selected_artifacts:
            available_df = available_df[available_df['name'].isin(selected_artifacts)]
    
    with col2:
        st.subheader("Параметры оптимизации")
        
        num_artifacts = st.slider("Количество артефактов в комбинации", 1, 5, 3)
        
        optimization_goal = st.selectbox(
            "Цель оптимизации",
            ["Максимальная выживаемость", "Баланс", "Минимальные штрафы", "Кастомная"]
        )
        
        if optimization_goal == "Кастомная":
            st.info("Выберите важные характеристики ниже")

        max_combinations = st.number_input(
            "Максимальное число комбинаций для перебора",
            min_value=100,
            max_value=1_000_000_000,
            value=10000,
            step=1000
        )

    if st.button("🚀 Найти оптимальные комбинации", type="primary"):
        if len(available_df) < num_artifacts:
            st.error(f"Недостаточно артефактов! Доступно: {len(available_df)}, требуется: {num_artifacts}")
        else:
            find_optimal_combinations(available_df, num_artifacts, optimization_goal, max_combinations=max_combinations)

def find_optimal_combinations(df, num_artifacts, goal, max_combinations=10000):
    """Поиск оптимальных комбинаций артефактов с учётом weights и abs_weights"""
    
    # Определяем веса для разных целей
    goal_weights = {
        "Максимальная выживаемость": {
            'bleeding_restore_speed_main': 2.0,
            "bleeding_restore_speed_hard_main": 2.0,
            'health_restore_speed_main':    2.0,
            'power_restore_speed_main':     0.25,
            'satiety_restore_speed_main':   1.5,
            "eat_thirstiness":              1.5,
            "eat_sleepiness":               3,
            'radiation_restore_speed_main': -3.0,  # знак в значении будет учтён
            "psy_health_restore_speed":     3,
        },
        # "Баланс": {
        #     # Используем равные веса для всех положительных характеристик
        # },
        # "Минимальные штрафы": {
        #     # Минимизируем негативные эффекты
        # }
    }

    weights_sign = {key: 1 if value >= 0 else -1 for key, value in st.session_state.STATS_WEIGHT.items()}

    # Получаем числовые колонки
    # Числовые колонки
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in ['artifact_id', 'cost_main', 'tier_main', 'af_rank_main']
    ]

    # Преобразуем в numpy для ускорения
    numeric_data = df[numeric_cols].to_numpy(dtype=float)
    artifact_names = df['name'].to_numpy()
    artifact_types = df['type'].to_numpy() if "type" in df.columns else np.array(["—"] * len(df))
    artifact_levels = df['level'].to_numpy() if "level" in df.columns else np.array(["—"] * len(df))


    artifact_indices = df.index.tolist()
    total_combinations = np.math.comb(len(artifact_indices), num_artifacts)

    # Если слишком много — случайная подвыборка
    if total_combinations > max_combinations:
        st.warning(f"Слишком много комбинаций ({total_combinations}). Анализируем случайные {max_combinations}")
        all_indices = np.arange(len(df))
        sampled_combos = [tuple(np.random.choice(all_indices, num_artifacts, replace=False)) for _ in range(max_combinations)]
    else:
        sampled_combos = list(combinations(range(len(df)), num_artifacts))

    results = []
    progress_bar = st.progress(0)

    for idx, combo in enumerate(sampled_combos, 1):
        combo_array = numeric_data[list(combo), :]
        combo_stats_arr = np.nansum(combo_array, axis=0)
        combo_stats = dict(zip(numeric_cols, combo_stats_arr))

        combo_names = artifact_names[list(combo)].tolist()

        # Подсчёт score
        score = 0.0

        # Обычные weights
        for col, weight in st.session_state.STATS_WEIGHT.items():
            val = combo_stats.get(col, 0)
            if pd.notna(val):
                score += val * weight

        # abs_weights логика
        for col, weight in st.session_state.STAT_ABS_WEIGHT.items():
            val = combo_stats.get(col, 0)
            if pd.isna(val):
                continue
            if val > 0:
                score += val * weight
            else:
                score += ((val + 1) * -1) * weight

        # Цель
        if goal == "Максимальная выживаемость":
            score += sum(combo_stats.get(col, 0) * weight
                         for col, weight in goal_weights["Максимальная выживаемость"].items())
        elif goal == "Минимальные штрафы":
            score += -sum(val for val in combo_stats.values() if val < 0)
        else:  # Баланс
            positive = sum(val for val in combo_stats.values() if val > 0)
            negative = sum(abs(val) for val in combo_stats.values() if val < 0)
            score += positive - negative * 0.5

        results.append({
            'combo': combo_names,
            'score': score,
            'stats': combo_stats
        })

        progress_bar.progress(idx / max_combinations if total_combinations > max_combinations else idx / total_combinations)

    progress_bar.empty()

    results.sort(key=lambda x: x['score'], reverse=True)

    st.subheader(f"🏆 Топ-10 комбинаций для цели: {goal}")
    for i, result in enumerate(results[:10], 1):
        with st.expander(f"#{i} Комбинация (счет: {result['score']:.2f})"):
            st.write("**Артефакты:**")
            for artifact_name in result['combo']:
                idx = np.where(artifact_names == artifact_name)[0][0]
                artifact_type = artifact_types[idx]
                artifact_level = artifact_levels[idx]
                st.markdown(f"**{artifact_name}** _({artifact_type} | {artifact_level})_")

            st.write("**Ключевые характеристики:**")
            sorted_stats = sorted(result['stats'].items(), key=lambda x: abs(x[1]), reverse=True)
            positive_stats = [(k, v) for k, v in sorted_stats if v * weights_sign.get(k, 1) > 0]
            negative_stats = [(k, v) for k, v in sorted_stats if v * weights_sign.get(k, 1) < 0]

            col1, col2 = st.columns(2)
            with col1:
                st.write("✅ **Положительные:**")
                for stat, val in positive_stats:
                    st.write(f"- {stat}: +{val:.3f}")
            with col2:
                st.write("❌ **Отрицательные:**")
                for stat, val in negative_stats:
                    st.write(f"- {stat}: {val:.3f}")

def balance_analyzer_tab(df: pd.DataFrame):
    """Вкладка анализатора проблемных зон баланса с учётом весов"""
    st.header("🔍 Анализатор проблемных зон баланса")

    # --- Считаем правильные score ---
    scored_df = compute_artifact_scores(df, st.session_state.STATS_WEIGHT, st.session_state.STAT_ABS_WEIGHT)

    # Подготовим дополнительные данные
    ignore_cols = ['artifact_id', 'cost_main', 'jump_height_main', 'tier_main', 'af_rank_main']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ignore_cols]

    # --------------------------
    # Анализ 1: Баланс по типам
    # --------------------------
    st.subheader("⚖️ Баланс между типами артефактов")

    type_stats = (
        scored_df.groupby("type")
        .agg(
            Количество=("artifact_id", "count"),
            Средние_плюсы=("positive_score", "mean"),
            Средние_минусы=("negative_score", "mean"),
            Баланс=("total_score", "mean"),
        )
        .reset_index()
    )

    # "Длинный" формат
    plot_df = type_stats.melt(
        id_vars=["type", "Количество"],
        value_vars=["Средние_плюсы", "Средние_минусы", "Баланс"],
        var_name="Метрика",
        value_name="Значение"
    )

    # Более мягкая палитра для тёмной темы
    color_map = {
        "Средние_плюсы": "#7fc97f",  # мягкий зелёный
        "Средние_минусы": "#f87c7c",  # мягкий красный
        "Баланс": "#7c9df8"          # мягкий синий
    }

    fig = px.bar(
        plot_df,
        x="type",
        y="Значение",
        color="Метрика",
        barmode="group",
        color_discrete_map=color_map,
        hover_data=["Количество"]
    )

    fig.update_layout(
        title="Сравнение плюсов, минусов и баланса по типам артефактов",
        xaxis_title="Тип артефакта",
        yaxis_title="Значение",
        legend_title="Метрика",
        bargap=0.25,
        height=500
    )

    # Подписи под 90 градусов
    fig.update_xaxes(tickangle=-90)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(type_stats, use_container_width=True)

    # --------------------------
    # Анализ 2: Прогрессия по уровням
    # --------------------------
    st.subheader("📈 Анализ прогрессии по уровням")

    level_analysis = (
        scored_df.groupby("level")
        .agg(
            Количество=("artifact_id", "count"),
            Средняя_сила=("total_score", "mean"),
            Средняя_стоимость=("cost_main", "mean"),
        )
        .reset_index()
    )

    # Сортировка уровней
    level_order = [
        "None", "0", "1", "2", "3", "4", "5", "6",
        "Базовый", "Модификат", "Мезомодификат", "Гипермодификат", "Абсолют"
    ]
    level_analysis["Уровень"] = pd.Categorical(level_analysis["level"], categories=level_order, ordered=True)
    level_analysis = level_analysis.sort_values("Уровень")
    level_analysis["x"] = level_analysis["Уровень"].cat.codes

    plot_df = scored_df.copy()
    plot_df["Уровень"] = pd.Categorical(plot_df["level"], categories=level_order, ordered=True)
    plot_df["x"] = plot_df["Уровень"].cat.codes

    # Джиттер для точек
    np.random.seed(42)
    plot_df["x_jitter"] = plot_df["x"] + np.random.uniform(-0.2, 0.2, size=len(plot_df))

    # --- График точек ---
    fig = px.scatter(
        plot_df,
        x="x_jitter",
        y="total_score",
        color="level",
        hover_data=["artifact_id", "name", "level", "total_score", "cost_main"],
        opacity=0.7
    )

    # Добавляем линию среднего
    fig.add_trace(go.Scatter(
        x=level_analysis["x"],
        y=level_analysis["Средняя_сила"],
        mode="lines+markers",
        name="Средняя сила",
        line=dict(color="#FC5F5F", width=2),
        marker=dict(size=10, color="#FECE74"),
        hovertemplate="Уровень: %{x}<br>Средняя сила: %{y:.2f}<extra></extra>"
    ))

    # Настройка оси X (чтобы показывались уровни, а не числа)
    fig.update_xaxes(
        tickvals=list(range(len(level_order))),
        ticktext=level_order
    )

    fig.update_layout(
        title="Прогрессия силы артефактов по уровням",
        xaxis_title="Уровень",
        yaxis_title="Сила (total_score)",
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # Анализ 3: Редко используемые характеристики
    # --------------------------
    st.subheader("❓ Редко используемые характеристики")

    with st.expander("Таблица", expanded=False):
        min_percent = 10
        char_usage = []
        for col in numeric_cols:
            non_zero = (df[col].notna() & (df[col] != 0)).sum()
            usage_percent = (non_zero / len(df)) * 100

            if usage_percent < min_percent:
                char_usage.append({
                    'Характеристика': col,
                    'Использование %': usage_percent,
                    'Артефактов': non_zero
                })

        if char_usage:
            st.text(f"Статы, которые имеются у менее {min_percent}% артефактов.")
            char_usage_df = pd.DataFrame(char_usage).sort_values('Использование %')
            st.dataframe(char_usage_df, use_container_width=True)
        else:
            st.info("Все характеристики достаточно используются")

    # --------------------------
    # Анализ 4: Выбросы по характеристикам
    # --------------------------
    st.subheader("📊 Артефакты-выбросы")
    with st.expander("Таблица", expanded=False):
        outliers = []
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                mean = df[col].mean()
                std = df[col].std()

                if std > 0:
                    for idx, row in df.iterrows():
                        val = row[col]
                        if pd.notna(val) and val != 0:
                            z_score = abs((val - mean) / std)
                            if z_score > 2:
                                outliers.append({
                                    'Артефакт': row['name'],
                                    'Характеристика': col,
                                    'Значение': val,
                                    'Z-score': z_score,
                                    'Тип': row['type'],
                                    'Уровень': row['level']
                                })

        if outliers:
            outliers_df = pd.DataFrame(outliers).sort_values('Z-score', ascending=False)
            st.text(f"Кол-во выбросов: {len(outliers_df)} из {len(df)} ({len(outliers_df) / len(df):.2%})")
            st.dataframe(outliers_df.head(50), use_container_width=True)
        else:
            st.info("Выбросов не обнаружено")

    # --- Улучшенный анализ: Распределение значений по выбранным характеристикам (с учётом NaN и флагом) ---
    st.subheader("🧩 Анализ распределения характеристик (улучшенный)")

    # Объединяем все нужные категории характеристик
    all_char_cols = IMMUNITY_COLS + CAP_COLS + RESTORE_COLS + UTILITY_COLS
    available_cols = [col for col in all_char_cols if col in df.columns]

    if not available_cols:
        st.warning("Нет доступных характеристик из указанных категорий.")
    else:
        selected_cols = st.multiselect(
            "Выберите характеристики для анализа:",
            options=available_cols,
            default=available_cols[:3] if len(available_cols) >= 3 else available_cols
        )

        # Флаг: показывать только ненулевые и не-NaN
        show_only_nonzero = st.checkbox(
            "Показывать только предметы с ненулевыми значениями (игнорировать 0 и NaN)",
            value=True,
            help="Если снять — будут показаны все предметы, включая со значением 0 или NaN (они будут серыми или помечены)."
        )
        
        # --- Переименование уровней: "1" → "1 ур.", "2" → "2 ур." и т.д. ---
        level_mapping = {
            "1": "1 ур.",
            "2": "2 ур.",
            "3": "3 ур.",
            "4": "4 ур.",
            "5": "5 ур.",
            "6": "6 ур."
        }

        if selected_cols:
            # Создаем копию для безопасной модификации
            working_df = df.copy()
            working_df['level'] = working_df['level'].replace(level_mapping)

            # Применяем фильтр в зависимости от флага
            if show_only_nonzero:
                # Фильтруем: не-NaN И не 0 хотя бы в одной выбранной колонке
                mask = working_df[selected_cols].notna().any(axis=1) & working_df[selected_cols].ne(0).any(axis=1)
                filtered_df = working_df[mask].copy()
                filter_note = " (только ненулевые и не-NaN)"
            else:
                # Показываем всё, но добавим столбец-индикатор
                filtered_df = working_df.copy()
                # Добавим вспомогательный столбец для визуального различия
                for col in selected_cols:
                    filtered_df[f"{col}_valid"] = filtered_df[col].notna() & (filtered_df[col] != 0)
                filter_note = " (все предметы, серым — 0 или NaN)"

            if filtered_df.empty:
                st.info(f"Нет артефактов{filter_note} в выбранных характеристиках.")
            else:
                # Добавляем столбцы для отображения
                display_cols = ['name', 'type', 'level'] + selected_cols
                if not show_only_nonzero:
                    # Добавляем индикаторы валидности
                    display_cols += [f"{col}_valid" for col in selected_cols]
                display_df = filtered_df[display_cols].copy()

                # Выбор столбца для сортировки
                sort_by = st.selectbox(
                    "Сортировать по:",
                    options=selected_cols,
                    index=0
                )
                sort_order = st.radio("Порядок сортировки:", ["По возрастанию", "По убыванию"], horizontal=True)
                ascending = sort_order == "По возрастанию"

                # Сортируем, но NaN/0 будут в конце при сортировке по возрастанию (если не фильтруем)
                display_df = display_df.sort_values(by=sort_by, ascending=ascending, na_position='last')

                st.write(f"### 📊 Распределение по характеристике: `{sort_by}`{filter_note}")
                st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

                # === Дополнительное окно: редактирование таблицей ===
                st.markdown("## 📋 Редактируемая таблица артефактов")
                st.info("Изменения сохраняются только по кнопке «Сохранить изменения». Можно редактировать сразу несколько строк.")

                # Информационные столбцы — показываем, но запрещаем редактирование
                info_cols = [
                    "artifact_id", "name", "type", "level", "main_description",
                    # "artifact", "new_artefact",
                    # "description_old", "main_description", "extra_text",
                    # "name_eng", "main_description_eng", "extra_text_term", "extra_text_eng",
                    # "extra_text_term_eng", "description_new", "description_new_eng"
                ]

                # Определяем столбцы для отображения:
                # 1. Информационные (все, что есть в filtered_df)
                # 2. Выбранные характеристики (selected_cols)
                editor_cols = []
                # Сначала добавляем информационные, если они есть
                for col in info_cols:
                    if col in filtered_df.columns and col not in editor_cols:
                        editor_cols.append(col)
                # Затем добавляем характеристики
                for col in selected_cols:
                    if col not in editor_cols:
                        editor_cols.append(col)

                if "artifact_id" not in filtered_df.columns:
                    st.error("❌ Ошибка: в данных отсутствует колонка 'artifact_id'. Редактирование невозможно.")
                else:
                    # Сортируем так же, как display_df
                    edit_view_df = filtered_df[editor_cols].copy()
                    if sort_by in editor_cols:
                        edit_view_df = edit_view_df.sort_values(by=sort_by, ascending=ascending, na_position='last')
                    else:
                        # Сохраняем порядок из display_df
                        edit_view_df = edit_view_df.reindex(display_df.index).reset_index(drop=True)

                    # Сохраняем в session_state
                    st.session_state.filtered_df_edit = edit_view_df.copy()

                    with st.form("artifact_table_edit_form_balance", clear_on_submit=False):
                        edited_df = st.data_editor(
                            st.session_state.filtered_df_edit.reset_index(drop=True),
                            use_container_width=True,
                            num_rows="dynamic",
                            disabled=info_cols,  # ← ЗАПРЕЩАЕМ редактирование информационных столбцов
                            column_config={
                                col: st.column_config.NumberColumn(col, format="%.3f")
                                for col in selected_cols  # ← Только для характеристик включаем NumberColumn
                                if col in edit_view_df.select_dtypes(include=[np.number]).columns
                            },
                            hide_index=True,
                            key="artifact_table_editor_balance",
                        )

                        save_table = st.form_submit_button("💾 Сохранить изменения (таблица)")

                    if save_table:
                        if "artifact_id" not in edited_df.columns:
                            st.error("❌ Ошибка: отсутствует колонка artifact_id")
                        elif 'df_data' not in st.session_state:
                            st.error("❌ Ошибка: исходные данные не найдены в session_state")
                        else:
                            # Определяем, какие столбцы можно сохранять — ТОЛЬКО selected_cols (и artifact_id для связи)
                            safe_editable_cols = [col for col in selected_cols if col in edited_df.columns]
                            safe_cols = ["artifact_id"] + safe_editable_cols

                            # Обрезаем edited_df до безопасных столбцов
                            edited_df = edited_df[safe_cols]

                            original_full = st.session_state.df_data.set_index("artifact_id")
                            edited_df = edited_df.set_index("artifact_id")

                            changes_made = False
                            for aid in edited_df.index:
                                if aid in original_full.index:
                                    common_cols = edited_df.columns.intersection(original_full.columns)
                                    if len(common_cols) == 0:
                                        continue

                                    edited_row = edited_df.loc[aid, common_cols]
                                    original_row = original_full.loc[aid, common_cols]

                                    # Выравниваем порядок
                                    edited_row = edited_row.reindex(common_cols)
                                    original_row = original_row.reindex(common_cols)

                                    # Сравниваем с учётом NaN
                                    diff_mask = ~((edited_row == original_row) | (edited_row.isna() & original_row.isna()))
                                    changed_cols = diff_mask[diff_mask].index.tolist()

                                    if changed_cols:
                                        changes_made = True
                                        for col in changed_cols:
                                            st.session_state.df_data.loc[
                                                st.session_state.df_data["artifact_id"] == aid, col
                                            ] = edited_row[col]

                            if changes_made:
                                st.success("✅ Изменения сохранены! Графики и статистика обновлены.")
                                df = st.session_state.df_data.copy()
                                st.rerun()
                            else:
                                st.info("ℹ️ Изменений не обнаружено.")

                # --- Графики распределения значений ---
                for col in selected_cols:
                    st.write(f"#### 📈 Значения характеристики: `{col}`{filter_note}")

                    # Для графика — всегда работаем с копией, где level уже переименован
                    plot_df = filtered_df.copy()

                    if show_only_nonzero:
                        # Фильтруем только не-NaN
                        plot_df = plot_df[plot_df[col].notna()].copy()
                    else:
                        # Оставляем всё, но будем визуально выделять невалидные
                        pass

                    if len(plot_df) == 0:
                        st.warning(f"Нет данных для графика `{col}`.")
                        continue

                    # График 1: По уровням
                    
                    # if not show_only_nonzero:
                    #     # Цвет по валидности
                    #     color_col = f"{col}_valid"
                    #     color_map = {True: 'blue', False: 'lightgrey'}
                    #     title_suffix = " (синие — валидные, серые — 0/NaN)"
                    # else:
                    color_col = "type"
                    color_map = None
                    title_suffix = ""

                    # --- Расширенная статистика по колонке (только по не-NaN значениям) ---
                    valid_mask = plot_df[col].notna()
                    valid_count = valid_mask.sum()
                    total_items = len(df)  # общее количество предметов в исходном датасете

                    non_zero_and_valid_mask = (plot_df[col].astype(int) != 0) & valid_mask
                    non_zero_count = non_zero_and_valid_mask.sum()
                    non_zero_pct_of_total = (non_zero_count / total_items) * 100 if total_items > 0 else 0

                    positive_mask = (plot_df[col] > 0) & valid_mask
                    negative_mask = (plot_df[col] < 0) & valid_mask

                    pos_count = positive_mask.sum()
                    neg_count = negative_mask.sum()

                    pos_pct_of_valid = (pos_count / non_zero_count * 100) if non_zero_count > 0 else 0
                    neg_pct_of_valid = (neg_count / non_zero_count * 100) if non_zero_count > 0 else 0

                    col_stats = plot_df[col].describe(percentiles=[.1, .25, .5, .75, .9])

                    st.write(f"**📊 Статистика по `{col}` (не-NaN значения)**")
                    stat_cols = st.columns(5)
                    stat_cols[0].metric("Среднее", f"{col_stats['mean']:.2f}" if not pd.isna(col_stats['mean']) else "—")
                    stat_cols[1].metric("Медиана", f"{col_stats['50%']:.2f}" if not pd.isna(col_stats['50%']) else "—")
                    stat_cols[2].metric("Ненулевых", f"{non_zero_count} ({non_zero_pct_of_total:.1f}% от всех)")
                    stat_cols[3].metric("Положительных", f"{pos_count} ({pos_pct_of_valid:.1f}%)")
                    stat_cols[4].metric("Отрицательных", f"{neg_count} ({neg_pct_of_valid:.1f}%)")

                    with st.expander(f"📈 Перцентили и подробности по `{col}`"):
                        perc_df = pd.DataFrame({
                            'Перцентиль': ['10%', '25%', '50%', '75%', '90%'],
                            'Значение': [
                                f"{col_stats['10%']:.2f}" if not pd.isna(col_stats['10%']) else "—",
                                f"{col_stats['25%']:.2f}" if not pd.isna(col_stats['25%']) else "—",
                                f"{col_stats['50%']:.2f}" if not pd.isna(col_stats['50%']) else "—",
                                f"{col_stats['75%']:.2f}" if not pd.isna(col_stats['75%']) else "—",
                                f"{col_stats['90%']:.2f}" if not pd.isna(col_stats['90%']) else "—"
                            ]
                        })
                        st.table(perc_df)

                        top_n = 10

                        top_positive = plot_df[positive_mask].nlargest(top_n, col)[['name', 'type', 'level', col]]
                        top_negative = plot_df[negative_mask].nsmallest(top_n, col)[['name', 'type', 'level', col]]

                        st.write(f"**🔝 Топ-{top_n} положительных:**")
                        st.dataframe(top_positive, use_container_width=True)
                        st.write(f"**🔻 Топ-{top_n} отрицательных:**")
                        st.dataframe(top_negative, use_container_width=True)

                    with st.expander(f"Распределение значений {col} по уровням{title_suffix}", expanded=False):
                        fig1 = px.strip(
                            plot_df,
                            x=col,
                            y="level",
                            color=color_col if not show_only_nonzero else "type",
                            hover_data=["name", "type", "level"],
                            title=f"Распределение значений {col} по уровням{title_suffix}",
                            orientation="h",
                            color_discrete_map=color_map
                        )
                        fig1.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="0", annotation_position="top left")
                        fig1.update_layout(height=600)
                        st.plotly_chart(fig1, use_container_width=True)

                    with st.expander(f"Распределение значений {col} по типам{title_suffix}", expanded=False):
                        # График 2: По типам
                        fig2 = px.strip(
                            plot_df,
                            x=col,
                            y="type",
                            color="level", # if show_only_nonzero else color_col,
                            hover_data=["name", "type", "level"],
                            title=f"Распределение значений {col} по типам{title_suffix}",
                            orientation="h",
                            color_discrete_map=color_map if not show_only_nonzero else None
                        )
                        fig2.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="0", annotation_position="top left")
                        fig2.update_layout(height=500)
                        st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("Выберите хотя бы одну характеристику для анализа.")

    # --- Глобальная статистика по всем характеристикам (с флагом) ---
    st.subheader("🌍 Глобальная статистика по характеристикам")

    global_stats = []
    for col in all_char_cols:
        if col not in df.columns:
            continue

        total_items = len(df)

        # Только не-NaN значения
        valid_mask = df[col].notna()
        valid_count = valid_mask.sum()
        valid_pct = (valid_count / total_items) * 100 if total_items > 0 else 0

        # Среди валидных — ненулевые
        non_zero_mask = (df[col] != 0) & valid_mask
        non_zero_count = non_zero_mask.sum()
        non_zero_pct_of_total = (non_zero_count / total_items) * 100 if total_items > 0 else 0

        positive_mask = (df[col] > 0) & valid_mask
        negative_mask = (df[col] < 0) & valid_mask

        pos_count = positive_mask.sum()
        neg_count = negative_mask.sum()

        pos_pct_of_valid = (pos_count / valid_count * 100) if valid_count > 0 else 0
        neg_pct_of_valid = (neg_count / valid_count * 100) if valid_count > 0 else 0

        mean_val = df[col].mean() if valid_count > 0 else float('nan')
        median_val = df[col].median() if valid_count > 0 else float('nan')

        global_stats.append({
            'Характеристика': col,
            'Всего предметов': total_items,
            'Не-NaN значений': valid_count,
            '% от всех': f"{valid_pct:.1f}%",
            'Ненулевых (из не-NaN)': non_zero_count,
            '% от всех': f"{non_zero_pct_of_total:.1f}%",
            'Положительных': pos_count,
            '% от валидных': f"{pos_pct_of_valid:.1f}%",
            'Отрицательных': neg_count,
            '% от валидных (отриц.)': f"{neg_pct_of_valid:.1f}%",
            'Среднее значение': f"{mean_val:.2f}" if not pd.isna(mean_val) else "—",
            'Медиана': f"{median_val:.2f}" if not pd.isna(median_val) else "—"
        })

    if global_stats:
        global_df = pd.DataFrame(global_stats)
        global_df = global_df.sort_values('Не-NaN значений', ascending=False)
        st.dataframe(global_df, use_container_width=True)

        # Интерактивный график с флагом
        st.write("### 📊 Интерактивный график: распределение значений по характеристикам")
        char_for_graph = st.selectbox(
            "Выберите характеристику для детального графика:",
            options=[col for col in all_char_cols if col in df.columns],
            index=0,
            key="global_char_select"
        )

        show_only_nonzero_global = st.checkbox(
            "Показывать только предметы с ненулевыми значениями",
            value=True,
            key="global_nonzero_filter"
        )

        if char_for_graph:
            # Создаем копию с переименованными уровнями
            graph_df = df.copy()
            graph_df['level'] = graph_df['level'].replace(level_mapping)

            if show_only_nonzero_global:
                graph_df = graph_df[graph_df[char_for_graph].notna() & (graph_df[char_for_graph] != 0)].copy()
                graph_note = " (только ненулевые и не-NaN)"
            else:
                graph_note = " (все предметы)"

            if len(graph_df) == 0:
                st.warning(f"Нет данных{graph_note} для характеристики `{char_for_graph}`.")
            else:
                graph_df = graph_df.sort_values(by=char_for_graph, ascending=True).reset_index(drop=True)

                fig_detail = px.scatter(
                    graph_df,
                    x=char_for_graph,
                    y=graph_df.index,
                    color="type",
                    symbol="level",
                    hover_data=["name", "type", "level", char_for_graph],
                    title=f"Значения {char_for_graph} по всем артефактам{graph_note}",
                    labels={"y": "Артефакт (индекс)"}
                )
                fig_detail.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="0", annotation_position="top left")
                fig_detail.update_layout(height=800, showlegend=True)
                fig_detail.update_traces(marker_size=10)
                st.plotly_chart(fig_detail, use_container_width=True)

                with st.expander("📊 Вертикальный график с именами артефактов"):
                    fig_bar = px.bar(
                        graph_df,
                        y="name",
                        x=char_for_graph,
                        color="type",
                        hover_data=["level", "type"],
                        orientation='h',
                        title=f"Значения {char_for_graph} по артефактам{graph_note}",
                        height=max(600, len(graph_df) * 25)
                    )
                    fig_bar.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="0", annotation_position="top right")
                    st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Нет характеристик для глобального анализа.")


def general_info(df):
    st.subheader("📌 Общая информация по артефактам")
    total_arts = len(df)
    total_types = df["type"].nunique()
    total_levels = df["level"].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Всего артефактов", total_arts)
    col2.metric("Уникальных типов", total_types)
    col3.metric("Уровней", total_levels)
    
    # Средние значения по ключевым метрикам
    st.markdown("### 📊 Средние значения")
    cols = st.columns(4)
    if "cost_main" in df:
        cols[0].metric("Средняя стоимость", f"{df['cost_main'].mean():.1f}")
    if "tier_main" in df:
        cols[1].metric("Средний Tier", f"{df['tier_main'].mean():.2f}")
    if "af_rank_main" in df:
        cols[2].metric("Средний Rank", f"{df['af_rank_main'].mean():.2f}")
    
    scored_df = compute_artifact_scores(df, st.session_state.STATS_WEIGHT, st.session_state.STAT_ABS_WEIGHT)
    cols[3].metric("Средний Score", f"{scored_df['total_score'].mean():.2f}")

    # 📊 График распределения артефактов по уровням с учётом новых
    st.markdown("### 📈 Распределение артефактов по уровням")

    def transform_level_column(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        df = df.copy()

        def transform_level(val):
            val_str = str(val).strip()
            return f"Ур. {val_str}" if val_str.isdigit() else val_str

        df["level"] = df["level"].apply(transform_level)

        # Получим уникальные уровни в порядке появления
        level_order = list(dict.fromkeys(df["level"].dropna()))
        df["level"] = pd.Categorical(df["level"], categories=level_order, ordered=True)

        return df, level_order


    # === Нормализуем колонку level ===
    transformed_df, level_order = transform_level_column(df)

    # Считаем общее количество и количество новых
    level_stats = (
        transformed_df.groupby("level")
        .agg(
            total_count=("level", "size"),
            new_count=("new_artefact", lambda x: (x == True).sum())
        )
        .reindex(level_order, fill_value=0)
        .reset_index()
    )

    # 📊 Строим график с двумя сериями
    fig = px.bar(
        level_stats.melt(id_vars="level", value_vars=["total_count", "new_count"],
                        var_name="Тип", value_name="Количество"),
        x="level",
        y="Количество",
        color="Тип",
        barmode="group",
        text="Количество",
        category_orders={"level": level_order},
        title="Количество артефактов по уровням (всего и новых)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 📊 График распределения по типам + новые артефакты
    st.markdown("### 📊 Распределение артефактов по типам (с выделением новых)")

    # Считаем количество по типам + new_artefact
    type_counts = (
        df.groupby(["type", "new_artefact"])
        .size()
        .reset_index(name="count")
    )

    # Чтобы легенда была понятной
    type_counts["new_artefact"] = type_counts["new_artefact"].map({True: "Новые", False: "Старые"})

    fig_type = px.bar(
        type_counts,
        x="type",
        y="count",
        color="new_artefact",   # цвет по признаку новые/старые
        text="count",
        title="Количество артефактов по типам (с разделением на новые/старые)",
        barmode="stack"
    )

    st.plotly_chart(fig_type, use_container_width=True)

def weights_editor_tab():
    st.header("⚙️ Настройка весов характеристик")

    stats_weights = st.session_state.STATS_WEIGHT.copy()
    abs_weights = st.session_state.STAT_ABS_WEIGHT.copy()

    # временные словари для редактирования
    tmp_stats = stats_weights.copy()
    tmp_abs = abs_weights.copy()

    def render_group(name, cols, weights, tmp_dict):
        st.subheader(name)
        col1, col2, col3 = st.columns(3)
        cols_split = [col1, col2, col3]

        for i, col in enumerate(cols):
            if col in weights:
                with cols_split[i % 3]:
                    new_val = st.text_input(
                        col,
                        value=str(weights[col]),
                        key=f"{name}_{col}"
                    )
                    try:
                        tmp_dict[col] = float(new_val)
                    except ValueError:
                        tmp_dict[col] = weights[col]

    # --- Группы ---
    render_group("🛡 Иммунитеты", IMMUNITY_COLS, stats_weights, tmp_stats)
    render_group("📏 Капы", CAP_COLS, stats_weights, tmp_stats)
    render_group("💉 Восстановление", RESTORE_COLS, stats_weights, tmp_stats)
    render_group("🧰 Утилити", UTILITY_COLS, stats_weights, tmp_stats)

    # Абсолютные капы
    render_group("📌 Абсолютные капы", list(abs_weights.keys()), abs_weights, tmp_abs)

    st.markdown("---")

    # Кнопка применения изменений
    if st.button("💾 Применить изменения"):
        st.session_state.STATS_WEIGHT = tmp_stats
        st.session_state.STAT_ABS_WEIGHT = tmp_abs
        st.success("✅ Веса сохранены и обновлены")

    # Сохранение в JSON
    weights_data = {
        "STATS_WEIGHT": st.session_state.STATS_WEIGHT,
        "STAT_ABS_WEIGHT": st.session_state.STAT_ABS_WEIGHT,
    }
    weights_json = json.dumps(weights_data, indent=4, ensure_ascii=False)

    st.download_button(
        label="📥 Скачать веса",
        data=weights_json,
        file_name="weights_config.json",
        mime="application/json"
    )

    # Загрузка из файла
    uploaded_file = st.file_uploader("Загрузить веса (JSON)", type="json")
    if uploaded_file is not None:
        try:
            loaded_data = json.load(uploaded_file)
            if "STATS_WEIGHT" in loaded_data and "STAT_ABS_WEIGHT" in loaded_data:
                st.session_state.STATS_WEIGHT = loaded_data["STATS_WEIGHT"]
                st.session_state.STAT_ABS_WEIGHT = loaded_data["STAT_ABS_WEIGHT"]
                st.success("✅ Веса успешно загружены из файла")
                st.rerun()
            else:
                st.error("❌ В файле нет нужных ключей (STATS_WEIGHT, STAT_ABS_WEIGHT)")
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {e}")



def main():
    st.title("🧪 Artifact Balance Manager - S.T.A.L.K.E.R. Anomaly G.A.M.M.A.")

    if 'df_data' not in st.session_state:
        raw_df = load_data()
        df, char_groups = prepare_data_not_cached(raw_df)
        st.session_state.df_data = df
        st.session_state.char_groups = char_groups
    else:
        df = st.session_state.df_data
        char_groups = st.session_state.char_groups

    if "STATS_WEIGHT" not in st.session_state or "STAT_ABS_WEIGHT" not in st.session_state:
        st.session_state.STATS_WEIGHT, st.session_state.STAT_ABS_WEIGHT = get_stats_weights()

    tab_list = [
        "📝 Редактор", 
        "📊 Графики", 
        "📔 Заметки",
        "🎯 Оптимизатор",
        "🔍 Анализ баланса",
        "ℹ️ Общая информация",
        "⚙️ Настройка весов",
        "💾 Экспорт / Импорт",
    ]

    # Главные вкладки (+ новая)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_list)

    with tab1:
        artifact_editor_tab(df, char_groups)

    with tab2:
        render_all_charts(df, char_groups)

    with tab3:
        balance_notes_tab(df)

    with tab4:
        combination_optimizer_tab(df)

    with tab5:
        balance_analyzer_tab(df)

    with tab6:
        general_info(df)

    with tab7:
        weights_editor_tab()

    with tab8:
        export_tab(df)
        

def export_tab(df):
    """Вкладка экспорта и импорта данных"""
    st.header("💾 Экспорт / Импорт данных")

    col1, col2 = st.columns(2)

    # ------------------- ЭКСПОРТ -------------------
    with col1:
        st.subheader("📄 Экспорт в LTX")

        if st.button("🚀 Подготовить финальный LTX"):
            start_time = time.time()

            progress = st.progress(0, text="⏳ Подготовка данных... (может занять до 1 минуты)")

            # Этап 1: финализация данных
            df_final = finalize_data(st.session_state.df_data)
            progress.progress(50, text="🔄 Генерация файла...")

            # Этап 2: сборка LTX
            buffer, filename = build_artifact_file(df_final)
            progress.progress(100, text="✅ Готово")

            elapsed = time.time() - start_time
            st.success(f"Файл успешно подготовлен за {elapsed:.2f} сек.")

            st.download_button(
                label="💾 Скачать финальный LTX",
                data=buffer,
                file_name=filename,
                mime="text/plain"
            )

        st.subheader("📄 Экспорт в CSV")
        st.download_button(
            label="💾 Скачать текущие данные (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"artifacts_balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        st.subheader("📊 Экспорт статистики")
        if st.button("📊 Сгенерировать статистику"):
            stats_data = []
            for idx, row in df.iterrows():
                positive_effects = 0
                negative_effects = 0

                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['artifact_id', 'cost_main', 'tier_main', 'af_rank_main']:
                        val = row[col]
                        if pd.notna(val) and val != 0:
                            if val > 0:
                                positive_effects += 1
                            else:
                                negative_effects += 1

                stats_data.append({
                    'name': row['name'],
                    'type': row['type'],
                    'level': row['level'],
                    'positive_effects_count': positive_effects,
                    'negative_effects_count': negative_effects,
                    'total_effects': positive_effects + negative_effects,
                    'balance_ratio': positive_effects / (negative_effects + 1)
                })

            stats_df = pd.DataFrame(stats_data)
            st.download_button(
                label="💾 Скачать статистику (CSV)",
                data=stats_df.to_csv(index=False).encode("utf-8"),
                file_name=f"artifacts_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        st.subheader("📋 Экспорт отчета")
        if st.button("📄 Сгенерировать отчет по балансу"):
            start_time = time.time()
            report = generate_balance_report(df)
            elapsed = time.time() - start_time

            st.success(f"✅ Отчет сгенерирован за {elapsed:.2f} сек.")
            st.download_button(
                label="💾 Скачать отчет (TXT)",
                data=report.encode("utf-8"),
                file_name=f"balance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

            with st.expander("Просмотр отчета"):
                st.text(report)

    # ------------------- ИМПОРТ -------------------
    with col2:
        st.subheader("📋 Импорт CSV")

        uploaded_csv = st.file_uploader("Загрузите CSV файл", type="csv")
        if uploaded_csv is not None:
            try:
                new_df = pd.read_csv(uploaded_csv)

                required_cols = ["name", "type", "level"]
                missing_cols = [c for c in required_cols if c not in new_df.columns]

                if missing_cols:
                    st.error(f"❌ В файле отсутствуют обязательные колонки: {missing_cols}")
                else:
                    st.session_state.df_data = new_df
                    _, st.session_state.char_groups = prepare_data_not_cached(new_df)
                    st.success("✅ Данные успешно загружены и заменены")
                    st.dataframe(new_df.head())
            except Exception as e:
                st.error(f"Ошибка при загрузке файла: {e}")


def generate_balance_report(df):
    """Генерация текстового отчета по балансу"""
    report = []
    report.append("="*60)
    report.append("ОТЧЕТ ПО БАЛАНСУ АРТЕФАКТОВ")
    report.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*60)
    report.append("")
    
    # Общая статистика
    report.append("ОБЩАЯ СТАТИСТИКА:")
    report.append(f"- Всего артефактов: {len(df)}")
    report.append(f"- Типов артефактов: {df['type'].nunique()}")
    report.append(f"- Уровней: {df['level'].nunique()}")
    report.append("")
    
    # Статистика по типам
    report.append("РАСПРЕДЕЛЕНИЕ ПО ТИПАМ:")
    type_counts = df['type'].value_counts()
    for artifact_type, count in type_counts.items():
        report.append(f"- {artifact_type}: {count} артефактов")
    report.append("")
    
    # Статистика по уровням
    report.append("РАСПРЕДЕЛЕНИЕ ПО УРОВНЯМ:")
    level_counts = df['level'].value_counts()
    for level, count in level_counts.items():
        report.append(f"- Уровень {level}: {count} артефактов")
    report.append("")
    
    # Анализ характеристик
    report.append("АНАЛИЗ ИСПОЛЬЗОВАНИЯ ХАРАКТЕРИСТИК:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    usage_stats = []
    for col in numeric_cols:
        if col not in ['artifact_id', 'cost_main', 'tier_main', 'af_rank_main']:
            non_zero = (df[col].notna() & (df[col] != 0)).sum()
            if non_zero > 0:
                usage_stats.append((col, non_zero))
    
    usage_stats.sort(key=lambda x: x[1], reverse=True)
    
    report.append("Топ-10 используемых характеристик:")
    for char, count in usage_stats[:10]:
        percent = (count / len(df)) * 100
        report.append(f"- {char}: {count} артефактов ({percent:.1f}%)")
    report.append("")
    
    # Проблемные зоны
    report.append("ПОТЕНЦИАЛЬНЫЕ ПРОБЛЕМЫ БАЛАНСА:")
    
    # Поиск слишком сильных артефактов
    strong_artifacts = []
    for idx, row in df.iterrows():
        positive_sum = 0
        negative_sum = 0
        
        for col in numeric_cols:
            if col not in ['artifact_id', 'cost_main', 'tier_main', 'af_rank_main']:
                val = row[col]
                if pd.notna(val):
                    if val > 0:
                        positive_sum += val
                    else:
                        negative_sum += abs(val)
        
        balance = positive_sum - negative_sum * 0.5
        if balance > df['cost_main'].mean() * 2:  # Условный критерий
            strong_artifacts.append((row['name'], balance))
    
    if strong_artifacts:
        report.append("Потенциально переусиленные артефакты:")
        for name, score in sorted(strong_artifacts, key=lambda x: x[1], reverse=True)[:5]:
            report.append(f"- {name} (баланс: {score:.2f})")
    report.append("")
    
    # Заключение
    # report.append("="*60)
    # report.append("РЕКОМЕНДАЦИИ:")
    # report.append("1. Проверить артефакты-выбросы на предмет дисбаланса")
    # report.append("2. Убедиться в правильной прогрессии по уровням")
    # report.append("3. Проверить синергию между типами артефактов")
    # report.append("4. Протестировать популярные комбинации в игре")
    # report.append("="*60)
    
    return "\n".join(report)

if __name__ == "__main__":
    main()