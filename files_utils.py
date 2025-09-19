import pandas as pd
import re
import chardet
import io

def parse_artifact_file(path: str, encoding=None) -> pd.DataFrame:
    def read_file_autoencoding(path):
        with open(path, 'rb') as f:
            raw = f.read()
        result = chardet.detect(raw)
        encoding = result['encoding'] or 'cp1251'
        print(f"Encoding result: {encoding}")
        return encoding

    section_re = re.compile(r'^(!?)\s*\[([^\]]+)\](?::([^\s]+))?')
    data = []

    current_artifact = None
    section_type = None
    inherits = None
    section_index = -1
    line_index = -1
    original_header = None

    if encoding is None:
        encoding = read_file_autoencoding(path)

    with open(path, encoding=encoding) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line_index += 1
        raw = line.rstrip("\n")
        stripped = line.strip()

        match = section_re.match(stripped)
        if match:
            bang, section_name, inherited = match.groups()
            inherits = inherited
            original_header = stripped

            section_index += 1
            line_index = 0  # reset for new section

            base_name = section_name
            if base_name.endswith('_absorbation'):
                current_artifact = base_name.replace('_absorbation', '')
                section_type = 'absorbation'
            else:
                current_artifact = base_name
                section_type = 'main'
            continue

        # Комментарии и пустые строки сохранять не будем, это только про ключи
        if not stripped:
            continue

        commented = False
        deleted = False
        key = None
        value = None

        line_clean = stripped

        if line_clean.startswith(';'):
            commented = True
            line_clean = line_clean[1:].strip()

        if '=' in line_clean:
            key, val = line_clean.split('=', 1)
            key = key.strip()
            value = val.strip()
        elif line_clean.startswith('!'):
            key = line_clean[1:].strip()
            deleted = True
            value = None
        else:
            continue  # unknown format, skip

        if current_artifact is not None and key:
            data.append({
                'artifact': current_artifact,
                'section_type': section_type,
                'key': key,
                'value': value,
                'commented': commented,
                'deleted': deleted,
                'inherits': inherits,
                'order': line_index,
                'section_order': section_index,
                'original_header': original_header
            })

    return pd.DataFrame(data)

def write_artifact_file_2(df: pd.DataFrame, path: str):
    df = df.copy()
    df['section_full'] = df['artifact'] + '__' + df['section_type']

    # Получаем уникальные секции в порядке их появления в df
    section_order = df[['section_full', 'section_order']].drop_duplicates().sort_values('section_order')
    section_full_list = section_order['section_full'].tolist()

    lines = []

    for section_full in section_full_list:
        section_df = df[df['section_full'] == section_full]
        artifact, section_type = section_full.split('__')
        section_name = artifact if section_type == 'main' else f"{artifact}_absorbation"

        inherits = section_df['inherits'].dropna().unique()
        inherits_str = f":{inherits[0]}" if len(inherits) > 0 else ""

        original_header = section_df.iloc[0]['original_header']
        is_overwritten = original_header.startswith("![") if original_header else False

        # Собираем заголовок секции
        header = original_header if original_header else f"{'!' if is_overwritten else ''}[{section_name}]{inherits_str}"
        lines.append(header)

        # Запоминаем уже записанные ключи (во избежание дубликатов)
        seen_keys = set()

        for _, row in section_df.iterrows():
            key = row['key']
            if key in seen_keys:
                continue  # пропускаем дубликаты
            seen_keys.add(key)

            value = row['value'] # Вот тут должны быть числа без e
            commented = row['commented']
            deleted = row['deleted']

            if deleted:
                line = f"!{key}"
            elif commented:
                line = f"; {key} = {value}"
            else:
                line = f"{key} = {value}"

            lines.append(line)

        lines.append("")  # пустая строка между секциями

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# Перевод в сводную таблицу
def artifacts_to_pivot(df: pd.DataFrame, include_empty: bool = True) -> pd.DataFrame:
    """
    Преобразует артефакт DataFrame в pivot таблицу для редактирования.
    Возвращает MultiIndex DataFrame (artifact, section_type) по строкам, keys по столбцам.
    """
    # Даже если строка закомментирована, мы сохраняем её значение
    df_clean = df.copy()
    df_clean["pivot_value"] = df_clean["value"]
    df_clean.loc[df_clean["commented"], "pivot_value"] = None  # отключаем закомментированные строки

    pivot = (
        df_clean
        .pivot_table(
            index=["artifact", "section_type"],
            columns="key",
            values="pivot_value",
            aggfunc="first"
        )
    )

    if include_empty:
        # Добавим артефакты, у которых вообще нет значений
        all_artifacts = df[["artifact", "section_type"]].drop_duplicates()
        pivot = all_artifacts.merge(pivot, how="left", on=["artifact", "section_type"]).set_index(["artifact", "section_type"])


    pivot = pivot.sort_index()
    return pivot

def pivot_to_artifacts(pivot_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует pivot таблицу обратно в long формат df, эффективно и без потерь.
    """
    # Словарь для быстрого доступа к оригинальным строкам
    original_map = {
        (row.artifact, row.section_type, row.key): row._asdict()
        for row in original_df.itertuples()
    }

    records = []
    used_keys = set()

    for (artifact, section_type), row in pivot_df.iterrows():
        for key, value in row.items():
            key_id = (artifact, section_type, key)
            used_keys.add(key_id)

            if pd.isna(value):
                # если в оригинале такая строка есть — оставим как есть
                if key_id in original_map:
                    records.append(original_map[key_id])
                continue

            value = str(value).strip()
            if key_id in original_map:
                base = original_map[key_id].copy()
                base["value"] = value
                base["deleted"] = False
                base["commented"] = False
            else:
                base = {
                    "artifact": artifact,
                    "section_type": section_type,
                    "key": key,
                    "value": value,
                    "commented": False,
                    "deleted": False,
                    "inherits": None,
                    "order": 999,
                    "section_order": 999,
                    "original_header": None
                }
            records.append(base)

    # Добавляем все нетронутые строки (комментарии, !удаления, которые не были в pivot)
    untouched = [
        v for k, v in original_map.items()
        if k not in used_keys
    ]
    records.extend(untouched)

    output_df = pd.DataFrame(records)
    if "Index" in pivot_df.columns:
        output_df.drop(columns=["Index"], inplace=True)
        
    return output_df

def filter_and_sort_pivot_df(pivot_df, arts_list, col_list):
    # Фильтруем строки по первому уровню MultiIndex ("artifact")
    # Фильтруем столбцы по col_list (оставляя только те, которые есть в pivot_df.columns)
    df = pivot_df.loc[pivot_df.index.get_level_values(0).isin(arts_list)].copy()
    df = df[[col for col in col_list if col in df.columns]]
    return df

# ----------------------------------------------------
# - Обработка файла с инфой об артефактах            -
# ----------------------------------------------------

def load_filter_ltx():
    filename = "mod_system_zzzzzzzzzzzzzzzzz_grok_artifacts.ltx"
    df = parse_artifact_file(filename)
    pivot_df = artifacts_to_pivot(df)

    # Находим список артефактов
    ignore_ending_list = ["_iam", "_lead_box", "_af_aac", "_af_aam", 
        "detector_scientific", "detector_elite", "af_base", "af_base_mlr",
        'artefact_spawn_zones','artefact_hud',
        ]

    arts_list = set([i for i in list(pivot_df.index.get_level_values(0)) if not any([i.endswith(j) for j in ignore_ending_list])])

    # Оставляем только нужные колонки
    col_list = [
        # Базовые
        'additional_inventory_weight', 'additional_inventory_weight2',
        
        'bleeding_restore_speed', 'health_restore_speed', 'power_restore_speed', 'radiation_restore_speed', 'satiety_restore_speed',

        'burn_immunity', 'chemical_burn_immunity', 'explosion_immunity',
        'fire_wound_immunity', 'radiation_immunity', 'shock_immunity',
        'strike_immunity', 'telepatic_immunity', 'wound_immunity',

        'telepatic_cap', 'wound_cap', 'strike_cap',
        'shock_cap', 'fire_wound_cap', 'explosion_cap',
        'chemical_burn_cap', 'burn_cap',

        'cost', 'tier', 'af_rank', 'inv_weight', 'jump_height',

        # Новые
        "br_class_artefact",
        "speed_modifier",
        
        "eat_thirstiness", "eat_sleepiness","eat_satiety",
        
        "bleeding_restore_speed_hard",
        
        "psy_health_restore_speed", "psy_health_cap",
        
        "psy_health_abs_cap", "telepatic_abs_cap", "fire_wound_abs_cap",
        "wound_abs_cap", "strike_abs_cap", "explosion_abs_cap",
        "burn_abs_cap", "shock_abs_cap", "chemical_burn_abs_cap",

        "k_art_level", "k_art_subtype", "k_art_temp","dizziness"
    ]

    filtered_df = filter_and_sort_pivot_df(pivot_df, arts_list, col_list)

    return df, filtered_df

# -----------------------------------
# - Нормализуем данные от редактора -
# -----------------------------------

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    immunity_adjusters = {
        "fire_wound_immunity": {
            "adjuster": 0.80,
            "arti_adjuster": 0.80
        },
        "wound_immunity": {
            "adjuster": 0.58 * 1.20,  # = 0.696
            "arti_adjuster": 0.58 * 2  # = 1.16
        },
        "strike_immunity": {
            "adjuster": 0.55,
            "arti_adjuster": 0.55 * 1.3  # = 0.715
        },
        "explosion_immunity": {
            "adjuster": 0.45,
            "arti_adjuster": 0.675
        },
        "burn_immunity": {
            "adjuster": 1.00,
            "arti_adjuster": 1.00 * 2.0  # = 2.0
        },
        "shock_immunity": {
            "adjuster": 0.102,
            "arti_adjuster": 0.102 * 26.5  # = 2.703
        },
        "chemical_burn_immunity": {
            "adjuster": 1.32,
            "arti_adjuster": 1.32
        },
        "telepatic_immunity": {
            "adjuster": 1.70,
            "arti_adjuster": 1.50
        }
    }

    normalized_data = {
        'burn_immunity': [1/(100*0.6*2.0), 1.0],                # 1/(100*0.6*2.0) = 1/120
        'shock_immunity': [1/(100*0.6*2.703), 1.0],             # 1/(100*0.6*2.703) ≈ 1/162.18
        'radiation_immunity': [1 / 11399.4, 1.0],               # НЕ использует arti_adjuster → 1/(126.66*90)
        'telepatic_immunity': [1/(100*0.6*1.5), 1.0],           # 1/(100*0.6*1.5) = 1/90
        'chemical_burn_immunity': [1/(100*0.6*1.32), 1.0],      # 1/(100*0.6*1.32) = 1/79.2
        'wound_immunity': [1/(100*0.6*1.16), 1.0],              # 1/(100*0.6*1.16) = 1/69.6
        'strike_immunity': [1/(100*0.6*0.715), 1.0],            # = 1/42.9 → можно заменить на 1/(100*0.6*0.715)
        'explosion_immunity': [1/40.5, 1.0],                    # 1/(100*0.6*0.675) = 1/40.5
        'fire_wound_immunity': [1/(100*0.6*0.8), 1.0],          # = 1/48 → можно заменить на 1/(100*0.6*0.8)

        # Restore speeds 
        'health_restore_speed': [1 / 10000, 1.0],    
        'radiation_restore_speed': [1 / 47000, 1.0],    
        'power_restore_speed': [1 / 30000, 1.0],             
        'bleeding_restore_speed': [1 / 150000, 1.0],

        # Остальные — без изменений
        "br_class_artefact":            [0.01, 1.0], 
        "speed_modifier":               [0.01, 1.0], 
        "eat_thirstiness":              [-1, 1.0],
        "eat_sleepiness":               [-1, 1.0], 
        "eat_satiety":                  [1, 1.0],
        "bleeding_restore_speed_hard":  [1 / 150000, 1.0],  
        "psy_health_restore_speed":     [1 / 360000, 1.0],
        "psy_health_cap":               [1, 1.0],    
        "psy_health_abs_cap":           [1, 1.0],        
        "telepatic_abs_cap":            [1, 1.0],       
        "fire_wound_abs_cap":           [1, 1.0],      
        "wound_abs_cap":                [1, 1.0],       
        "strike_abs_cap":               [1, 1.0],   
        "explosion_abs_cap":            [1, 1.0],     
        "burn_abs_cap":                 [1, 1.0],    
        "shock_abs_cap":                [1, 1.0],       
        "chemical_burn_abs_cap":        [1, 1.0],    
        "dizziness":                    [0.01, 1.0],  
    }

    df_last_normalized = df.copy()

    for base_key, (multiplier, _) in normalized_data.items():
        # Найдём все колонки, начинающиеся на этот ключ
        matching_cols = [col for col in df.columns if col.startswith(base_key)]
        for col in matching_cols:
            df_last_normalized[col] = df[col] * multiplier

    # rank, tier -> int
    df_last_normalized['af_rank_main'] = df_last_normalized['af_rank_main'].apply(
        lambda x: int(x) if pd.notna(x) else x
    )
    df_last_normalized['tier_main'] = df_last_normalized['tier_main'].apply(
        lambda x: int(x) if pd.notna(x) else x
    )
    df_last_normalized['af_rank_main'] = pd.to_numeric(df_last_normalized['af_rank_main'], downcast='integer')
    df_last_normalized['tier_main'] = pd.to_numeric(df_last_normalized['tier_main'], downcast='integer')
    df_last_normalized['af_rank_main'] = df_last_normalized['af_rank_main'].astype('Int64')
    df_last_normalized['tier_main'] = df_last_normalized['tier_main'].astype('Int64')

    # ПЕРЕИМЕНОВЫВАЕМ КОЛОНКИ ТИПА, УРОВНЯ И ТЕМПЕРАТУРЫ, ЧТОБЫ ОНИ ДОБАВИЛИСЬ В ФАЙЛ С ХАРАКТЕРИСТИКАМИ.
    df_last_normalized = df_last_normalized.rename({
        'k_art_subtype':'k_art_subtype_main',
        'k_art_level':'k_art_level_main',
        'k_art_temp':'k_art_temp_main'},
        axis=1
    )

    df_last_normalized = df_last_normalized.rename({
        'speed_modifier_main':'speed_modifier_art_main'},
        axis=1
    )

    return df_last_normalized

# ---------------------------------
# - Подготавливаем финальный файл -
# ---------------------------------

def update_df_from_normalized(df: pd.DataFrame, df_last_normalized: pd.DataFrame) -> pd.DataFrame:
    # Пример списка обязательных ключей
    required_main_columns = [
    'br_class_artefact_main',
    'speed_modifier_art_main',
    'eat_thirstiness_main',
    'eat_sleepiness_main',
    'eat_satiety_main',
    'bleeding_restore_speed_hard_main',
    'psy_health_restore_speed_main',
    'psy_health_cap_main',
    'psy_health_abs_cap_main',
    'telepatic_abs_cap_main',
    'fire_wound_abs_cap_main',
    'wound_abs_cap_main',
    'strike_abs_cap_main',
    'explosion_abs_cap_main',
    'burn_abs_cap_main',
    'shock_abs_cap_main',
    'chemical_burn_abs_cap_main',
    'dizziness_main',

    'bleeding_restore_speed_main',
    'health_restore_speed_main',
    'power_restore_speed_main',
    'radiation_restore_speed_main',
    'satiety_restore_speed_main',
    
    'wound_cap_main',
    'strike_cap_main',
    'explosion_cap_main',
    'fire_wound_cap_main',
    'chemical_burn_cap_main',
    'burn_cap_main',
    'shock_cap_main',
    'telepatic_cap_main',
    'wound_immunity_abs',
    'strike_immunity_abs',
    'explosion_immunity_abs',
    'fire_wound_immunity_abs',
    'chemical_burn_immunity_abs',
    'burn_immunity_abs',
    'shock_immunity_abs',
    'telepatic_immunity_abs',
    'radiation_immunity_abs',
    ]

    df_updated = df.copy()
    df_last = df_last_normalized.copy()

    # Обеспечим наличие всех обязательных столбцов в df_last и заполним NaN нулями
    for col in required_main_columns:
        if col not in df_last.columns:
            df_last[col] = 0.0
        else:
            df_last[col] = df_last[col].fillna(0.0)

    # Все значения, которые нас интересуют (main и abs)
    value_columns = [col for col in df_last.columns if col.endswith("_main") or col.endswith("_abs")]
    value_prefixes = {"_main": "main", "_abs": "absorbation"}

    new_rows = []

    for _, row in df_last.iterrows():
        artifact = row["artifact"]

        for col in value_columns:
            # Берём значение — если NaN, то 0.0 (уже заполнено выше, но на всякий случай)
            value = row[col] if not pd.isna(row[col]) else 0.0

            # Определяем section_type и key
            section_type = None
            for suffix, stype in value_prefixes.items():
                if col.endswith(suffix):
                    section_type = stype
                    key = col.replace(suffix, "")
                    break
            if section_type is None:
                continue  # Не main и не abs — пропускаем

            # Проверяем, есть ли такая строка уже в df_updated
            mask = (
                (df_updated["artifact"] == artifact) &
                (df_updated["section_type"] == section_type) &
                (df_updated["key"] == key)
            )

            if mask.any():
                # Обновляем ВСЕГДА — даже если value был 0.0
                df_updated.loc[mask, ["value", "commented", "deleted"]] = [value, False, False]
            else:
                # Добавляем новую строку
                section_mask = (df_updated["artifact"] == artifact) & (df_updated["section_type"] == section_type)
                next_order = df_updated.loc[section_mask, "order"].max()
                next_order = int(next_order) + 1 if pd.notna(next_order) else 1

                try:
                    original_header = df_updated.loc[section_mask, "original_header"].iat[0]
                except IndexError:
                    print(f"[WARN] Не удалось получить значение 'original_header' для {artifact} (секция {section_type}).")
                    continue

                new_row = {
                    "artifact": artifact,
                    "section_type": section_type,
                    "key": key,
                    "value": value,
                    "commented": False,
                    "deleted": False,
                    "inherits": f"af_base{'' if section_type == 'main' else '_absorbation'}",
                    "order": next_order,
                    "section_order": df_updated.loc[section_mask, "section_order"].iloc[0] if not df_updated.loc[section_mask].empty else 100,
                    "original_header": original_header,
                }
                new_rows.append(new_row)

    # Добавляем новые строки
    if new_rows:
        df_updated = pd.concat([df_updated, pd.DataFrame(new_rows)], ignore_index=True)

    return df_updated

def finalize_data(df_last):
    df, filtered_df = load_filter_ltx()

    df_last_normalized = normalize_dataframe(df_last.copy())

    df_final = update_df_from_normalized(df.copy(), df_last_normalized.copy())

    return df_final


def build_artifact_file(df: pd.DataFrame) -> tuple[io.BytesIO, str]:
    df = df.copy()
    df['section_full'] = df['artifact'] + '__' + df['section_type']

    # Уникальные секции в порядке появления
    section_order = df[['section_full', 'section_order']].drop_duplicates().sort_values('section_order')
    section_full_list = section_order['section_full'].tolist()

    lines = []

    for section_full in section_full_list:
        section_df = df[df['section_full'] == section_full]
        artifact, section_type = section_full.split('__')
        section_name = artifact if section_type == 'main' else f"{artifact}_absorbation"

        inherits = section_df['inherits'].dropna().unique()
        inherits_str = f":{inherits[0]}" if len(inherits) > 0 else ""

        original_header = section_df.iloc[0]['original_header']
        is_overwritten = original_header.startswith("![") if original_header else False

        # Заголовок секции
        header = original_header if original_header else f"{'!' if is_overwritten else ''}[{section_name}]{inherits_str}"
        lines.append(header)

        seen_keys = set()

        for _, row in section_df.iterrows():
            key = row['key']
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # === ВАЖНО ===
            # Берём значение ровно как в старой функции, без преобразования
            value = row['value']  

            commented = row['commented']
            deleted = row['deleted']

            if deleted:
                line = f"!{key}"
            elif commented:
                line = f"; {key} = {value}"
            else:
                line = f"{key} = {value}"

            lines.append(line)

        lines.append("")  # пустая строка между секциями

    content = "\n".join(lines)
    buffer = io.BytesIO(content.encode("utf-8"))
    filename = "mod_system_zzzzzzzzzzzzzzzzz_grok_artifacts.ltx"

    return buffer, filename


# write_artifact_file_2(df_final, "mod_system_zzzzzzzzzzzzzzzzz_grok_artifacts.ltx")