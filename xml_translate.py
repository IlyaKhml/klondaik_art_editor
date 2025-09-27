import xml.etree.ElementTree as ET
import pandas as pd
import io
import codecs
import sys
import unicodedata

# Пути до файлов локализации
XML_PATHS = {
    # Файлы с описанием артефактов
    "st_items_artefacts_eng":r"xml\st_items_artefacts_eng.xml",
    "st_items_artefacts_rus":r"xml\st_items_artefacts_rus.xml",

    # Файлы для инциклопедии
    "ui_st_encyclopedia_artifacts_eng":r"xml\ui_st_encyclopedia_artifacts_eng.xml",
    "ui_st_encyclopedia_artifacts_rus":r"xml\ui_st_encyclopedia_artifacts_rus.xml",
}

# ------------------ RU ------------------
art_type_names = {
    -1: "Неизвестный",
    0: "Мусорный",
    1: "Термозащитный",
    2: "Витальный",
    3: "Кислотнозащитный",
    4: "Энергетический",
    5: "Пси-защитный",
    6: "Антигравитационный",
    7: "Антирадиационный",
    8: "Коагуляционный",
    9: "Электрозащитный",
    10: "Пулезащитный",
    11: "Разрывозащитный",
    12: "Уникальный",
    13: "Синтетический",
}

art_levels_names = {
    -1: "Неизвестный",
    0: "Мусорный",
    1: "1 уровень",
    2: "2 уровень",
    3: "3 уровень",
    4: "4 уровень",
    5: "5 уровень",
    6: "6 уровень",
    7: "Абсолют",
    8: "Базовый (синтетический)",
    9: "Модификат",
    10: "Мезомодификат",
    11: "Гипермодификат",
    12: "Абсолют (синтетический)",
    13: "1 уровень (уникальный)",
    14: "2 уровень (уникальный)",
    15: "3 уровень (уникальный)",
    16: "4 уровень (уникальный)",
    17: "Абсолют (уникальный)",
}

art_temp_names = {
    -1: 'Охлаждает',
    0: "",
    1: "Согревает",
}

val_defs = ["Тип", "Уровень", "Температура"]

# ------------------ EN ------------------
art_type_names_en = {
    -1: "Unknown",
    0: "Junk",
    1: "Thermal Protection",
    2: "Vital",
    3: "Acid Protection",
    4: "Energy",
    5: "Psi Protection",
    6: "Anti-Gravity",
    7: "Anti-Radiation",
    8: "Coagulation",
    9: "Electric Protection",
    10: "Bullet Protection",
    11: "Wound Protection",
    12: "Unique",
    13: "Synthetic",
}

art_levels_names_en = {
    -1: "Unknown",
    0: "Junk",
    1: "1 level",
    2: "2 level",
    3: "3 level",
    4: "4 level",
    5: "5 level",
    6: "6 level",
    7: "Absolute",
    8: "Basic (synthetic)",
    9: "Modifier (synthetic)",
    10: "Mesomodifier (synthetic)",
    11: "Hypermodifier (synthetic)",
    12: "Absolute (synthetic)",
    13: "1 level (unique)",
    14: "2 level (unique)",
    15: "3 level (unique)",
    16: "4 level (unique)",
    17: "Absolute (unique)",
}

art_temp_names_en = {
    -1: 'Cools',
    0: "",
    1: "Warms",
}

val_defs_en = ["Type", "Level", "Temperature"]

# Общие для всех языков цвета.
# Цвета ARGB
art_type_colors = {
    -1: [0, 80, 80, 80],      # Неизвестный
    0: [0, 128, 128, 128],    # Мусорный
    1: [0, 255, 160, 0],      # Термозащитный
    2: [0, 213, 70, 125],     # Витальный
    3: [0, 200, 255, 0],      # Кислотнозащитный
    4: [0, 0, 200, 255],      # Энергетический
    5: [0, 125, 0, 210],      # Пси-защитный
    6: [0, 255, 255, 0],      # Антигравитационный
    7: [0, 255, 217, 102],    # Антирадиационный
    8: [0, 133, 32, 12],      # Коагуляционный
    9: [0, 74, 134, 232],     # Электрозащитный
    10: [0, 120, 120, 120],   # Пулезащитный
    11: [0, 180, 225, 255],   # Разрывозащитный
    12: [0, 255, 0, 0],       # Уникальный
    13: [0, 255, 255, 255],   # Синтетический
}

art_levels_colors = {
    -1: [255, 80, 80, 80],        # Неизвестный
    0: [255, 80, 80, 80],         # 0/мусорный — светло-серый
    1: [255, 220, 220, 220],      # 1 — почти белый
    2: [255, 224, 224, 224],      # 2 — серебристый
    3: [255, 255, 200, 100],      # 3 — жёлтый
    4: [255, 255, 160, 60],       # 4 — оранжевый
    5: [255, 229, 91, 130],       # 5 — тёмно-оранжевый
    6: [255, 145, 61, 219],       # 6 — красно-розовый
    7: [255, 58, 220, 69],        # Абсолют — ярко-зелёный

    # Синтетические
    8: [255, 255, 255, 255],      # Базовый (синтетический)
    9: [255, 153, 205, 255],      # Модификат
    10: [255, 114, 163, 255],     # Мезомодификат 
    11: [255, 142, 127, 255],     # Гипермодификат 
    12: [255, 120, 230, 130],     # Абсолют (синтетический)

    # Уникальные
    13: [255, 250, 170, 100],     # 1 ур
    14: [255, 240, 130, 70],      # 2 ур
    15: [255, 230, 90, 40],       # 3 ур
    16: [255, 230, 45, 20],       # 4 ур
    17: [255, 58, 220, 69],       # Абсолют-уникальный
}

art_thermo_type_colors = {
    -1: [255, 100, 100, 255],     # Охлаждает
    0: [255, 140, 140, 140],      # 
    1: [255, 238, 153, 26],       # Согревает
}


# -------------------------------- Функции ------------------------------

def add_extra_text_columns(df_last, art_type_names, art_levels_names, art_temp_names,
                           art_type_colors, art_levels_color, art_thermo_type_colors,
                           val_defs, col_suffix = ''):
    """
    Добавляет в датафрейм df_last три новых столбца с текстом:
    extra_text_type, extra_text_level, extra_text_term
    
    Параметры:
    df_last - исходный датафрейм с колонками 'k_art_subtype', 'k_art_level', 'k_art_temp'
    словари названий и цветов
    val_defs - список названий строк ["Тип", "Уровень", "Температура"]
    col_suffix - строка, чтобы добавить в конец называния новых столбцов (например '_eng')
    """
    
    # Создаем копию датафрейма для работы
    df = df_last.copy()

    new_cols = ['extra_text_type' + col_suffix, 'extra_text_level' + col_suffix, 'extra_text_term' + col_suffix]
    
    # Словари для сопоставления колонок с параметрами
    column_mapping = {
        'k_art_subtype': {
            'names_dict': art_type_names,
            'colors_dict': art_type_colors,
            'val_def': val_defs[0],  # "Тип"
            'output_col': new_cols[0]
        },
        'k_art_level': {
            'names_dict': art_levels_names,
            'colors_dict': art_levels_color,
            'val_def': val_defs[1],  # "Уровень"
            'output_col': new_cols[1]
        },
        'k_art_temp': {
            'names_dict': art_temp_names,
            'colors_dict': art_thermo_type_colors,
            'val_def': val_defs[2],  # "Температура"
            'output_col': new_cols[2]
        }
    }
    
    # Обрабатываем каждую колонку
    for col_name, params in column_mapping.items():
        names_dict = params['names_dict']
        colors_dict = params['colors_dict']
        val_def = params['val_def']
        output_col = params['output_col']
        
        # Создаем новый столбец
        df[output_col] = df[col_name].apply(lambda x: format_text_cell(x, names_dict, colors_dict, val_def))
    
    return df, new_cols

def format_text_cell(value, names_dict, colors_dict, val_def):
    """
    Форматирует текст для одной ячейки согласно заданным правилам
    """
    # Получаем название типа
    type_val = names_dict.get(value, "")
    
    # Если название пустое, возвращаем пустую строку
    if type_val == "":
        return ""
    
    # Получаем цвета
    colors = colors_dict.get(value, [0, 0, 0, 0])
    c0, c1, c2, c3 = colors
    
    # Формируем строку по шаблону
    result = f'\\n%c[0,0,150,0] • %c[255,140,140,140] {val_def}: %c[{c0},{c1},{c2},{c3}]{type_val}'
    
    return result

# import unicodedata

def clean_for_cp1251(text: str, replacement: str = " ") -> str:
    """
    Заменяет все символы, которые не могут быть закодированы в windows-1251,
    на указанный replacement (по умолчанию — пробел).
    Также выводит в stderr список уникальных недопустимых символов и их названия (если есть).
    """
    # Получаем таблицу кодировки cp1251
    try:
        encoder = codecs.getencoder('cp1251')
    except LookupError:
        # fallback на latin1 или что-то ещё — но cp1251 всегда есть в Windows Python
        raise ValueError("Кодировка cp1251 недоступна")

    bad_chars = set()
    cleaned = []

    for char in text:
        try:
            encoder(char)
            cleaned.append(char)
        except UnicodeEncodeError:
            bad_chars.add(char)
            cleaned.append(replacement)

    if bad_chars:
        # Выводим информацию о проблемных символах
        info_lines = ["Обнаружены символы, не поддерживаемые в windows-1251:"]
        for ch in sorted(bad_chars):
            name = unicodedata.name(ch, "UNKNOWN NAME")
            code = f"U+{ord(ch):04X}"
            info_lines.append(f"  '{ch}' → {code} ({name})")
        # Выводим в stderr (видно в консоли при запуске Streamlit)
        print("\n".join(info_lines), file=sys.stderr)

    return "".join(cleaned)

### -----------------------------------------------------------------------
## st_items_artefacts.xml
# > Полный путь: gamedata\configs\text\eng\st_items_artefacts.xml

def create_description(row, col_list):
    """
    Создает объединенное описание из нескольких столбцов

    Parameters:
    row: строка датафрейма
    col_list: названия колонок в порядке конкотинации
    
    Returns:
    str: объединенная строка описания
    """
    
    string = ''
    for i, col in enumerate(col_list):
        # Проверяем, что значение существует и не является NaN
        value = row[col] if pd.notna(row[col]) else ''
        if value == '':
            continue
        string += str(value)
        
        if i == 0:  # Первый элемент
            string += "\n\t\t"
        elif i == 1:  # Второй элемент
            string += "\n\t\t"
        elif i == 2:  # Третий элемент
            string += "\n\t\t"
        else:
            string += "\n\t\t\\n"
    
    return string

def update_artifact_descriptions_in_xml(
    xml_path: str,
    df: pd.DataFrame,
    descr_col: str = None,
    name_col: str = None
) -> io.BytesIO:
    """
    Обновляет описание и/или названия артефактов в XML-файле на основе указанных столбцов в датафрейме.
    Возвращает объект BytesIO с XML-данными в кодировке windows-1251.

    :param xml_path: Путь к исходному XML-файлу.
    :param df: DataFrame, содержащий 'artifact_id' и нужные столбцы.
    :param descr_col: Название столбца с данными для обновления _descr тегов. Если None, не обновляет.
    :param name_col: Название столбца с данными для обновления _name тегов. Если None, не обновляет.
    :return: io.BytesIO с XML-контентом в кодировке windows-1251.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Создаем словари для каждого типа тегов
    descr_map = dict(zip(df["artifact_id"], df[descr_col])) if descr_col else {}
    name_map = dict(zip(df["artifact_id"], df[name_col])) if name_col else {}

    updated_descr_count = 0
    updated_name_count = 0
    
    for elem in root.findall("string"):
        id_attr = elem.attrib.get("id", "")
        if not id_attr.startswith("st_"):
            continue

        # Определяем тип тега и извлекаем artifact_id
        if id_attr.endswith("_descr"):
            tag_type = "descr"
            artifact_id = id_attr[3:-6]  # st_af_itcher_descr → af_itcher
            data_map = descr_map
        elif id_attr.endswith("_name"):
            tag_type = "name"
            artifact_id = id_attr[3:-5]  # st_af_itcher_name → af_itcher
            data_map = name_map
        else:
            continue

        # Проверяем, есть ли данные для обновления
        if not data_map or artifact_id not in data_map:
            print(f"Не нашли {id_attr}")
            continue

        text_elem = elem.find("text")
        if text_elem is not None:
            new_text = data_map[artifact_id]
            new_text = str(new_text).strip()

            # Обновляем содержимое
            text_elem.clear()
            text_elem.text = new_text

            if tag_type == "descr":
                updated_descr_count += 1
            else:
                updated_name_count += 1

    print(f"✅ Обновлено _descr тегов: {updated_descr_count}")
    print(f"✅ Обновлено _name тегов: {updated_name_count}")

    # Записываем XML в строку с XML-декларацией
    # ElementTree не добавляет декларацию при tostring, поэтому добавим вручную
    xml_str = ET.tostring(root, encoding="unicode")
    xml_with_decl = f'<?xml version="1.0" encoding="windows-1251"?>\n{xml_str}'

    xml_with_decl = clean_for_cp1251(xml_with_decl)

    # Кодируем в windows-1251 и оборачиваем в BytesIO
    buffer = io.BytesIO(xml_with_decl.encode("windows-1251"))
    return buffer


def get_st_items_artefacts(df_last:pd.DataFrame, lang:str):
    """
    Создаёт файл локализации для пути:
    gamedata\\configs\\text\\rus\\st_items_artefacts.xml
    gamedata\\configs\\text\\eng\\st_items_artefacts.xml

    Args:
        df_last (pd.DataFrame): Данные
        lang (str): Язык из ['rus', 'eng']
    """
    if lang not in ['rus', 'eng']:
        raise Exception(f"Язык {lang} не поддерживается.")
    
    df_last = df_last.copy()
    
    if lang == 'rus':
        # Создаём новые колонки
        df_last, new_cols = add_extra_text_columns(
            df_last, 
            art_type_names, 
            art_levels_names, 
            art_temp_names,
            art_type_colors, 
            art_levels_colors, 
            art_thermo_type_colors,
            val_defs
        )

        # Для русского описания
        col_list = ['main_description'] + new_cols
        df_last['description_new'] = df_last.apply(lambda row: create_description(row, col_list), axis=1)

        descr_col = 'description_new'
        name_col = 'name'

    elif lang == 'eng':
        df_last, new_cols_eng = add_extra_text_columns(
            df_last,
            art_type_names_en, 
            art_levels_names_en, 
            art_temp_names_en,
            art_type_colors, 
            art_levels_colors, 
            art_thermo_type_colors,
            val_defs_en,
            '_eng'
        )

        # Для английского описания
        eng_col_list = ['main_description_eng'] + new_cols_eng
        df_last['description_new_eng'] = df_last.apply(lambda row: create_description(row, eng_col_list), axis=1)

        descr_col = 'description_new_eng'
        name_col = 'name_eng'

    # Путь до xml файла
    key_name = "st_items_artefacts_" + lang
    filepath = XML_PATHS.get(key_name)

    if filepath is None:
        raise Exception(f"Не удалось найти файл: {key_name}")

    buffer = update_artifact_descriptions_in_xml(
        filepath,
        df_last,
        descr_col,
        name_col
    )

    filename = "st_items_artefacts.xml"

    return buffer, filename

### -----------------------------------------------------------------------
## ui_st_encyclopedia_artifacts.xml
# > Полный путь: gamedata\configs\text\eng\ui_st_encyclopedia_artifacts.xml

def create_description_2(row, use_english=False):
    """
    Создает объединенное описание из нескольких столбцов
    
    Parameters:
    row: строка датафрейма
    use_english: если True, использует английские столбцы, иначе русские
    
    Returns:
    str: объединенная строка описания
    """
    add_col_dict = {
        '_ADD_COL_0':"%c[d_green]Информация\n%c[ui_gray_2]",
        '_ADD_COL_1':"\n \n%c[0,200,200,200]ХАРАКТЕРИСТИКИ:%c[ui_gray_2]",
        '_ADD_COL_2':"\n%c[0,100,100,255] • %c[ui_gray_2] можно прикрепить на пояс",
    }

    add_col_dict_eng = {
        '_ADD_COL_0':"%c[d_green]Information\n%c[ui_gray_2]",
        '_ADD_COL_1':"\n \n%c[0,200,200,200]PROPERTIES:%c[ui_gray_2]",
        '_ADD_COL_2':"\n%c[0,100,100,255] • %c[ui_gray_2] attachable",
    }

    if use_english:
        col_list = ['_ADD_COL_0', 'main_description_eng', '_ADD_COL_1', 'extra_text_type_eng', 'extra_text_level_eng', 'extra_text_term_eng', '_ADD_COL_2']
    else:
        col_list = ['_ADD_COL_0', 'main_description', '_ADD_COL_1', 'extra_text_type', 'extra_text_level', 'extra_text_term', '_ADD_COL_2']

    string = ''
    for i, col in enumerate(col_list):
        if col.startswith("_ADD_COL_"):
            if use_english:
                value = add_col_dict_eng[col]
            else:
                value = add_col_dict[col]
        else:
            # Проверяем, что значение существует и не является NaN
            value = row[col] if pd.notna(row[col]) else ''
        
        if value == '':
            continue

        if i == 3:
            string += '\n' + str(value)[5:]
        elif i == 4:
            string += '\n' + str(value)[2:]
        elif i == 5:
            string += '\n' + str(value)
        else:
            string += str(value)

        if i == 5:
            continue
        
        if i == 0:
            pass
        elif i == 1:  
            string += "\n\t\t"
        elif i == 2:  
            string += "\n\t\t\n "
        elif i == 3:
            string += "\n\t\t"  # "\n\t\t\\n"
        elif i == 4:
            string += "\n\t\t"
        elif i == 5:
            string += "\n\t\t"
        else:
            string += "\n\t\t\n"
    
    return string


def update_artifact_descriptions_in_xml_2(xml_path: str, df: pd.DataFrame, descr_col: str = None, name_col: str = None) -> None:
    """
    Обновляет описание и/или названия артефактов в XML-файле на основе указанных столбцов в датафрейме.
    
    :param xml_path: Путь к исходному XML-файлу.
    :param df: DataFrame, содержащий 'artifact_id' и нужные столбцы.
    :param output_path: Путь для сохранения нового XML-файла. Если не задан, перезапишет исходный.
    :param descr_col: Название столбца с данными для обновления _descr тегов. Если None, не обновляет.
    :param name_col: Название столбца с данными для обновления _name тегов. Если None, не обновляет.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Создаем словари для каждого типа тегов
    descr_map = dict(zip(df["artifact_id"], df[descr_col])) if descr_col else {}
    name_map = dict(zip(df["artifact_id"], df[name_col])) if name_col else {}

    # Множества для отслеживания использованных и неиспользованных ID
    used_descr_ids = set()
    used_name_ids = set()
    all_descr_ids = set(descr_map.keys()) if descr_map else set()
    all_name_ids = set(name_map.keys()) if name_map else set()

    updated_descr_count = 0
    updated_name_count = 0
    
    for elem in root.findall("string"):
        id_attr = elem.attrib.get("id", "")
        if not id_attr.startswith("encyclopedia_artifacts_"):
            continue

        # Определяем тип тега и извлекаем artifact_id
        if id_attr.endswith("_text"):
            tag_type = "descr"
            # Исправлено: правильное извлечение artifact_id
            artifact_id = 'af_' + id_attr[len("encyclopedia_artifacts_"):-len("_text")]
            data_map = descr_map
            used_ids_set = used_descr_ids
        elif not id_attr.endswith("_text") and id_attr.startswith("encyclopedia_artifacts_"):
            tag_type = "name"
            artifact_id = 'af_' + id_attr[len("encyclopedia_artifacts_"):]
            data_map = name_map
            used_ids_set = used_name_ids
        else:
            continue

        # Проверяем, есть ли данные для обновления
        if not data_map or artifact_id not in data_map:
            print(f"Не нашли данные для {id_attr} (artifact_id: {artifact_id})")
            continue

        text_elem = elem.find("text")
        if text_elem is not None:
            new_text = data_map[artifact_id]
            new_text = str(new_text).strip()

            # Экранируем escape-последовательности для сохранения их как текста
            new_text = new_text.replace('\\', '\\\\')
            new_text = new_text.replace('\n', '\\n')
            new_text = new_text.replace('\t', '\\t')

            # Обновляем содержимое
            text_elem.clear()
            text_elem.text = new_text

            # Добавляем в использованные
            used_ids_set.add(artifact_id)

            # Увеличиваем соответствующий счетчик
            if tag_type == "descr":
                updated_descr_count += 1
            else:
                updated_name_count += 1

    # Логирование результатов
    print(f"✅ Обновлено encyclopedia_artifacts_*artid*_text тегов: {updated_descr_count}")
    print(f"✅ Обновлено encyclopedia_artifacts_*artid* тегов: {updated_name_count}")

    # Вывод неиспользованных строк
    unused_descr_ids = all_descr_ids - used_descr_ids
    unused_name_ids = all_name_ids - used_name_ids
    
    if unused_descr_ids:
        print(f"\n⚠️  Неиспользованные artifact_id для описаний ({len(unused_descr_ids)} шт.):")
        for unused_id in sorted(unused_descr_ids):
            print(f"   - {unused_id}")
    
    if unused_name_ids:
        print(f"\n⚠️  Неиспользованные artifact_id для названий ({len(unused_name_ids)} шт.):")
        for unused_id in sorted(unused_name_ids):
            print(f"   - {unused_id}")

    # Записываем XML в строку с XML-декларацией
    # ElementTree не добавляет декларацию при tostring, поэтому добавим вручную
    xml_str = ET.tostring(root, encoding="unicode")
    xml_with_decl = f'<?xml version="1.0" encoding="windows-1251"?>\n{xml_str}'

    xml_with_decl = clean_for_cp1251(xml_with_decl)

    # Кодируем в windows-1251 и оборачиваем в BytesIO
    buffer = io.BytesIO(xml_with_decl.encode("windows-1251"))

    return buffer

def get_ui_st_encyclopedia_artifacts(df_last, lang):
    """
    Создаёт файл локализации для пути:
    gamedata\\configs\\text\\rus\\ui_st_encyclopedia_artifacts.xml
    gamedata\\configs\\text\\eng\\ui_st_encyclopedia_artifacts.xml

    Args:
        df_last (pd.DataFrame): Данные
        lang (str): Язык из ['rus', 'eng']
    """
    if lang not in ['rus', 'eng']:
        raise Exception(f"Язык {lang} не поддерживается.")
    
    df_last = df_last.copy()
    
    if lang == 'rus':
        # Создаём новые колонки
        df_last, new_cols = add_extra_text_columns(
            df_last, 
            art_type_names, 
            art_levels_names, 
            art_temp_names,
            art_type_colors, 
            art_levels_colors, 
            art_thermo_type_colors,
            val_defs
        )

        # Для русского описания
        col_list = ['main_description'] + new_cols
        df_last['description_new'] = df_last.apply(lambda row: create_description(row, col_list), axis=1)

        # Для русского описания
        df_last['description_new_2'] = df_last.apply(create_description_2, axis=1)

        descr_col = 'description_new'
        name_col = 'name'

    elif lang == 'eng':
        df_last, new_cols_eng = add_extra_text_columns(
            df_last,
            art_type_names_en, 
            art_levels_names_en, 
            art_temp_names_en,
            art_type_colors, 
            art_levels_colors, 
            art_thermo_type_colors,
            val_defs_en,
            '_eng'
        )

        # Для английского описания
        eng_col_list = ['main_description_eng'] + new_cols_eng
        df_last['description_new_eng'] = df_last.apply(lambda row: create_description(row, eng_col_list), axis=1)

        # Для английского описания
        df_last['description_new_eng_2'] = df_last.apply(lambda row: create_description_2(row, use_english=True), axis=1)

        descr_col = 'description_new_eng_2'
        name_col = 'name_eng'

    # Путь до xml файла
    key_name = "ui_st_encyclopedia_artifacts_" + lang
    filepath = XML_PATHS.get(key_name)

    if filepath is None:
        raise Exception(f"Не удалось найти файл: {key_name}")

    # Обновить и описания, и названия:
    buffer = update_artifact_descriptions_in_xml_2(
        filepath, 
        df_last, 
        descr_col=descr_col, 
        name_col=name_col
    )

    filename = "ui_st_encyclopedia_artifacts.xml"

    return buffer, filename