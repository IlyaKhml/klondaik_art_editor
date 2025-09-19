import pandas as pd

# Определение категорий характеристик
IMMUNITY_COLS = [
'wound_immunity_abs',
'strike_immunity_abs',
'explosion_immunity_abs',
'fire_wound_immunity_abs',
'chemical_burn_immunity_abs',
'burn_immunity_abs',
'shock_immunity_abs',
'telepatic_immunity_abs',
'radiation_immunity_abs',
'br_class_artefact_main',
]

CAP_COLS = [
'wound_cap_main',
'wound_abs_cap_main',
'strike_cap_main',
'strike_abs_cap_main',
'explosion_cap_main',
'explosion_abs_cap_main',
'fire_wound_cap_main',
'fire_wound_abs_cap_main',
'chemical_burn_cap_main',
'chemical_burn_abs_cap_main',
'burn_cap_main',
'burn_abs_cap_main',
'shock_cap_main',
'shock_abs_cap_main',
'telepatic_cap_main',
'telepatic_abs_cap_main',
'psy_health_cap_main',
'psy_health_abs_cap_main',
]

RESTORE_COLS = [
'bleeding_restore_speed_hard_main',
'bleeding_restore_speed_main',
'health_restore_speed_main',
'power_restore_speed_main',
'radiation_restore_speed_main',
'satiety_restore_speed_main',
"eat_satiety_main",
'eat_thirstiness_main',
'eat_sleepiness_main',
'psy_health_restore_speed_main'
]

# Расширенная категоризация характеристик
UTILITY_COLS = [
'speed_modifier_main',
'dizziness_main',
'inv_weight_main', # Вес артефакта 
'additional_inventory_weight_main',  # Бонусный вес, который может переносить персонаж
'additional_inventory_weight2_main', # Тоже самое что и "additional_inventory_weight_main", только для другой игровой системы.
]

ALL_STAT_COLS = IMMUNITY_COLS + CAP_COLS + RESTORE_COLS + UTILITY_COLS

def get_stats_weights():
    STATS_WEIGHT = {
        'inv_weight_main':              -1.5,
        'additional_inventory_weight_main':     0.75,
        'additional_inventory_weight2_main':    0.75,

        'bleeding_restore_speed_main':  0.015,
        "bleeding_restore_speed_hard_main": -0.0225,
        'health_restore_speed_main':    3.0,
        'power_restore_speed_main':     0.3,
        'satiety_restore_speed_main':   150,
        "eat_satiety_main":             1,
        "eat_thirstiness_main":         1,
        "eat_sleepiness_main":          1.5,
        'radiation_restore_speed_main': -3.0,  # знак в значении будет учтён
        "psy_health_restore_speed_main": 2.5,

        'radiation_immunity_abs':       1.0,
        'shock_immunity_abs':           0.5,
        'telepatic_immunity_abs':       0.8,
        'wound_immunity_abs':           1.1,
        'chemical_burn_immunity_abs':   0.9,
        'burn_immunity_abs':            0.9,
        'explosion_immunity_abs':       1.05,
        'strike_immunity_abs':          1.15,
        'fire_wound_immunity_abs':      1.3,

        'telepatic_cap_main':           100,  # Пси предел
        'wound_cap_main':               100,  # Предел разрыва
        'strike_cap_main':              100,  # Предел удара
        'shock_cap_main':               100,  # Предел электричества
        'chemical_burn_cap_main':       100,  # Предел хим ожога
        'burn_cap_main':                100,  # Предел ожога
        'explosion_cap_main':           100,  # Предел взрыва
        'fire_wound_cap_main':          110,  # Баллистический предел
        "psy_health_cap_main":          150,  # Предел пси-здоровья

        "br_class_artefact_main":       5,
        "speed_modifier_main":          2.5,
        "dizziness_main":               -2.5,
    }

    STAT_ABS_WEIGHT = {
        "psy_health_abs_cap_main":      250, 
        "telepatic_abs_cap_main":       200, 
        "fire_wound_abs_cap_main":      225,
        "wound_abs_cap_main":           200, 
        "strike_abs_cap_main":          200, 
        "explosion_abs_cap_main":       200,
        "burn_abs_cap_main":            200, 
        "shock_abs_cap_main":           200, 
        "chemical_burn_abs_cap_main":   200,
    }

    return STATS_WEIGHT, STAT_ABS_WEIGHT

# STATS_WEIGHT, STAT_ABS_WEIGHT = get_stats_weights()

def compute_artifact_scores(df: pd.DataFrame, weights, abs_weigths) -> pd.DataFrame:
    df = df.copy()

    pos_scores = []
    neg_scores = []

    def_limit = 0.65 # ДЕФОЛТНЫЙ ЛИМИТ ДЛЯ ХАРАКТЕРИСТИК
    def_limit_neg = 1 - def_limit # ОБРАТНЫЙ ДЕФОЛТНЫЙ ЛИМИТ ДЛЯ ХАРАКТЕРИСТИК

    for _, row in df.iterrows():
        pos = 0
        neg = 0
        for col, weight in weights.items():
            val = row.get(col, 0)
            if pd.isna(val):
                continue
            score = val * weight
            if score > 0:
                pos += score
            elif score < 0:
                neg += abs(score)
        
        for col, weight in abs_weigths.items():
            val = row.get(col, 0)
            if pd.isna(val) or val == 0:
                continue
            
            # Другая логика для abs_weights
            if col == "psy_health_abs_cap_main":
                if val > 0:
                    score = val * weight
                else:
                    score = ((val + 1) * -1) * weight

                if score > 0:
                    pos += score
                elif score < 0:
                    neg += abs(score)
            else:
                if val > 0:
                    # Делим на 2 части: до def_limit и после
                    def_val = val - def_limit
                    if def_val > 0:
                        # Сверх лимита добавляет больше ценности
                        score = (def_val * weight * 1.25) + (def_limit * weight)
                    else:
                        score = val * weight
                else:
                    # Негативное значение
                    def_val = ((val + 1) * -1) + def_limit_neg
                    if def_val < 0:
                        # Ниже лимита добавляет ещё меньше ценности
                        score = (def_val * weight) + (def_limit_neg * weight * 0.75)
                    else:
                        # Умножаем на 0.75, т.к. убавление нижи ниже или равно базовому лимиту
                        score = ((val + 1) * -1) * weight * 0.75

                if score > 0:
                    pos += score
                elif score < 0:
                    neg += abs(score)                          

        pos_scores.append(pos)
        neg_scores.append(neg)

    df['positive_score'] = pos_scores
    df['negative_score'] = neg_scores
    df['total_score'] = df['positive_score'] - df['negative_score']

    return df

