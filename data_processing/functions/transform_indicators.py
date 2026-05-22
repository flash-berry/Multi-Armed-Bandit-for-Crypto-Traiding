import numpy as np


# Indicator categories used before rolling z-score.
# Absolute/non-stationary intermediate columns are skipped explicitly in map_indicator_category.
transform_dict = {
    "bounded": (
        "RSI",
    ),
    "scaled": (
        "ADX",
    ),
    "log1p": (
        "vol_ratio",
        "vol_expansion",
    ),
    "sqrt": (
        "bb_width",
        "volatility",
        "body",
        "range",
        "dist_to_high",
        "dist_to_low",
    ),
    "norm": (
        "bb_pos",
        "price_pos_in_range",
    ),
    "raw": (
        "MACD_hist_pct",
        "MOM_pct",
        "NATR",
        "dist_ema",
        "log_ret",
        "ema_slope",
        "ema_spread",
        "ret_accel",
    ),
}


def map_indicator_category(name):
    """
    Возвращает категорию трансформации для индикатора.

    skip означает, что колонка является промежуточной или слишком нестационарной
    для линейного MAB и не должна попадать в feature space:
        - ema_{p}: абсолютные EMA-levels;
        - vol_ma_{p}: absolute moving average объёма;
        - range_ma_{p}: absolute moving average диапазона.
    """
    if name.startswith("ema_") and not name.startswith(("ema_slope", "ema_spread")):
        return "skip"

    if name.startswith("vol_ma"):
        return "skip"

    if name.startswith("range_ma"):
        return "skip"

    # Safety: old absolute price-unit indicators should not silently pass through.
    if name.startswith("MOM_") and not name.startswith("MOM_pct"):
        return "skip"

    if name == "MACD_hist":
        return "skip"

    for category, prefixes in transform_dict.items():
        if name.startswith(prefixes):
            return category

    return None


def transform_indicators_df(df, meta_cols):
    if df.empty:
        raise ValueError("Передан пустой DataFrame")

    missing_meta_cols = [col for col in meta_cols if col not in df.columns]
    if missing_meta_cols:
        raise ValueError(f"Нет мета-колонок: {missing_meta_cols}")

    results = df[meta_cols].copy()

    for col in df.columns:
        if col in meta_cols:
            continue

        cat = map_indicator_category(col)

        if cat == "skip":
            continue

        if cat is None:
            raise ValueError(f"Для индикатора {col} не найдено трансформации")

        x = df[col].astype(float)

        if cat == "bounded":
            # RSI: 0..100 -> approximately -1..1 around neutral 50.
            results[f"{col}_bounded"] = (x - 50.0) / 50.0

        elif cat == "scaled":
            # ADX: trend strength. 25 is a common practical threshold.
            results[f"{col}_scaled"] = (x - 25.0) / 25.0

        elif cat == "log1p":
            # Positive ratio-like features. log1p dampens right tails.
            results[f"{col}_log1p"] = np.log1p(np.maximum(x, 0.0))

        elif cat == "sqrt":
            # Positive magnitude features. sqrt dampens tails while preserving order.
            results[f"{col}_sqrt"] = np.sqrt(np.maximum(x, 0.0))

        elif cat == "norm":
            # Position indicators usually live in [0, 1]. Center around zero.
            results[f"{col}_norm"] = x - 0.5

        elif cat == "raw":
            # Already dimensionless or naturally signed financial features.
            results[col] = x

        else:
            raise ValueError(f"Неизвестная категория трансформации: {cat}")

    return results
