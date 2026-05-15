import numpy as np


def rolling_z_score_clip_df(
    df,
    meta_cols,
    window,
    clip_value,
    shift_by_one=False,
):
    """
    Векторный rolling z-score для DataFrame.

    shift_by_one=False:
        mean/std считаются по окну [t-window+1 : t].
        Это соответствует логике: текущая закрытая свеча уже известна.

    shift_by_one=True:
        mean/std считаются только по прошлым значениям [t-window : t-1].
        Это более строгий anti-leakage режим. Для backtesting/feature selection
        обычно предпочтителен именно он.

    Колонки с suffix _bounded, _scaled, _norm уже приведены к устойчивой шкале
    и не z-score'ятся повторно.
    """
    if df.empty:
        raise ValueError("Передан пустой DataFrame")

    if window is None or int(window) <= 1:
        raise ValueError(f"window должен быть int > 1, получено: {window}")

    if clip_value is None or float(clip_value) <= 0:
        raise ValueError(f"clip_value должен быть > 0, получено: {clip_value}")

    window = int(window)
    clip_value = float(clip_value)

    missing_meta_cols = [col for col in meta_cols if col not in df.columns]
    if missing_meta_cols:
        raise ValueError(f"Нет мета-колонок: {missing_meta_cols}")

    forbidden_suffixes = ("_bounded", "_scaled", "_norm")

    out = df[meta_cols].copy()

    feature_cols = [col for col in df.columns if col not in meta_cols]

    for col in feature_cols:
        if col.endswith(forbidden_suffixes):
            out[col] = df[col]
            continue

        x = df[col].astype(float)

        rolling_mean = x.rolling(
            window=window,
            min_periods=window,
        ).mean()

        rolling_std = x.rolling(
            window=window,
            min_periods=window,
        ).std(ddof=1)

        if shift_by_one:
            rolling_mean = rolling_mean.shift(1)
            rolling_std = rolling_std.shift(1)

        z = (x - rolling_mean) / (rolling_std + 1e-6)

        out[f"{col}_z"] = z.clip(-clip_value, clip_value)

    return out
