import numpy as np
import talib as ta


_EPS = 1e-9


def _as_sorted_periods(value, name):
    """
    Нормализует параметр периода к sorted list[int].

    Поддерживает обратную совместимость:
        vol_ma_period=24
        vol_ma_period=[24, 72, 168]
    """
    if value is None:
        raise ValueError(f"{name} не должен быть None")

    if isinstance(value, int):
        periods = [value]
    else:
        try:
            periods = list(value)
        except TypeError as exc:
            raise TypeError(f"{name} должен быть int или iterable[int]") from exc

    periods = sorted({int(p) for p in periods})

    if not periods:
        raise ValueError(f"{name} не должен быть пустым")

    if any(p <= 0 for p in periods):
        raise ValueError(f"Все периоды в {name} должны быть > 0: {periods}")

    return periods


def stream_TA_lib(
    df,
    meta_cols,
    ema_periods=None,
    momentum_indicators_periods=None,
    return_indicators_periods=None,
    volatility_indicators_periods=None,
    level_periods=None,
    vol_ma_period=None,
    range_ma_period=None,
):
    """
    Векторный production-style расчёт технических индикаторов для crypto contextual bandit.

    Основные принципы feature engineering:
        1. Абсолютные EMA-level, vol_ma и range_ma считаются только как промежуточные
           величины для derived-признаков и затем отсекаются в transform_indicators_df.
        2. MOM и MACD переведены в percentage / dimensionless форму, чтобы убрать
           зависимость от абсолютного уровня цены и улучшить cross-asset стабильность.
        3. vol_ma_period и range_ma_period поддерживают как int, так и список периодов.
    """

    ema_periods = _as_sorted_periods(ema_periods or [], "ema_periods")
    momentum_indicators_periods = _as_sorted_periods(
        momentum_indicators_periods or [],
        "momentum_indicators_periods",
    )
    return_indicators_periods = _as_sorted_periods(
        return_indicators_periods or [],
        "return_indicators_periods",
    )
    volatility_indicators_periods = _as_sorted_periods(
        volatility_indicators_periods or [],
        "volatility_indicators_periods",
    )
    level_periods = _as_sorted_periods(level_periods or [], "level_periods")

    vol_ma_periods = _as_sorted_periods(vol_ma_period, "vol_ma_period")
    range_ma_periods = _as_sorted_periods(range_ma_period, "range_ma_period")

    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    missing_meta_cols = [col for col in meta_cols if col not in df.columns]
    if missing_meta_cols:
        raise ValueError(f"Нет мета-колонок: {missing_meta_cols}")

    required_price_cols = ["open", "high", "low", "close", "volume"]
    missing_price_cols = [col for col in required_price_cols if col not in df.columns]
    if missing_price_cols:
        raise ValueError(f"Нет OHLCV-колонок: {missing_price_cols}")

    open_p = df["open"].astype(float)
    high_p = df["high"].astype(float)
    low_p = df["low"].astype(float)
    close_p = df["close"].astype(float)
    volume = df["volume"].astype(float)

    results = df[meta_cols].copy()

    # ============== EMA / trend distance ==============
    ema_cache = {}

    for p in ema_periods:
        ema = ta.EMA(close_p, timeperiod=p)
        ema_cache[p] = ema

        # Raw EMA values are intentionally kept only as intermediate/output diagnostics.
        # transform_indicators_df skips ema_{p}; usable features are dist/slope/spread.
        results[f"ema_{p}"] = ema
        results[f"dist_ema_{p}"] = (close_p - ema) / (ema + _EPS)
        results[f"ema_slope_{p}"] = (ema / (ema.shift(6) + _EPS)) - 1.0

    for i in range(len(ema_periods) - 1):
        fast_p = ema_periods[i]
        slow_p = ema_periods[i + 1]

        fast = ema_cache[fast_p]
        slow = ema_cache[slow_p]

        results[f"ema_spread_{fast_p}_{slow_p}"] = (fast / (slow + _EPS)) - 1.0

    # ============== Momentum ==============
    for p in momentum_indicators_periods:
        mom_abs = ta.MOM(close_p, timeperiod=p)

        # Dimensionless momentum. Old absolute MOM_{p} is intentionally not emitted.
        results[f"MOM_pct_{p}"] = mom_abs / (close_p + _EPS)

        results[f"ADX_{p}"] = ta.ADX(high_p, low_p, close_p, timeperiod=p)
        results[f"RSI_{p}"] = ta.RSI(close_p, timeperiod=p)

    # ============== MACD ==============
    # TA-Lib MACD_hist is in absolute price units. Use percentage MACD instead.
    ema_fast_12 = ta.EMA(close_p, timeperiod=12)
    ema_slow_26 = ta.EMA(close_p, timeperiod=26)
    macd_line_pct = (ema_fast_12 - ema_slow_26) / (close_p + _EPS)
    signal_line_pct = ta.EMA(macd_line_pct, timeperiod=9)
    results["MACD_hist_pct"] = macd_line_pct - signal_line_pct

    # ============== Volatility ==============
    results["NATR_14"] = ta.NATR(high_p, low_p, close_p, timeperiod=14)

    upper, middle, lower = ta.BBANDS(
        close_p,
        timeperiod=20,
        nbdevup=2,
        nbdevdn=2,
    )

    results["bb_width_20"] = (upper - lower) / (middle + _EPS)
    results["bb_pos_20"] = (close_p - lower) / (upper - lower + _EPS)

    log_ret1 = np.log(close_p / close_p.shift(1))

    for p in volatility_indicators_periods:
        results[f"volatility_{p}"] = log_ret1.rolling(window=p).std(ddof=1)

    # ============== Returns ==============
    for p in return_indicators_periods:
        results[f"log_ret_{p}"] = np.log(close_p / close_p.shift(p))

    for i in range(len(return_indicators_periods) - 1):
        fast_p = return_indicators_periods[i]
        slow_p = return_indicators_periods[i + 1]

        fast = np.log(close_p / close_p.shift(fast_p))
        slow = np.log(close_p / close_p.shift(slow_p))

        results[f"ret_accel_{fast_p}_{slow_p}"] = fast - slow

    # ============== Volume ==============
    # vol_ma is emitted for diagnostics/intermediate use, but skipped as a final feature.
    for p in vol_ma_periods:
        vol_ma = volume.rolling(window=p).mean()
        results[f"vol_ma_{p}"] = vol_ma
        results[f"vol_ratio_{p}"] = volume / (vol_ma + _EPS)

    # ============== Candle / range ==============
    range_series = (high_p - low_p) / (close_p + _EPS)

    results["range"] = range_series
    results["body"] = (close_p - open_p).abs() / (close_p + _EPS)

    # range_ma is emitted for diagnostics/intermediate use, but skipped as a final feature.
    for p in range_ma_periods:
        range_ma = range_series.rolling(window=p).mean()
        results[f"range_ma_{p}"] = range_ma
        results[f"vol_expansion_{p}"] = range_series / (range_ma + _EPS)

    # ============== Distance to local high / low ==============
    for p in level_periods:
        rolling_high = high_p.rolling(window=p).max()
        rolling_low = low_p.rolling(window=p).min()

        results[f"dist_to_high_{p}"] = (rolling_high - close_p) / (close_p + _EPS)
        results[f"dist_to_low_{p}"] = (close_p - rolling_low) / (close_p + _EPS)
        results[f"price_pos_in_range_{p}"] = (
            (close_p - rolling_low) / (rolling_high - rolling_low + _EPS)
        )

    return results
