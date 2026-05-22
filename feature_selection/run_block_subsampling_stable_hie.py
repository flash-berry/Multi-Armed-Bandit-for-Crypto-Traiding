"""
End-to-end pipeline для Block Subsampling Stable HIE feature selection
на 3 активах (BTCUSDT, ETHUSDT, SOLUSDT) для 3 z_windows ([24, 48, 72]).

Запуск:
    cd D:\\PythonProjects\\BTCTrading
    python feature_selection/run_block_subsampling_stable_hie.py

Что делает:
    1. Загружает OHLCV через KlinesDataLoader, фильтрует на 3 актива.
    2. Для каждого z_window:
       a. Compute features через stream_TA_lib + transform_indicators_df
          + rolling_z_score_clip_df (shift_by_one=True, без leakage).
       b. Chronological split с embargo (default 0; можно поставить 10).
       c. На train: BS-Stable-HIE через block subsampling, B=50, L=50, π=0.6.
       d. Train-only Spearman корреляционный ranking + greedy pruning.
    3. Global aggregation BS-Stable-HIE поперёк z_windows.
    4. Hybrid set builder: corr top-5 + BS-stable HIE top-5 (incremental).
    5. Save в формате, совместимом с downstream screening:
       - feature_sets_hybrid_corr_bs_stable_hie_alpha0p5.json
       - feature_set_meta_hybrid_corr_bs_stable_hie_alpha0p5.json
       - feature_sets_summary_alpha0p5.csv
       - bs_stable_hie_ranking_z{Z}.csv (per-z aggregated)
       - bs_stable_hie_per_subsample_z{Z}.csv (raw subsample-level)
       - bs_stable_hie_global_stability.csv
       - run_meta.json

Параметры (фиксируются здесь):
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]      # 3 актива
    Z_WINDOWS = [24, 48, 72]
    ALPHA_OUT = 0.5
    HORIZON = 10
    TRADE_COST = 0.0025
    EMBARGO_BARS = 0           # online-bandit постановка, pending mechanism защищает
    B (n_subsamples) = 50
    L (block_length) = 50      # ~5 × reward_horizon
    subsample_fraction = 0.5
    pi_threshold = 0.6
    main_top_k = 14
    sensitivity_grid_top_k = (7, 14, 20, 26)
    HIE: n_bins=15, n_bootstrap=30, min_bin_size=40

Compute оценка: ~50-60 минут на z_window на стандартном CPU,
total ~2.5-3 часа для всех 3 z_windows. Параллелизация по z_windows возможна
через joblib (выключено в этом скрипте для простоты).
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =====================================================================
# Paths and sys.path
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OHLCV_RELATIVE_PATH = r"data\klines_data\crypto_240m_bybit_TEST.parquet"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "feature_selection"
    / "bs_stable_hie_outputs_alpha0p5_train_only"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for p in [
    PROJECT_ROOT,
    PROJECT_ROOT / "data_processing",
    PROJECT_ROOT / "feature_selection",
    PROJECT_ROOT / "backtest",
]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# =====================================================================
# Configuration
# =====================================================================

SEED = 42
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # 3 активa
INTERVAL = 240
HORIZON = 10
TRADE_COST = 0.0025

# HIE / BS-Stable-HIE
ALPHA_OUT = 0.5
Z_WINDOWS = [24, 48, 72]

N_SUBSAMPLES_B = 50           # B
BLOCK_LENGTH_L = 50           # L: 50 баров H4 ≈ 5 * reward_horizon
SUBSAMPLE_FRACTION = 0.5
PI_THRESHOLD = 0.6
MAIN_TOP_K = 14
SENSITIVITY_GRID_TOP_K = (7, 14, 20, 26)
OVERLAP_BLOCKS = False         # non-overlapping (Politis-Romano)

N_BINS = 15
N_BOOTSTRAP = 30
MIN_BIN_SIZE = 40
MIN_ACTION_COUNT_PER_BIN = 20

# Splits
VAL_BARS = 2000
TEST_BARS = 2000
# EMBARGO_BARS = 0 для consistency с online-bandit screening/HPO postановкой:
# в pipeline future_log_ret для HIE labels вычисляется через
# add_precomputed_future_return на train_df ПОСЛЕ split (train-only), последние
# H rows train сбрасываются через dropна. Значит val/test цены НЕ используются
# для train labels — explicit embargo не требуется. Это соответствует
# методологии online-bandit ветки, где pending-update mechanism сам гарантирует,
# что boundary решения без созревшего reward не применяются.
EMBARGO_BARS = 0

# Hybrid builder
PRUNE_THRESHOLD = 0.85
CORR_METHOD = "spearman"
DEFAULT_FINAL_K = 10
DEFAULT_CORR_CORE_K = 5
DEFAULT_HIE_TARGET_K = 5

# Hybrid stability gate: feature considered stable if frequency_top_k at main_top_k
# >= PI_THRESHOLD in BS-Stable-HIE. Global gate: ≥ MIN_GLOBAL_STABILITY_Z_WINDOWS.
MIN_GLOBAL_STABILITY_Z_WINDOWS = 1

META_COLS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]

CONFIG_FOR_INDICATORS = {
    "ema_periods": [9, 21, 50, 100, 200],
    "momentum_indicators_periods": [14, 30, 72],
    "return_indicators_periods": [6, 24, 72, 168],
    "volatility_indicators_periods": [24, 72, 168],
    "level_periods": [24, 72, 168],
    "vol_ma_period": [24, 72, 168],
    "range_ma_period": [24, 72, 168],
}


# =====================================================================
# Project imports
# =====================================================================

from data_processing.functions.klines_dataloader import KlinesDataLoader
from data_processing.functions.stream_indicators import stream_TA_lib
from data_processing.functions.transform_indicators import transform_indicators_df
from data_processing.functions.rolling_z_score_clip import rolling_z_score_clip_df

from block_subsampling_stable_hie import (
    BlockSubsamplingConfig,
    HIEScoreConfig,
    BlockSubsamplingStableHIE,
    aggregate_global_stability,
)
from causal_feature_selector_hie_binary import add_precomputed_future_return

# Имя колонки precomputed future return.
# Должно быть согласовано между BS-Stable-HIE и corr-ranking target.
FUTURE_LOG_RET_COL = "future_log_ret_true"


# =====================================================================
# Data loading and feature engineering helpers
# =====================================================================


def load_ohlcv() -> pd.DataFrame:
    loader = KlinesDataLoader(symbols=SYMBOLS)
    df = loader.load_data(
        download_path=OHLCV_RELATIVE_PATH,
        analyse_data=True,
        cleaning=True,
    )
    missing = sorted(set(META_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"OHLCV missing meta columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["symbol"].isin(SYMBOLS)].copy()
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if df.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("Duplicated (symbol, timestamp) in OHLCV.")
    return df


def compute_features_for_z_window(
    ohlcv_df: pd.DataFrame, z_window: int
) -> tuple[pd.DataFrame, list[str]]:
    """Identical pipeline to existing time-stable notebook for direct comparability."""
    parts = []
    for sym, g in ohlcv_df.groupby("symbol", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True).copy()
        ind = stream_TA_lib(g, meta_cols=META_COLS, **CONFIG_FOR_INDICATORS)
        transformed = transform_indicators_df(ind, meta_cols=META_COLS)
        zdf = rolling_z_score_clip_df(
            transformed,
            meta_cols=META_COLS,
            window=z_window,
            clip_value=5.0,
            shift_by_one=True,
        )
        parts.append(zdf)
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    feature_cols = [c for c in out.columns if c not in META_COLS]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=feature_cols + META_COLS).reset_index(drop=True)
    feature_cols = [c for c in out.columns if c not in META_COLS]
    return out, feature_cols


def chronological_split_per_symbol(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True).copy()
        n = len(g)
        if n <= VAL_BARS + TEST_BARS + HORIZON + EMBARGO_BARS:
            raise ValueError(f"Too few rows for symbol={sym}: n={n}")

        test_start = n - TEST_BARS
        val_start = n - TEST_BARS - VAL_BARS
        train_end = val_start - EMBARGO_BARS
        if train_end <= HORIZON:
            raise ValueError(f"Train too short for symbol={sym}: train_end={train_end}")

        phase = np.array(["unused_gap"] * n, dtype=object)
        phase[:train_end] = "train"
        if EMBARGO_BARS > 0:
            phase[train_end:val_start] = "embargo"
        phase[val_start:test_start] = "val"
        phase[test_start:] = "test"

        g["phase"] = phase
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# =====================================================================
# Spearman corr ranking (train-only)
# =====================================================================


def make_train_corr_target(train_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for sym, g in train_df.groupby("symbol", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True).copy()
        g["future_close"] = g["close"].shift(-HORIZON)
        g["future_log_return"] = np.log(g["future_close"] / g["close"])
        g = g.replace([np.inf, -np.inf], np.nan)
        g = g.dropna(subset=["future_log_return"]).copy()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def compute_spearman_corr_ranking(
    train_df_with_target: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []
    y = train_df_with_target["future_log_return"]
    for f in feature_cols:
        if f not in train_df_with_target.columns:
            continue
        x = train_df_with_target[f]
        valid = x.notna() & y.notna()
        if valid.sum() < 20 or x[valid].nunique() <= 1:
            corr = np.nan
        else:
            corr = float(x[valid].corr(y[valid], method="spearman"))
        rows.append({
            "feature": f,
            "spearman_corr": corr,
            "abs_spearman_corr": abs(corr) if pd.notna(corr) else np.nan,
            "n_obs": int(valid.sum()),
        })
    out = pd.DataFrame(rows).dropna(subset=["abs_spearman_corr"])
    out = out.sort_values(["abs_spearman_corr", "feature"], ascending=[False, True]).reset_index(drop=True)
    out["corr_rank"] = np.arange(1, len(out) + 1)
    return out


def greedy_spearman_prune_ordered(
    candidates: list[str],
    train_feature_df: pd.DataFrame,
    threshold: float = 0.85,
    top_k: int = 10,
) -> list[str]:
    available = [f for f in candidates if f in train_feature_df.columns]
    good = []
    for c in available:
        s = train_feature_df[c]
        if s.notna().sum() >= 5 and s.nunique(dropna=True) > 1:
            good.append(c)
    if len(good) == 0:
        raise ValueError("No non-constant candidate features for pruning.")

    corr = train_feature_df[good].corr(method="spearman").abs()
    selected: list[str] = []
    for f in good:
        if len(selected) == 0:
            selected.append(f)
            continue
        corr_to_selected = corr.loc[f, selected]
        max_corr = float(corr_to_selected.max())
        if pd.isna(max_corr) or max_corr < threshold:
            selected.append(f)
        if len(selected) >= top_k:
            break
    return selected[:top_k]


# =====================================================================
# Hybrid set builder: corr_core + BS-stable HIE incremental
# =====================================================================


def build_hybrid_for_z(
    z_window: int,
    corr_ranking: pd.DataFrame,
    train_feature_df: pd.DataFrame,
    stable_hie_features_at_top_k: list[str],
    final_k: int = DEFAULT_FINAL_K,
    corr_core_k: int = DEFAULT_CORR_CORE_K,
    hie_target_k: int = DEFAULT_HIE_TARGET_K,
    prune_threshold: float = PRUNE_THRESHOLD,
) -> dict:
    """Build corr_pruned + hybrid for one z_window using BS-stable HIE features."""
    corr_ordered = corr_ranking["feature"].tolist()

    corr_pruned = greedy_spearman_prune_ordered(
        candidates=corr_ordered,
        train_feature_df=train_feature_df,
        threshold=prune_threshold,
        top_k=max(30, final_k),
    )
    if len(corr_pruned) < final_k:
        raise RuntimeError(
            f"z={z_window}: corr_pruned has {len(corr_pruned)} features, need >= {final_k}"
        )

    corr_pruned_topK = corr_pruned[:final_k]
    corr_core = corr_pruned_topK[:corr_core_k]

    # HIE incremental candidates = stable HIE not in corr_pruned_topK
    hie_pool = [f for f in stable_hie_features_at_top_k if f not in set(corr_pruned_topK)]

    # Greedy hybrid build: take corr_core, then HIE incremental, then corr fallback
    hybrid: list[str] = list(corr_core)
    n_corr_core_kept = len(hybrid)
    n_hie_incremental_kept = 0
    for f in hie_pool:
        if len(hybrid) >= corr_core_k + hie_target_k:
            break
        if f not in hybrid:
            hybrid.append(f)
            n_hie_incremental_kept += 1

    # Fallback: fill remaining slots from corr_pruned (excluding already in hybrid)
    n_corr_fallback_kept = 0
    for f in corr_pruned:
        if len(hybrid) >= final_k:
            break
        if f not in hybrid:
            hybrid.append(f)
            n_corr_fallback_kept += 1

    # Last resort: more stable HIE
    n_hie_fallback_kept = 0
    for f in stable_hie_features_at_top_k:
        if len(hybrid) >= final_k:
            break
        if f not in hybrid:
            hybrid.append(f)
            n_hie_fallback_kept += 1

    if len(hybrid) < final_k:
        raise RuntimeError(
            f"z={z_window}: hybrid has only {len(hybrid)} features, need {final_k}"
        )
    hybrid = hybrid[:final_k]

    shared = sorted(set(hybrid) & set(corr_pruned_topK))
    only_h = sorted(set(hybrid) - set(corr_pruned_topK))
    only_c = sorted(set(corr_pruned_topK) - set(hybrid))
    jacc = len(set(hybrid) & set(corr_pruned_topK)) / max(1, len(set(hybrid) | set(corr_pruned_topK)))

    return {
        "z_window": int(z_window),
        "corr_pruned": corr_pruned_topK,
        "corr_core": corr_core,
        "hie_incremental_pool": hie_pool,
        "hybrid": hybrid,
        "final_k": int(final_k),
        "corr_core_k": int(corr_core_k),
        "hie_target_k": int(hie_target_k),
        "n_corr_core_kept": int(n_corr_core_kept),
        "n_hie_incremental_kept": int(n_hie_incremental_kept),
        "n_corr_fallback_kept": int(n_corr_fallback_kept),
        "n_hie_fallback_kept": int(n_hie_fallback_kept),
        "jaccard_hybrid_vs_corr": float(jacc),
        "shared_features": shared,
        "only_hybrid": only_h,
        "only_corr": only_c,
    }


# =====================================================================
# Main pipeline
# =====================================================================


def main():
    print("=" * 80)
    print("BS-Stable-HIE feature selection pipeline")
    print(f"Symbols: {SYMBOLS}")
    print(f"Z_windows: {Z_WINDOWS}")
    print(f"B={N_SUBSAMPLES_B}, L={BLOCK_LENGTH_L}, π={PI_THRESHOLD}, main_top_k={MAIN_TOP_K}")
    print("=" * 80)

    t_start = datetime.now()

    ohlcv = load_ohlcv()
    print(f"\nOHLCV loaded: {ohlcv.shape}, symbols={sorted(ohlcv['symbol'].unique())}, "
          f"range={ohlcv['timestamp'].min()} → {ohlcv['timestamp'].max()}")

    block_cfg = BlockSubsamplingConfig(
        n_subsamples=N_SUBSAMPLES_B,
        subsample_fraction=SUBSAMPLE_FRACTION,
        block_length=BLOCK_LENGTH_L,
        pi_threshold=PI_THRESHOLD,
        main_top_k=MAIN_TOP_K,
        sensitivity_grid_top_k=SENSITIVITY_GRID_TOP_K,
        overlap_blocks=OVERLAP_BLOCKS,
        random_state=SEED,
    )
    hie_cfg = HIEScoreConfig(
        n_bins=N_BINS,
        n_bootstrap=N_BOOTSTRAP,
        min_bin_size=MIN_BIN_SIZE,
        min_action_count_per_bin=MIN_ACTION_COUNT_PER_BIN,
    )

    results_per_z: dict[int, dict] = {}
    feature_tables_by_z: dict[int, pd.DataFrame] = {}
    feature_cols_by_z: dict[int, list[str]] = {}
    corr_rankings_by_z: dict[int, pd.DataFrame] = {}

    split_audit_rows: list[dict] = []

    for z_window in Z_WINDOWS:
        print("\n" + "=" * 80)
        print(f"z_window = {z_window}")
        print("=" * 80)

        zdf, feature_cols = compute_features_for_z_window(ohlcv, z_window)
        zdf = chronological_split_per_symbol(zdf)
        train_df = zdf[zdf["phase"] == "train"].copy().reset_index(drop=True)

        for sym, g in zdf.groupby("symbol", sort=True):
            split_audit_rows.append({
                "z_window": z_window,
                "symbol": sym,
                "n_train": int((g["phase"] == "train").sum()),
                "n_val": int((g["phase"] == "val").sum()),
                "n_test": int((g["phase"] == "test").sum()),
                "n_embargo": int((g["phase"] == "embargo").sum()),
            })

        # КРИТИЧЕСКИ ВАЖНО: precompute future_log_ret ДО block subsampling.
        # На full continuous train per-symbol shift(-HORIZON) корректен; после
        # block subsampling он бы дал bogus return через временные gaps между
        # блоками. Эта строка решает данный bug.
        n_before = len(train_df)
        train_df = add_precomputed_future_return(
            train_df,
            horizon=HORIZON,
            future_log_ret_col=FUTURE_LOG_RET_COL,
            close_col="close",
            symbol_col="symbol",
            timestamp_col="timestamp",
        )
        n_after = len(train_df)

        feature_tables_by_z[z_window] = train_df
        feature_cols_by_z[z_window] = feature_cols

        print(f"  features: {len(feature_cols)}, train rows: {n_after} "
              f"(было {n_before}, dropped {n_before - n_after} последних баров per symbol "
              f"при precompute future_log_ret), "
              f"symbols: {sorted(train_df['symbol'].unique())}")

        # --- corr ranking ---
        # Используем уже precomputed future_log_ret_true для согласованности
        # с BS-Stable-HIE. Создаём alias future_log_return для совместимости с
        # существующим compute_spearman_corr_ranking.
        train_df_with_target = train_df.copy()
        train_df_with_target["future_log_return"] = train_df_with_target[FUTURE_LOG_RET_COL]
        corr_ranking = compute_spearman_corr_ranking(train_df_with_target, feature_cols)
        corr_ranking["z_window"] = z_window
        corr_rankings_by_z[z_window] = corr_ranking
        print(f"  corr top-10 raw: {corr_ranking['feature'].head(10).tolist()}")

        # --- BS-Stable-HIE ---
        selector = BlockSubsamplingStableHIE(
            block_config=block_cfg,
            hie_config=hie_cfg,
            alpha_out=ALPHA_OUT,
            horizon=HORIZON,
            trade_cost=TRADE_COST,
            future_log_ret_col=FUTURE_LOG_RET_COL,
            verbose=True,
        )
        result = selector.fit_per_z_window(
            train_df=train_df,
            feature_cols=feature_cols,
            z_window=z_window,
            symbol_col="symbol",
            random_state_offset=0,
        )
        results_per_z[z_window] = result

        # Save per-z artifacts
        result["ranking_df"].to_csv(
            OUTPUT_DIR / f"bs_stable_hie_ranking_z{z_window}.csv", index=False
        )
        result["per_subsample_df"].to_csv(
            OUTPUT_DIR / f"bs_stable_hie_per_subsample_z{z_window}.csv", index=False
        )
        result["union_full_train"].to_csv(
            OUTPUT_DIR / f"bs_stable_hie_union_full_train_z{z_window}.csv", index=False
        )
        with open(OUTPUT_DIR / f"bs_stable_hie_stable_features_z{z_window}.json", "w", encoding="utf-8") as f:
            json.dump({
                "z_window": int(z_window),
                "alpha_out": float(ALPHA_OUT),
                "stable_features_by_top_k": result["stable_features_by_top_k"],
                "mb_stability_bound_by_top_k": result["mb_stability_bound_by_top_k"],
                "bin_validity_summary": result["bin_validity_summary"],
                "config_used": result["config_used"],
                "compute_seconds": result["compute_seconds"],
            }, f, ensure_ascii=False, indent=2)

        n_stable_main = len(result["stable_features_by_top_k"][MAIN_TOP_K])
        print(f"  → stable at main_top_k={MAIN_TOP_K}, π={PI_THRESHOLD}: "
              f"{n_stable_main} features")

    # --- Global stability across z_windows ---
    print("\n" + "=" * 80)
    print("Global stability aggregation across z_windows")
    print("=" * 80)
    ranking_dfs = {z: r["ranking_df"] for z, r in results_per_z.items()}
    global_stab = aggregate_global_stability(
        ranking_dfs_by_z=ranking_dfs,
        main_top_k=MAIN_TOP_K,
        pi_threshold=PI_THRESHOLD,
    )
    global_stab.to_csv(OUTPUT_DIR / "bs_stable_hie_global_stability.csv", index=False)

    # --- Split audit ---
    pd.DataFrame(split_audit_rows).to_csv(OUTPUT_DIR / "split_audit.csv", index=False)

    # --- Hybrid set builder ---
    print("\n" + "=" * 80)
    print("Hybrid set construction (per z_window)")
    print("=" * 80)

    main_scenario_name = "hybrid_top10_corr5_bs_stable_hie5_alpha0p5"
    feature_sets: dict[str, list[str]] = {}
    feature_set_meta: dict[str, dict] = {}
    hybrid_summary_rows: list[dict] = []

    alpha_tag = "a0p5"
    for z_window in Z_WINDOWS:
        stable_features_main = results_per_z[z_window]["stable_features_by_top_k"][MAIN_TOP_K]

        # Apply global stability gate: feature must be stable in >= MIN_GLOBAL_STABILITY_Z_WINDOWS z_windows
        if MIN_GLOBAL_STABILITY_Z_WINDOWS > 1:
            global_stable_set = set(
                global_stab.loc[global_stab["n_z_windows_stable"] >= MIN_GLOBAL_STABILITY_Z_WINDOWS, "feature"]
            )
            stable_features_main = [f for f in stable_features_main if f in global_stable_set]

        hybrid_result = build_hybrid_for_z(
            z_window=z_window,
            corr_ranking=corr_rankings_by_z[z_window],
            train_feature_df=feature_tables_by_z[z_window],
            stable_hie_features_at_top_k=stable_features_main,
            final_k=DEFAULT_FINAL_K,
            corr_core_k=DEFAULT_CORR_CORE_K,
            hie_target_k=DEFAULT_HIE_TARGET_K,
            prune_threshold=PRUNE_THRESHOLD,
        )

        corr_name = f"z{z_window}_{alpha_tag}_corr_pruned_top{DEFAULT_FINAL_K}"
        hybrid_name = f"z{z_window}_{alpha_tag}_hybrid_corr{DEFAULT_CORR_CORE_K}_bsstablehie{DEFAULT_HIE_TARGET_K}_top{DEFAULT_FINAL_K}"

        feature_sets[corr_name] = hybrid_result["corr_pruned"]
        feature_sets[hybrid_name] = hybrid_result["hybrid"]

        feature_set_meta[corr_name] = {
            "set_name": corr_name,
            "family": "corr_pruned",
            "z_window": z_window,
            "alpha_out": ALPHA_OUT,
            "n_features": len(hybrid_result["corr_pruned"]),
            "prune_threshold": PRUNE_THRESHOLD,
            "features": hybrid_result["corr_pruned"],
            "notes": "Train-only Spearman ranking pruned by greedy Spearman threshold.",
        }
        feature_set_meta[hybrid_name] = {
            "set_name": hybrid_name,
            "family": "hybrid_corr_bs_stable_hie",
            "z_window": z_window,
            "alpha_out": ALPHA_OUT,
            "n_features": len(hybrid_result["hybrid"]),
            "prune_threshold": PRUNE_THRESHOLD,
            "block_subsampling": {
                "n_subsamples_B": N_SUBSAMPLES_B,
                "block_length_L": BLOCK_LENGTH_L,
                "subsample_fraction": SUBSAMPLE_FRACTION,
                "pi_threshold": PI_THRESHOLD,
                "main_top_k": MAIN_TOP_K,
                "overlap_blocks": OVERLAP_BLOCKS,
                "min_global_stability_z_windows": MIN_GLOBAL_STABILITY_Z_WINDOWS,
                "mb_stability_bound_at_main_top_k": (
                    results_per_z[z_window]["mb_stability_bound_by_top_k"][MAIN_TOP_K]
                ),
                "bin_validity_at_main_top_k": results_per_z[z_window]["bin_validity_summary"],
            },
            "corr_core_k": DEFAULT_CORR_CORE_K,
            "hie_target_k": DEFAULT_HIE_TARGET_K,
            "corr_pruned_reference": hybrid_result["corr_pruned"],
            "corr_core": hybrid_result["corr_core"],
            "hie_incremental_pool_size": len(hybrid_result["hie_incremental_pool"]),
            "n_corr_core_kept": hybrid_result["n_corr_core_kept"],
            "n_hie_incremental_kept": hybrid_result["n_hie_incremental_kept"],
            "n_corr_fallback_kept": hybrid_result["n_corr_fallback_kept"],
            "n_hie_fallback_kept": hybrid_result["n_hie_fallback_kept"],
            "hybrid_vs_corr_jaccard": hybrid_result["jaccard_hybrid_vs_corr"],
            "shared_features_with_corr": hybrid_result["shared_features"],
            "only_hybrid_vs_corr": hybrid_result["only_hybrid"],
            "only_corr_vs_hybrid": hybrid_result["only_corr"],
            "features": hybrid_result["hybrid"],
            "notes": (
                "Hybrid set: corr_core (top-K Spearman pruned) + incremental BS-stable HIE "
                "features. BS-Stable-HIE = Stability Selection (Meinshausen & Bühlmann 2010) "
                "поверх HIE filter (Zhao & Jiang 2024), реализованный через non-overlapping "
                "moving-block subsampling (Künsch 1989; Politis & Romano 1994) с длиной "
                "блока L=50 (≈5×reward_horizon), что сохраняет локальную автокорреляцию. "
                "future_log_ret precomputed на full continuous train per-symbol ДО "
                "subsampling, чтобы избежать bogus cross-block returns. Theoretical "
                "motivation — MB-stability bound; CPSS (Shah & Samworth 2013) complementary "
                "pairs не реализован, обозначен как related work."
            ),
        }

        hybrid_summary_rows.append({
            "z_window": z_window,
            "set_name_corr": corr_name,
            "set_name_hybrid": hybrid_name,
            "n_stable_hie_main_top_k": len(stable_features_main),
            "n_hie_incremental_kept": hybrid_result["n_hie_incremental_kept"],
            "jaccard_hybrid_vs_corr": hybrid_result["jaccard_hybrid_vs_corr"],
            "features_hybrid": "|".join(hybrid_result["hybrid"]),
            "features_corr": "|".join(hybrid_result["corr_pruned"]),
        })

        print(f"\n  z={z_window}: {hybrid_name}")
        print(f"    hybrid: {hybrid_result['hybrid']}")
        print(f"    corr5 / bs-hie5 split: {hybrid_result['n_corr_core_kept']} / "
              f"{hybrid_result['n_hie_incremental_kept']} (+fallback corr={hybrid_result['n_corr_fallback_kept']}, "
              f"+fallback hie={hybrid_result['n_hie_fallback_kept']})")

    # --- Save feature sets in screening-compatible format ---
    with open(OUTPUT_DIR / "feature_sets_hybrid_corr_bs_stable_hie_alpha0p5.json", "w", encoding="utf-8") as f:
        json.dump(feature_sets, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "feature_set_meta_hybrid_corr_bs_stable_hie_alpha0p5.json", "w", encoding="utf-8") as f:
        json.dump(feature_set_meta, f, ensure_ascii=False, indent=2)

    summary_df = pd.DataFrame([
        {
            "set_name": name,
            "family": meta["family"],
            "z_window": meta["z_window"],
            "alpha_out": meta["alpha_out"],
            "n_features": meta["n_features"],
            "features": "|".join(meta["features"]),
        }
        for name, meta in feature_set_meta.items()
    ])
    summary_df.to_csv(OUTPUT_DIR / "feature_sets_summary_alpha0p5.csv", index=False)

    pd.DataFrame(hybrid_summary_rows).to_csv(
        OUTPUT_DIR / "hybrid_summary_per_z.csv", index=False
    )

    # --- Run meta ---
    t_end = datetime.now()
    run_meta = {
        "started_at": t_start.isoformat(),
        "finished_at": t_end.isoformat(),
        "duration_seconds": (t_end - t_start).total_seconds(),
        "symbols": SYMBOLS,
        "z_windows": Z_WINDOWS,
        "alpha_out": ALPHA_OUT,
        "horizon": HORIZON,
        "trade_cost": TRADE_COST,
        "val_bars": VAL_BARS,
        "test_bars": TEST_BARS,
        "embargo_bars": EMBARGO_BARS,
        "block_subsampling": {
            "n_subsamples_B": N_SUBSAMPLES_B,
            "block_length_L": BLOCK_LENGTH_L,
            "subsample_fraction": SUBSAMPLE_FRACTION,
            "pi_threshold": PI_THRESHOLD,
            "main_top_k": MAIN_TOP_K,
            "sensitivity_grid_top_k": list(SENSITIVITY_GRID_TOP_K),
            "overlap_blocks": OVERLAP_BLOCKS,
        },
        "hie": {
            "n_bins": N_BINS,
            "n_bootstrap": N_BOOTSTRAP,
            "min_bin_size": MIN_BIN_SIZE,
            "min_action_count_per_bin": MIN_ACTION_COUNT_PER_BIN,
        },
        "hybrid": {
            "final_k": DEFAULT_FINAL_K,
            "corr_core_k": DEFAULT_CORR_CORE_K,
            "hie_target_k": DEFAULT_HIE_TARGET_K,
            "prune_threshold": PRUNE_THRESHOLD,
            "min_global_stability_z_windows": MIN_GLOBAL_STABILITY_Z_WINDOWS,
        },
    }
    with open(OUTPUT_DIR / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"DONE. Total duration: {(t_end - t_start).total_seconds():.0f}s")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Feature sets ready for downstream screening: ")
    print(f"  {OUTPUT_DIR / 'feature_sets_hybrid_corr_bs_stable_hie_alpha0p5.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
