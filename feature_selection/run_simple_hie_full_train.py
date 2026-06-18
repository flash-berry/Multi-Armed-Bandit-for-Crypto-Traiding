"""
Simple HIE-only feature selection on full train data.

Purpose:
    Create a separate, non-overwriting set of artifacts for comparing ordinary
    one-shot HIE with BS-Stable-HIE.

Run from project root:
    cd D:\\PythonProjects\\BTCTrading
    python feature_selection/run_simple_hie_full_train.py

Default output:
    feature_selection/hie_simple_outputs_alpha0p5_train_only/<timestamp>/

The script does NOT overwrite existing BS-Stable-HIE artifacts. By default it
creates a new timestamped subfolder on every run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =====================================================================
# Paths
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [
    PROJECT_ROOT,
    PROJECT_ROOT / "data_processing",
    PROJECT_ROOT / "feature_selection",
    PROJECT_ROOT / "backtest",
]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse the exact same data/feature/split constants and helpers as the
# BS-Stable-HIE runner. This makes ordinary HIE and BS-Stable-HIE comparable.
from run_block_subsampling_stable_hie import (  # noqa: E402
    load_ohlcv,
    compute_features_for_z_window,
    chronological_split_per_symbol,
    SYMBOLS,
    Z_WINDOWS,
    HORIZON,
    TRADE_COST,
    ALPHA_OUT,
    FUTURE_LOG_RET_COL,
    MAIN_TOP_K,
    SENSITIVITY_GRID_TOP_K,
    N_BINS,
    N_BOOTSTRAP,
    MIN_BIN_SIZE,
    MIN_ACTION_COUNT_PER_BIN,
    EMBARGO_BARS,
    VAL_BARS,
    TEST_BARS,
)
from causal_feature_selector_hie_binary import (  # noqa: E402
    CausalBanditFeatureSelector,
    FeatureSelectionConfig,
    add_precomputed_future_return,
    make_direct_counterfactual_log_binary_success_from_precomputed,
)
from block_subsampling_stable_hie import build_hie_union  # noqa: E402


# =====================================================================
# Utility functions
# =====================================================================


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def make_output_dir(base_dir: Path, use_timestamp: bool = True) -> Path:
    """Create output directory without overwriting previous runs."""
    if use_timestamp:
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        out = base_dir / stamp
        out.mkdir(parents=True, exist_ok=False)
        return out

    # Non-timestamp mode is allowed only if the folder is absent or empty.
    if base_dir.exists() and any(base_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {base_dir}\n"
            "Use default timestamped mode or pass another --output-base."
        )
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def fit_hie_for_regime(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    prev_position: int,
    random_state: int,
) -> pd.DataFrame:
    """Run ordinary HIE on full train for one trading regime."""
    cf = make_direct_counterfactual_log_binary_success_from_precomputed(
        train_df,
        feature_cols=feature_cols,
        trade_cost=TRADE_COST,
        alpha_out=ALPHA_OUT,
        prev_position=prev_position,
        future_log_ret_col=FUTURE_LOG_RET_COL,
        action_col_name="raw_action",
        random_state=random_state,
        horizon=HORIZON,
    )

    cfg = FeatureSelectionConfig(
        n_bins=N_BINS,
        n_bootstrap=N_BOOTSTRAP,
        min_bin_size=MIN_BIN_SIZE,
        min_action_count_per_bin=MIN_ACTION_COUNT_PER_BIN,
        random_state=random_state,
    )
    selector = CausalBanditFeatureSelector(cfg)
    scores = selector.fit(
        cf,
        feature_cols=feature_cols,
        action_col="raw_action",
        reward_col="reward",
    )
    scores = scores.copy()
    scores["prev_position"] = int(prev_position)
    scores["regime"] = "entry" if prev_position == 0 else "exit_hold"
    return scores


def add_union_topk_flags(union_df: pd.DataFrame, topk_grid: tuple[int, ...]) -> pd.DataFrame:
    out = union_df.copy()
    for k in topk_grid:
        out[f"in_top_{int(k)}"] = out["union_rank"] <= int(k)
    return out


def aggregate_global_simple_hie(union_by_z: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate one-shot HIE ranks across z-windows for comparison diagnostics."""
    rows = []
    for z, union in union_by_z.items():
        for _, r in union.iterrows():
            rows.append(
                {
                    "z_window": int(z),
                    "feature": r["feature"],
                    "union_rank": int(r["union_rank"]),
                    "best_rank": float(r["best_rank"]),
                    "best_hie_norm": float(r["best_hie_norm"]),
                    "best_p_value": float(r["best_p_value"]),
                    "regime_coverage_topk": int(r["regime_coverage_topk"]),
                    "best_regime_source": r["best_regime_source"],
                    "in_main_top_k": bool(r["union_rank"] <= MAIN_TOP_K),
                }
            )
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df

    agg = (
        long_df.groupby("feature", as_index=False)
        .agg(
            n_z_windows_seen=("z_window", "nunique"),
            z_windows_seen=("z_window", lambda s: "|".join(map(str, sorted(s.unique())))),
            n_z_windows_in_main_top_k=("in_main_top_k", "sum"),
            best_union_rank_any_z=("union_rank", "min"),
            median_union_rank=("union_rank", "median"),
            mean_union_rank=("union_rank", "mean"),
            max_best_hie_norm=("best_hie_norm", "max"),
            mean_best_hie_norm=("best_hie_norm", "mean"),
            min_best_p_value=("best_p_value", "min"),
        )
        .sort_values(
            [
                "n_z_windows_in_main_top_k",
                "best_union_rank_any_z",
                "max_best_hie_norm",
                "min_best_p_value",
                "feature",
            ],
            ascending=[False, True, False, True, True],
        )
        .reset_index(drop=True)
    )
    agg["simple_hie_global_rank"] = np.arange(1, len(agg) + 1)
    agg["main_top_k"] = int(MAIN_TOP_K)
    return agg


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ordinary one-shot HIE on full train and save artifacts separately."
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=str(PROJECT_ROOT / "feature_selection" / "hie_simple_outputs_alpha0p5_train_only"),
        help="Base output directory. Default: feature_selection/hie_simple_outputs_alpha0p5_train_only",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Write directly to --output-base. Fails if directory is not empty.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for HIE bootstrap null.",
    )
    args = parser.parse_args()

    t_start = datetime.now()
    out_dir = make_output_dir(Path(args.output_base), use_timestamp=not args.no_timestamp)

    print("=" * 90)
    print("Simple HIE-only feature selection on full train")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir:   {out_dir}")
    print(f"Symbols:      {SYMBOLS}")
    print(f"Z windows:    {Z_WINDOWS}")
    print(f"HORIZON:      {HORIZON}")
    print(f"ALPHA_OUT:    {ALPHA_OUT}")
    print(f"TRADE_COST:   {TRADE_COST}")
    print(f"HIE params:   bins={N_BINS}, bootstrap={N_BOOTSTRAP}, min_bin={MIN_BIN_SIZE}, min_action={MIN_ACTION_COUNT_PER_BIN}")
    print("=" * 90)

    ohlcv = load_ohlcv()
    print(
        f"\nOHLCV loaded: {ohlcv.shape}, "
        f"range={ohlcv['timestamp'].min()} → {ohlcv['timestamp'].max()}\n"
    )

    union_by_z: dict[int, pd.DataFrame] = {}
    split_audit_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []

    for z in Z_WINDOWS:
        print("=" * 90)
        print(f"z_window = {z}")
        print("=" * 90)
        t_z = time.time()

        zdf, feature_cols = compute_features_for_z_window(ohlcv, z)
        zdf = chronological_split_per_symbol(zdf)
        train_df = zdf[zdf["phase"] == "train"].copy().reset_index(drop=True)

        for sym, g in zdf.groupby("symbol", sort=True):
            split_audit_rows.append(
                {
                    "z_window": int(z),
                    "symbol": sym,
                    "n_train_before_future_drop": int((g["phase"] == "train").sum()),
                    "n_val": int((g["phase"] == "val").sum()),
                    "n_test": int((g["phase"] == "test").sum()),
                    "n_embargo": int((g["phase"] == "embargo").sum()),
                }
            )

        n_train_before = len(train_df)
        train_df = add_precomputed_future_return(
            train_df,
            horizon=HORIZON,
            future_log_ret_col=FUTURE_LOG_RET_COL,
            close_col="close",
            symbol_col="symbol",
            timestamp_col="timestamp",
        )
        n_train_after = len(train_df)

        print(
            f"  features={len(feature_cols)}, train rows={n_train_after} "
            f"(before future drop={n_train_before}, dropped={n_train_before - n_train_after})"
        )

        # Ordinary HIE on full train: entry and exit/hold regimes.
        print("  Running HIE for entry regime (prev_position=0) ...")
        entry_scores = fit_hie_for_regime(
            train_df=train_df,
            feature_cols=feature_cols,
            prev_position=0,
            random_state=args.seed + 1000 * int(z) + 1,
        )
        print("  Running HIE for exit/hold regime (prev_position=1) ...")
        exit_scores = fit_hie_for_regime(
            train_df=train_df,
            feature_cols=feature_cols,
            prev_position=1,
            random_state=args.seed + 1000 * int(z) + 2,
        )

        union = build_hie_union(
            entry_scores=entry_scores,
            exit_scores=exit_scores,
            stability_top_k=MAIN_TOP_K,
        )
        union = add_union_topk_flags(union, SENSITIVITY_GRID_TOP_K)
        union_by_z[int(z)] = union

        # Save per-z artifacts.
        entry_scores.to_csv(out_dir / f"simple_hie_entry_scores_z{z}.csv", index=False)
        exit_scores.to_csv(out_dir / f"simple_hie_exit_scores_z{z}.csv", index=False)
        union.to_csv(out_dir / f"simple_hie_union_ranking_z{z}.csv", index=False)

        top_features_by_top_k = {
            str(int(k)): union.head(int(k))["feature"].tolist()
            for k in SENSITIVITY_GRID_TOP_K
        }
        with open(out_dir / f"simple_hie_top_features_z{z}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "z_window": int(z),
                    "alpha_out": float(ALPHA_OUT),
                    "horizon": int(HORIZON),
                    "trade_cost": float(TRADE_COST),
                    "main_top_k": int(MAIN_TOP_K),
                    "top_features_by_top_k": top_features_by_top_k,
                    "top_main_features": union.head(MAIN_TOP_K)["feature"].tolist(),
                    "n_features_total": int(len(feature_cols)),
                    "n_train_rows_after_future_drop": int(n_train_after),
                },
                f,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )

        for k in SENSITIVITY_GRID_TOP_K:
            for rank, f_name in enumerate(union.head(int(k))["feature"].tolist(), start=1):
                topk_rows.append(
                    {
                        "z_window": int(z),
                        "top_k": int(k),
                        "rank_within_top_k": int(rank),
                        "feature": f_name,
                    }
                )

        print(f"  Top-{MAIN_TOP_K} simple HIE features:")
        for i, f_name in enumerate(union.head(MAIN_TOP_K)["feature"].tolist(), start=1):
            print(f"    {i:02d}. {f_name}")
        print(f"  Saved z={z} artifacts in {time.time() - t_z:.1f}s\n")

    # Global summaries.
    topk_long = pd.DataFrame(topk_rows)
    topk_long.to_csv(out_dir / "simple_hie_topk_long.csv", index=False)

    split_audit = pd.DataFrame(split_audit_rows)
    split_audit.to_csv(out_dir / "split_audit_simple_hie.csv", index=False)

    global_summary = aggregate_global_simple_hie(union_by_z)
    global_summary.to_csv(out_dir / "simple_hie_global_summary.csv", index=False)

    # Manifest for downstream comparison.
    t_end = datetime.now()
    run_meta = {
        "run_type": "simple_one_shot_hie_full_train",
        "started_at": t_start.isoformat(),
        "finished_at": t_end.isoformat(),
        "duration_seconds": float((t_end - t_start).total_seconds()),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(out_dir),
        "symbols": SYMBOLS,
        "z_windows": Z_WINDOWS,
        "alpha_out": float(ALPHA_OUT),
        "horizon": int(HORIZON),
        "trade_cost": float(TRADE_COST),
        "future_log_ret_col": FUTURE_LOG_RET_COL,
        "embargo_bars": int(EMBARGO_BARS),
        "val_bars": int(VAL_BARS),
        "test_bars": int(TEST_BARS),
        "main_top_k": int(MAIN_TOP_K),
        "sensitivity_grid_top_k": [int(x) for x in SENSITIVITY_GRID_TOP_K],
        "hie": {
            "n_bins": int(N_BINS),
            "n_bootstrap": int(N_BOOTSTRAP),
            "min_bin_size": int(MIN_BIN_SIZE),
            "min_action_count_per_bin": int(MIN_ACTION_COUNT_PER_BIN),
            "seed": int(args.seed),
        },
        "notes": (
            "Ordinary one-shot HIE is computed once on the full train split for each z-window. "
            "It uses the same feature engineering, chronological split, precomputed future_log_ret, "
            "counterfactual payoff semantics, entry/exit union ranking, and HIE parameters as the "
            "BS-Stable-HIE pipeline, but WITHOUT block subsampling and WITHOUT selection-frequency filtering. "
            "Artifacts are saved into a separate timestamped folder to avoid overwriting existing data."
        ),
    }
    with open(out_dir / "simple_hie_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2, default=_json_default)

    print("=" * 90)
    print("DONE")
    print(f"Outputs saved to: {out_dir}")
    print("Main files:")
    print("  simple_hie_union_ranking_z{24,48,72}.csv")
    print("  simple_hie_entry_scores_z{24,48,72}.csv")
    print("  simple_hie_exit_scores_z{24,48,72}.csv")
    print("  simple_hie_top_features_z{24,48,72}.json")
    print("  simple_hie_topk_long.csv")
    print("  simple_hie_global_summary.csv")
    print("  simple_hie_run_meta.json")
    print("=" * 90)


if __name__ == "__main__":
    main()
