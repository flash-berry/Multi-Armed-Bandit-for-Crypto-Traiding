"""
Статистика по closed trades в screening results.

Closed trade = entry + exit pair = 1 SELL event в trade_log_val.
Используем колонку n_sells_val из screening_results_per_run.csv.

Note: n_buys_val ≥ n_sells_val потому что bandit может оставить позицию
открытой в конце validation. Поэтому closed_trades = n_sells_val.

Запуск:
    cd D:\\PythonProjects\\BTCTrading
    python feature_selection/trades_stats_screening.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENING_DIR = (
    PROJECT_ROOT
    / "feature_selection"
    / "bs_stable_hie_outputs_alpha0p5_train_only"
    / "screening"
)
INPUT_CSV = SCREENING_DIR / "screening_results_per_run.csv"


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows (per (algo × set × z × seed × symbol))")

    # Add set_family helper
    df["set_family"] = df["set_name"].apply(
        lambda s: "hybrid" if "hybrid" in s else ("corr" if "corr_pruned" in s else "unknown")
    )

    # Closed trades = SELL events (entry+exit pair)
    df["closed_trades_val"] = df["n_sells_val"]
    # Detect open position at end of validation (n_buys > n_sells means позиция осталась)
    df["open_position_at_end"] = (df["n_buys_val"] - df["n_sells_val"]).astype(int)

    # ============================================================
    # 1. Overall statistics
    # ============================================================
    print("\n" + "=" * 90)
    print("OVERALL closed trades statistics (per symbol per seed per config)")
    print("=" * 90)
    desc = df["closed_trades_val"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    print(desc.to_string())
    print(f"\nDecisions per validation: {int(df['n_decisions_val'].median())}")
    print(f"Median validation length per symbol: {int(df['n_decisions_val'].median())} decision bars")
    print(f"Open positions at end of val (count > 0): "
          f"{int((df['open_position_at_end'] > 0).sum())} of {len(df)} rows "
          f"({100*(df['open_position_at_end'] > 0).mean():.1f}%)")

    # ============================================================
    # 2. Per algorithm × set_family
    # ============================================================
    print("\n" + "=" * 90)
    print("Per (algorithm × set_family): closed trades distribution per symbol per seed")
    print("=" * 90)
    grp = df.groupby(["algorithm", "set_family"], as_index=False).agg(
        n_runs=("closed_trades_val", "count"),
        median_closed_trades=("closed_trades_val", "median"),
        mean_closed_trades=("closed_trades_val", "mean"),
        std_closed_trades=("closed_trades_val", "std"),
        min_closed_trades=("closed_trades_val", "min"),
        max_closed_trades=("closed_trades_val", "max"),
        median_buys=("n_buys_val", "median"),
        median_sells=("n_sells_val", "median"),
        median_decisions=("n_decisions_val", "median"),
        median_exposure_ratio=("exposure_ratio_val", "median"),
    ).sort_values(["algorithm", "set_family"]).reset_index(drop=True)
    print(grp.to_string(index=False))

    # ============================================================
    # 3. Per algorithm × set × z_window (aggregated across seeds and symbols)
    # ============================================================
    print("\n" + "=" * 90)
    print("Per (algorithm × set × z_window): closed trades summary")
    print("  Aggregated across seeds AND symbols (median, q25, q75)")
    print("=" * 90)
    detail = df.groupby(["algorithm", "set_name", "z_window"], as_index=False).agg(
        n_runs=("closed_trades_val", "count"),
        median_closed_trades=("closed_trades_val", "median"),
        q25_closed_trades=("closed_trades_val", lambda s: float(np.percentile(s, 25))),
        q75_closed_trades=("closed_trades_val", lambda s: float(np.percentile(s, 75))),
        mean_closed_trades=("closed_trades_val", "mean"),
        median_decisions=("n_decisions_val", "median"),
        trades_per_decision_rate=(
            "closed_trades_val",
            lambda s: float(s.mean() / df.loc[s.index, "n_decisions_val"].mean())
            if df.loc[s.index, "n_decisions_val"].mean() > 0 else 0.0,
        ),
    ).sort_values(["algorithm", "median_closed_trades"], ascending=[True, False]).reset_index(drop=True)
    print(detail.to_string(index=False))

    # ============================================================
    # 4. Total trades across full screening
    # ============================================================
    print("\n" + "=" * 90)
    print("TOTAL closed trades across all screening runs")
    print("=" * 90)
    total_closed_trades = int(df["closed_trades_val"].sum())
    total_buys = int(df["n_buys_val"].sum())
    total_sells = int(df["n_sells_val"].sum())
    print(f"  Total BUY events:        {total_buys:,}")
    print(f"  Total SELL events:       {total_sells:,}")
    print(f"  Total CLOSED trades:     {total_closed_trades:,}")
    print(f"  Open positions at end:   {total_buys - total_sells:,}")

    n_unique_configs = df.groupby(["algorithm", "set_name", "z_window"]).ngroups
    n_unique_seeds = df.groupby(["algorithm", "set_name", "z_window", "seed"]).ngroups
    n_unique_runs = df.groupby(["algorithm", "set_name", "z_window", "seed", "symbol"]).ngroups
    print(f"\n  Unique (algo, set, z) configs: {n_unique_configs}")
    print(f"  Unique (algo, set, z, seed) seeds: {n_unique_seeds}")
    print(f"  Unique (algo, set, z, seed, symbol) runs: {n_unique_runs}")
    print(f"  Avg closed trades per run: {total_closed_trades / max(1, n_unique_runs):.1f}")

    # ============================================================
    # 5. Per symbol breakdown
    # ============================================================
    print("\n" + "=" * 90)
    print("Per symbol: closed trades distribution")
    print("=" * 90)
    sym_grp = df.groupby("symbol", as_index=False).agg(
        n_runs=("closed_trades_val", "count"),
        median_closed_trades=("closed_trades_val", "median"),
        mean_closed_trades=("closed_trades_val", "mean"),
        std_closed_trades=("closed_trades_val", "std"),
        median_exposure_ratio=("exposure_ratio_val", "median"),
        median_constraint_applied=("constraint_applied_ratio", "median"),
    ).reset_index(drop=True)
    print(sym_grp.to_string(index=False))

    # ============================================================
    # 6. Hybrid vs Corr comparison (closed trades)
    # ============================================================
    print("\n" + "=" * 90)
    print("Hybrid vs Corr: closed trades comparison")
    print("=" * 90)
    pivot = (
        df.groupby(["algorithm", "z_window", "set_family"], as_index=False)
          .agg(median_closed=("closed_trades_val", "median"),
               mean_closed=("closed_trades_val", "mean"))
    )
    pivot_med = pivot.pivot_table(
        index=["algorithm", "z_window"], columns="set_family", values="median_closed"
    ).reset_index()
    if "corr" in pivot_med.columns and "hybrid" in pivot_med.columns:
        pivot_med["hybrid_minus_corr_trades"] = pivot_med["hybrid"] - pivot_med["corr"]
        pivot_med = pivot_med.rename(columns={"corr": "corr_median_trades",
                                                "hybrid": "hybrid_median_trades"})
    print(pivot_med.to_string(index=False))

    print("\n" + "=" * 90)
    print("FILES SAVED:")
    print("=" * 90)
    out_grp = SCREENING_DIR / "trades_stats_per_algo_set_z.csv"
    detail.to_csv(out_grp, index=False)
    print(f"  {out_grp}")
    out_sym = SCREENING_DIR / "trades_stats_per_symbol.csv"
    sym_grp.to_csv(out_sym, index=False)
    print(f"  {out_sym}")
    out_hvc = SCREENING_DIR / "trades_stats_hybrid_vs_corr.csv"
    pivot_med.to_csv(out_hvc, index=False)
    print(f"  {out_hvc}")


if __name__ == "__main__":
    main()
