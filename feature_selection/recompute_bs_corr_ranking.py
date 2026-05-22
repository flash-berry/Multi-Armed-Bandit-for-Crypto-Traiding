"""
Standalone пересчёт Spearman corr ranking + greedy Spearman pruning на тех же
3 активах и с теми же splits, что использовались в BS-stable HIE pipeline.

Не запускает HIE (это часы). Только feature engineering + corr ranking + pruning,
~5-10 минут total на 3 z_windows.

Запуск:
    cd D:\\PythonProjects\\BTCTrading
    python feature_selection/recompute_bs_corr_ranking.py

Выходы в feature_selection/bs_stable_hie_outputs_alpha0p5_train_only/:
    bs_corr_ranking_z{Z}.csv         — full Spearman corr ranking (все 54 признака,
                                       с corr_rank, abs_spearman_corr, n_obs).
    bs_corr_pruning_audit_z{Z}.csv   — greedy Spearman pruning decisions (selected /
                                       dropped, reason, max corr with already-selected,
                                       most correlated already-selected feature).
    bs_corr_pruned_top30_z{Z}.json   — финальный pruned top-30 список + threshold.

Используется тот же подход что и в BS-stable runner:
- pooled Spearman corr на конкатенированном train (BTC+ETH+SOL).
- greedy Spearman pruning по corr_rank order с threshold 0.85.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [
    PROJECT_ROOT,
    PROJECT_ROOT / "data_processing",
    PROJECT_ROOT / "feature_selection",
    PROJECT_ROOT / "backtest",
]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Импортируем helper'ы из BS-stable runner — там уже всё что нужно
from run_block_subsampling_stable_hie import (
    load_ohlcv,
    compute_features_for_z_window,
    chronological_split_per_symbol,
    compute_spearman_corr_ranking,
    OUTPUT_DIR,
    Z_WINDOWS,
    HORIZON,
    FUTURE_LOG_RET_COL,
    PRUNE_THRESHOLD,
    SYMBOLS,
)
from causal_feature_selector_hie_binary import add_precomputed_future_return


def greedy_spearman_prune_with_audit(
    candidates: list[str],
    train_feature_df: pd.DataFrame,
    threshold: float,
    audit_top_n: int = 30,
) -> tuple[list[str], pd.DataFrame]:
    """
    Greedy Spearman pruning с полным audit'ом decisions для top-N селекций.

    На каждой итерации feature из ordered candidates либо selected (если max
    corr с уже выбранными < threshold), либо dropped. Audit фиксирует решение,
    причину, max corr с selected, и какой именно selected feature был наиболее
    скоррелирован.

    Останавливается после audit_top_n selected (записывает решения для всех
    candidates пока не наберёт audit_top_n).

    Returns:
        selected: список selected features (ровно min(audit_top_n, len(candidates)))
        diag_df: DataFrame с decisions для всех просмотренных candidates
    """
    available = [c for c in candidates if c in train_feature_df.columns]
    good = [
        c for c in available
        if train_feature_df[c].notna().sum() >= 5
        and train_feature_df[c].nunique(dropna=True) > 1
    ]
    if not good:
        raise ValueError("No good candidates after NA/constant filter.")

    corr_mat = train_feature_df[good].corr(method="spearman").abs()

    selected: list[str] = []
    diag_rows: list[dict] = []

    for f in good:
        if len(selected) == 0:
            selected.append(f)
            diag_rows.append({
                "feature": f,
                "decision": "selected",
                "reason": "first_feature",
                "max_abs_corr_with_selected": float("nan"),
                "most_correlated_selected_feature": None,
            })
        else:
            corr_to_selected = corr_mat.loc[f, selected]
            max_corr = float(corr_to_selected.max())
            most_corr_f = str(corr_to_selected.idxmax())
            if pd.isna(max_corr) or max_corr < threshold:
                selected.append(f)
                diag_rows.append({
                    "feature": f,
                    "decision": "selected",
                    "reason": "below_threshold",
                    "max_abs_corr_with_selected": max_corr,
                    "most_correlated_selected_feature": most_corr_f,
                })
            else:
                diag_rows.append({
                    "feature": f,
                    "decision": "dropped",
                    "reason": "too_correlated",
                    "max_abs_corr_with_selected": max_corr,
                    "most_correlated_selected_feature": most_corr_f,
                })

        if len(selected) >= audit_top_n:
            break

    diag_df = pd.DataFrame(diag_rows)
    return selected, diag_df


def main():
    print("=" * 80)
    print("Recompute BS-stable corr ranking (3 symbols, same splits as BS-stable HIE)")
    print(f"Symbols: {SYMBOLS}")
    print(f"Z_windows: {Z_WINDOWS}")
    print(f"Prune threshold: {PRUNE_THRESHOLD}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ohlcv = load_ohlcv()
    print(f"\nOHLCV loaded: {ohlcv.shape}, symbols={sorted(ohlcv['symbol'].unique())}\n")

    for z_window in Z_WINDOWS:
        print("=" * 80)
        print(f"z_window = {z_window}")
        print("=" * 80)

        # Feature engineering + split (exact мirror BS-stable runner)
        zdf, feature_cols = compute_features_for_z_window(ohlcv, z_window)
        zdf = chronological_split_per_symbol(zdf)
        train_df = zdf[zdf["phase"] == "train"].copy().reset_index(drop=True)

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
        print(f"  train rows: {n_after} (было {n_before}, dropped {n_before - n_after} "
              f"last bars per symbol при precompute future_log_ret)")

        # Alias для compute_spearman_corr_ranking ожидает 'future_log_return'
        train_df_with_target = train_df.copy()
        train_df_with_target["future_log_return"] = train_df_with_target[FUTURE_LOG_RET_COL]

        # === Full Spearman corr ranking (pooled across BTC/ETH/SOL) ===
        corr_ranking = compute_spearman_corr_ranking(
            train_df_with_target, feature_cols
        )
        corr_ranking["z_window"] = z_window
        out_ranking = OUTPUT_DIR / f"bs_corr_ranking_z{z_window}.csv"
        corr_ranking.to_csv(out_ranking, index=False)
        print(f"  Saved: {out_ranking.name} ({len(corr_ranking)} features)")
        print(f"  corr top-10 raw: {corr_ranking['feature'].head(10).tolist()}")

        # === Greedy Spearman pruning audit ===
        candidates_ordered = corr_ranking["feature"].tolist()
        pruned, prune_diag = greedy_spearman_prune_with_audit(
            candidates=candidates_ordered,
            train_feature_df=train_df,
            threshold=PRUNE_THRESHOLD,
            audit_top_n=30,
        )
        prune_diag["z_window"] = z_window
        out_pruning = OUTPUT_DIR / f"bs_corr_pruning_audit_z{z_window}.csv"
        prune_diag.to_csv(out_pruning, index=False)
        print(f"  Saved: {out_pruning.name} ({len(prune_diag)} decisions, "
              f"{(prune_diag['decision']=='selected').sum()} selected, "
              f"{(prune_diag['decision']=='dropped').sum()} dropped)")
        print(f"  corr_pruned top-10: {pruned[:10]}")

        # === Final pruned list ===
        out_json = OUTPUT_DIR / f"bs_corr_pruned_top30_z{z_window}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "z_window": int(z_window),
                "symbols": SYMBOLS,
                "horizon": HORIZON,
                "future_log_ret_col": FUTURE_LOG_RET_COL,
                "prune_threshold": float(PRUNE_THRESHOLD),
                "n_features_total": int(len(corr_ranking)),
                "pruned_top30": pruned[:30],
                "pruned_top10": pruned[:10],
                "pruned_top5": pruned[:5],
            }, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_json.name}\n")

    print("=" * 80)
    print(f"DONE. Outputs in: {OUTPUT_DIR}")
    print("Files per z_window: bs_corr_ranking_z{Z}.csv, "
          "bs_corr_pruning_audit_z{Z}.csv, bs_corr_pruned_top30_z{Z}.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
