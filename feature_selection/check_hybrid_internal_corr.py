"""
Проверка внутренней корреляции hybrid feature sets для подтверждения,
что hybrid сам по себе не содержит pairs с |Spearman corr| > 0.85,
даже если некоторые HIE-incremental признаки были dropped в corr_pruning
из-за корреляции с признаками, которые НЕ попали в hybrid.

Запуск:
    cd D:\\PythonProjects\\BTCTrading
    python feature_selection/check_hybrid_internal_corr.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "data_processing", PROJECT_ROOT / "feature_selection"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_block_subsampling_stable_hie import (
    load_ohlcv,
    compute_features_for_z_window,
    chronological_split_per_symbol,
    HORIZON,
    FUTURE_LOG_RET_COL,
    PRUNE_THRESHOLD,
    OUTPUT_DIR,
    Z_WINDOWS,
)
from causal_feature_selector_hie_binary import add_precomputed_future_return


def main():
    with open(OUTPUT_DIR / "feature_sets_hybrid_corr_bs_stable_hie_alpha0p5.json") as f:
        feature_sets = json.load(f)

    ohlcv = load_ohlcv()

    summary_rows = []

    for z in Z_WINDOWS:
        print("=" * 100)
        print(f"z_window = {z}")
        print("=" * 100)

        zdf, _ = compute_features_for_z_window(ohlcv, z)
        zdf = chronological_split_per_symbol(zdf)
        train_df = zdf[zdf["phase"] == "train"].copy().reset_index(drop=True)
        train_df = add_precomputed_future_return(
            train_df,
            horizon=HORIZON,
            future_log_ret_col=FUTURE_LOG_RET_COL,
        )

        for set_key in [
            f"z{z}_a0p5_corr_pruned_top10",
            f"z{z}_a0p5_hybrid_corr5_bsstablehie5_top10",
        ]:
            features = feature_sets[set_key]
            available = [f for f in features if f in train_df.columns]
            if len(available) < len(features):
                missing = sorted(set(features) - set(available))
                print(f"  WARN: missing features in train_df: {missing}")

            corr_mat = train_df[available].corr(method="spearman").abs()

            # Save full matrix
            out_csv = OUTPUT_DIR / f"internal_corr_matrix_{set_key}.csv"
            corr_mat.to_csv(out_csv)

            # Find max off-diagonal correlation.
            # pandas 3.x returns read-only ndarray view, копируем для записи.
            mat = corr_mat.values.copy()
            np.fill_diagonal(mat, 0.0)
            max_off = float(mat.max())
            i, j = np.unravel_index(mat.argmax(), mat.shape)
            max_pair = (corr_mat.index[i], corr_mat.columns[j])

            # All pairs above threshold
            pairs_above = []
            for ii in range(len(available)):
                for jj in range(ii + 1, len(available)):
                    c = corr_mat.iloc[ii, jj]
                    if c > 0.5:
                        pairs_above.append((available[ii], available[jj], c))
            pairs_above.sort(key=lambda x: -x[2])

            print(f"\n  {set_key}")
            print(f"    n_features: {len(available)}")
            print(f"    max off-diagonal |corr|: {max_off:.3f} "
                  f"({max_pair[0]} ↔ {max_pair[1]})")
            print(f"    pairs with |corr| > 0.5: {len(pairs_above)}")
            print(f"    pairs with |corr| > {PRUNE_THRESHOLD}: "
                  f"{sum(1 for _, _, c in pairs_above if c > PRUNE_THRESHOLD)}")
            for f1, f2, c in pairs_above[:10]:
                marker = "  ⚠ ABOVE 0.85" if c > PRUNE_THRESHOLD else ""
                print(f"      |corr|={c:.3f}  {f1} ↔ {f2}{marker}")

            summary_rows.append({
                "set_name": set_key,
                "z_window": z,
                "n_features": len(available),
                "max_off_diagonal_corr": max_off,
                "max_pair_a": max_pair[0],
                "max_pair_b": max_pair[1],
                "n_pairs_above_0p5": len(pairs_above),
                "n_pairs_above_threshold": sum(1 for _, _, c in pairs_above if c > PRUNE_THRESHOLD),
                "prune_threshold": PRUNE_THRESHOLD,
            })

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "internal_corr_summary.csv", index=False
    )
    print("\n" + "=" * 100)
    print("Summary saved:")
    print(f"  {OUTPUT_DIR / 'internal_corr_summary.csv'}")
    print(f"  + internal_corr_matrix_*.csv per feature set")
    print("=" * 100)


if __name__ == "__main__":
    main()
