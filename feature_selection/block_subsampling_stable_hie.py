"""
Block Subsampling Stable HIE (BS-Stable-HIE).

Stability Selection поверх HIE-фильтра Zhao & Jiang (2024), применённого к
нестационарным финансовым данным через moving-block subsampling.

Методологический контекст:

1. HIE (Zhao & Jiang, 2024) — model-free filter для contextual MAB, оценивающий
   способность признака менять локально лучший arm/action через bin-разбиение.
   Базовый метод валидирован на стационарных синтетических данных и онлайн
   recommender; авторы в Conclusion указывают non-stationary environments как
   открытое направление будущей работы.

2. Stability Selection (Meinshausen & Bühlmann, 2010) — общая парадигма
   повышения устойчивости любого feature selector через повторные запуски на
   подвыборках train data с агрегацией selection frequency. Theorem 1 даёт
   finite-sample upper bound на ожидаемое число false positives:

        E[#{j : V_j ≥ π} ∩ {θ_j = 0}] ≤ q² / ((2π - 1) · p)

   где V_j — selection frequency признака j, π > 0.5 — threshold, q — top-K,
   p — общее число признаков. Bound выводится при определённых условиях,
   включая exchangeability наблюдений. В нестационарных автокоррелированных
   финансовых time series exchangeability не выполняется напрямую, поэтому
   bound используется как theoretical motivation, не как формальная гарантия
   false-discovery rate.

3. CPSS (Shah & Samworth, 2013) — refinement Stability Selection через
   complementary pairs subsampling. Даёт более тугие bounds под более слабыми
   предположениями (через r-concavity условия). В данной работе CPSS-стратегия
   complementary pairs НЕ реализована; используется обычное block subsampling
   без replacement. CPSS обозначен как related work и направление возможного
   будущего улучшения; именования полей в коде соответственно используют
   Meinshausen-Bühlmann bound, а не CPSS bound.

4. Block bootstrap / block subsampling (Künsch, 1989; Politis & Romano, 1994) —
   адаптация resampling-методов для time series, сохраняющая локальную
   автокорреляционную структуру через выбор блоков последовательных observations.

5. Subsampling-safe future return (важный методологический нюанс):
   При обычной обработке HIE counterfactual log делается shift(-horizon) внутри
   df. На block subsample с дискретными блоками это приводит к bogus future
   return через временные gaps между блоками. BS-Stable-HIE требует
   precomputed future_log_ret на полном непрерывном train ДО subsampling,
   через `add_precomputed_future_return(...)` и
   `make_direct_counterfactual_log_binary_success_from_precomputed(...)`.

Алгоритм BS-Stable-HIE (формализация):

    Input: train data D, feature set F = {f_1, ..., f_p}, alpha_out α,
           B (n_subsamples), L (block length), q (top-K),
           π (threshold), n_bins, n_bootstrap_internal.
    For b = 1 to B:
        S_b = non-overlapping moving-block subsample of D, block length L,
              target size ≈ |D|/2, sampled per-symbol independently to
              preserve symbol-level chronology.
        For each prev_position ∈ {0, 1}:
            CF_b,prev = direct counterfactual log on S_b for regime prev.
            HIE_b,prev = HIE ranking via CausalBanditFeatureSelector
                         with n_bootstrap shuffles for internal null normalization.
        Union_b = best-rank union of HIE_b,0 and HIE_b,1 with regime-coverage tie-break.
        T_b,q = top-q features of Union_b.
    For each f ∈ F and each q ∈ sensitivity grid:
        V_{f,q} = |{b : f ∈ T_b,q}| / B.
    Stable_q = {f : V_{f,q} ≥ π}.

Совместимость с downstream pipeline:

    Выход BS-Stable-HIE имеет тот же информационный профиль, что и time-stable
    HIE (stable_features_by_top_k, per-feature ranking, sensitivity grid).
    Поэтому существующий hybrid builder (corr_core + stable_hie incremental)
    можно использовать без изменений, заменив источник stable features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import time

import numpy as np
import pandas as pd

# Локальный импорт: модуль предполагается частью пакета feature_selection
# в проекте BTCTrading. При запуске из родительского каталога путь обеспечивается
# через sys.path.insert в run-скрипте.
from causal_feature_selector_hie_binary import (
    CausalBanditFeatureSelector,
    FeatureSelectionConfig,
    add_precomputed_future_return,
    make_direct_counterfactual_log_binary_success_from_precomputed,
)


# =====================================================================
# Configuration dataclasses
# =====================================================================


@dataclass
class BlockSubsamplingConfig:
    """Конфигурация stability selection поверх HIE."""

    n_subsamples: int = 50
    """B — число block-subsampling итераций. Стандарт в литературе 50–100."""

    subsample_fraction: float = 0.5
    """Доля train, выбираемая в subsample (per-symbol). 0.5 — рекомендация
    Meinshausen-Bühlmann."""

    block_length: int = 50
    """L — длина последовательного блока баров. Должна превышать
    autocorrelation scale данных. Для H4 крипто-данных при reward_horizon=10
    разумный диапазон 30–100 баров."""

    pi_threshold: float = 0.6
    """π — порог selection frequency для определения stable feature.
    Meinshausen-Bühlmann (2010) stability bound требует π > 0.5 для positive bound."""

    main_top_k: int = 14
    """Основной q (top-K) для отбора кандидатов; матчит MAIN_HIE_STABILITY_TOPK
    из существующего pipeline для совместимости."""

    sensitivity_grid_top_k: tuple[int, ...] = (7, 14, 20, 26)
    """Sensitivity grid для top-K, как в существующем time-stable approach."""

    overlap_blocks: bool = False
    """False = non-overlapping moving blocks (Politis-Romano subsampling).
    True = overlapping moving blocks. Non-overlapping — более консервативный
    выбор для stability selection."""

    random_state: int = 42


@dataclass
class HIEScoreConfig:
    """Параметры внутреннего HIE selector. Должны соответствовать базовому
    pipeline для прямой сопоставимости результатов."""

    n_bins: int = 15
    n_bootstrap: int = 100
    min_bin_size: int = 40
    min_action_count_per_bin: int = 20


# =====================================================================
# Internal helpers
# =====================================================================


def _safe_int(v) -> int:
    """Convert v to int, returning -1 for NaN/None — для diagnostic полей,
    где NaN означает 'feature отсутствует на одной из сторон union outer-merge'."""
    if v is None:
        return -1
    try:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return -1
        return int(v)
    except (ValueError, TypeError):
        return -1


def _safe_str(v) -> str:
    """Convert v to str, returning 'absent' for NaN/None."""
    if v is None:
        return "absent"
    try:
        if isinstance(v, float) and np.isnan(v):
            return "absent"
        return str(v)
    except (ValueError, TypeError):
        return "absent"


# =====================================================================
# Block subsampling utility
# =====================================================================


def block_subsample_indices_per_symbol(
    df: pd.DataFrame,
    symbol_col: str,
    block_length: int,
    subsample_fraction: float,
    rng: np.random.Generator,
    overlap: bool = False,
) -> np.ndarray:
    """
    Block subsampling индексов c сохранением symbol-level chronology.

    Algorithm (non-overlapping moving blocks subsampling):
        For each symbol s:
            Partition rows of s into ⌊n_s / L⌋ non-overlapping blocks of length L.
            Sample ⌊n_blocks_total * fraction⌋ blocks without replacement.
            Concatenate row indices of selected blocks (in original time order).
        Return concatenated indices across all symbols.

    При overlap=True используются overlapping blocks (start positions выбираются
    случайно из [0, n - L]).

    Returns:
        np.ndarray of integer indices into df (0-based, contiguous index assumed).
    """
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if not (0.0 < subsample_fraction <= 1.0):
        raise ValueError("subsample_fraction must be in (0, 1]")

    parts: list[np.ndarray] = []
    for sym in sorted(df[symbol_col].unique()):
        sym_idx = df.index[df[symbol_col] == sym].to_numpy()
        n = len(sym_idx)

        if n < block_length:
            # Слишком мало данных в этом символе для одного блока — пропускаем.
            continue

        if overlap:
            max_start = n - block_length
            n_blocks_target = max(1, int(round(n * subsample_fraction / block_length)))
            starts = rng.integers(0, max_start + 1, size=n_blocks_target)
            block_rows = []
            for s in starts:
                block_rows.extend(sym_idx[s:s + block_length].tolist())
        else:
            n_total_blocks = n // block_length
            if n_total_blocks == 0:
                continue
            n_blocks_target = max(1, int(round(n_total_blocks * subsample_fraction)))
            n_blocks_target = min(n_blocks_target, n_total_blocks)
            block_ids = rng.choice(n_total_blocks, size=n_blocks_target, replace=False)
            block_ids = np.sort(block_ids)
            block_rows = []
            for bid in block_ids:
                start = bid * block_length
                end = start + block_length
                block_rows.extend(sym_idx[start:end].tolist())
        parts.append(np.array(block_rows, dtype=np.int64))

    if not parts:
        raise ValueError(
            f"Empty subsample: block_length={block_length} is too large for available "
            f"per-symbol sizes."
        )
    return np.concatenate(parts)


# =====================================================================
# HIE union (matches main pipeline semantics)
# =====================================================================


def build_hie_union(
    entry_scores: pd.DataFrame,
    exit_scores: pd.DataFrame,
    stability_top_k: int = 14,
) -> pd.DataFrame:
    """
    Объединяет entry/exit HIE rankings, как в основном pipeline.

    Sort priority:
        1. best_rank = min(entry_rank, exit_rank), ascending
        2. regime_coverage_topk, descending
        3. best_hie_norm, descending
        4. best_p_value, ascending
        5. feature name, ascending

    Это идентично build_hie_union_unpruned из notebook'а
    hybrid_hie_corr_stability_pipeline_alpha0p5_v4 для прямой совместимости
    union-семантики между time-stable и BS-stable approaches.

    Дополнительно сохраняются bin diagnostics (n_samples, n_bins_used,
    min_bin_size_used, min_action_count_in_bin) для entry и exit как
    proof-of-validity HIE на subsamples.
    """
    diag_cols = [
        "n_samples", "n_bins_used", "min_bin_size_used",
        "min_action_count_in_bin", "reason",
    ]
    e = entry_scores[["feature", "hie_rank", "hie_norm", "hie_p_value", *diag_cols]].copy()
    x = exit_scores[["feature", "hie_rank", "hie_norm", "hie_p_value", *diag_cols]].copy()

    e = e.rename(columns={
        "hie_rank": "entry_rank",
        "hie_norm": "entry_hie_norm",
        "hie_p_value": "entry_p_value",
        "n_samples": "entry_n_samples",
        "n_bins_used": "entry_n_bins_used",
        "min_bin_size_used": "entry_min_bin_size_used",
        "min_action_count_in_bin": "entry_min_action_count_in_bin",
        "reason": "entry_reason",
    })
    x = x.rename(columns={
        "hie_rank": "exit_rank",
        "hie_norm": "exit_hie_norm",
        "hie_p_value": "exit_p_value",
        "n_samples": "exit_n_samples",
        "n_bins_used": "exit_n_bins_used",
        "min_bin_size_used": "exit_min_bin_size_used",
        "min_action_count_in_bin": "exit_min_action_count_in_bin",
        "reason": "exit_reason",
    })

    union = e.merge(x, on="feature", how="outer")

    for col in ["entry_rank", "exit_rank"]:
        union[col] = union[col].astype(float)
    union["entry_rank_filled"] = union["entry_rank"].fillna(np.inf)
    union["exit_rank_filled"] = union["exit_rank"].fillna(np.inf)
    union["best_rank"] = union[["entry_rank_filled", "exit_rank_filled"]].min(axis=1)
    union["mean_rank"] = union[["entry_rank_filled", "exit_rank_filled"]].replace(np.inf, np.nan).mean(axis=1)

    union["entry_hie_norm_filled"] = union["entry_hie_norm"].fillna(-np.inf)
    union["exit_hie_norm_filled"] = union["exit_hie_norm"].fillna(-np.inf)
    union["best_hie_norm"] = union[["entry_hie_norm_filled", "exit_hie_norm_filled"]].max(axis=1)

    union["entry_p_value_filled"] = union["entry_p_value"].fillna(1.0)
    union["exit_p_value_filled"] = union["exit_p_value"].fillna(1.0)
    union["best_p_value"] = union[["entry_p_value_filled", "exit_p_value_filled"]].min(axis=1)

    union["regime_coverage_topk"] = (
        (union["entry_rank_filled"] <= stability_top_k).astype(int)
        + (union["exit_rank_filled"] <= stability_top_k).astype(int)
    )

    def _source(row: pd.Series) -> str:
        if row["entry_rank_filled"] < row["exit_rank_filled"]:
            return "entry"
        if row["exit_rank_filled"] < row["entry_rank_filled"]:
            return "exit"
        return "both_tie"

    union["best_regime_source"] = union.apply(_source, axis=1)

    union = union.sort_values(
        ["best_rank", "regime_coverage_topk", "best_hie_norm", "best_p_value", "mean_rank", "feature"],
        ascending=[True, False, False, True, True, True],
    ).reset_index(drop=True)
    union["union_rank"] = np.arange(1, len(union) + 1)

    drop_cols = [
        "entry_rank_filled", "exit_rank_filled",
        "entry_hie_norm_filled", "exit_hie_norm_filled",
        "entry_p_value_filled", "exit_p_value_filled",
    ]
    union = union.drop(columns=[c for c in drop_cols if c in union.columns])
    return union


# =====================================================================
# Main class
# =====================================================================


class BlockSubsamplingStableHIE:
    """
    Stability selection over HIE via block subsampling for non-stationary
    financial contextual MAB feature selection.
    """

    def __init__(
        self,
        block_config: BlockSubsamplingConfig,
        hie_config: HIEScoreConfig,
        alpha_out: float,
        horizon: int,
        trade_cost: float,
        future_log_ret_col: str = "future_log_ret_true",
        verbose: bool = True,
    ):
        self.block_config = block_config
        self.hie_config = hie_config
        self.alpha_out = float(alpha_out)
        self.horizon = int(horizon)
        self.trade_cost = float(trade_cost)
        self.future_log_ret_col = str(future_log_ret_col)
        self.verbose = bool(verbose)
        self._rng = np.random.default_rng(block_config.random_state)

    def fit_per_z_window(
        self,
        train_df: pd.DataFrame,
        feature_cols: Sequence[str],
        z_window: int,
        symbol_col: str = "symbol",
        random_state_offset: int = 0,
    ) -> dict:
        """
        Run BS-Stable-HIE for one z_window on the given train_df.

        train_df:
            Train-only dataframe (assert_train_only обеспечивается вызывающим
            кодом). Должен содержать колонки: symbol_col, "timestamp", "close",
            а также все feature_cols.

        feature_cols:
            Список кандидатных признаков (выходы rolling_z_score_clip_df,
            фильтрованных от метаданных).

        z_window:
            Метка z-окна для логирования и сохранения в результаты.

        random_state_offset:
            Дополнительный сдвиг random_state — полезно при параллельных запусках
            на разных z_windows для воспроизводимости.

        Returns:
            dict с ключами:
                ranking_df: per-feature × top_K aggregated frequency и метрики
                per_subsample_df: long-format per-subsample × top_K × feature
                union_full_train: HIE union на полном train (для compatibility audit)
                stable_features_by_top_k: dict[top_k] -> list стабильных признаков
                config_used: snapshot параметров
                mb_stability_bound_by_top_k: dict[top_k] -> MB stability selection
                    finite-sample bound (theoretical motivation only)
                bin_validity_summary: per-config aggregate of HIE bin diagnostics
                    (median min_bin_size_used, min_action_count_in_bin) для proof
                    что HIE не работает на шуме после subsampling
                compute_seconds: float
        """
        feature_cols = list(feature_cols)
        train_df = train_df.reset_index(drop=True)

        # Critical sanity check: future_log_ret должен быть precomputed
        # ДО block subsampling, иначе shift(-horizon) внутри subsample даёт
        # bogus return через временные gaps между блоками.
        if self.future_log_ret_col not in train_df.columns:
            raise ValueError(
                f"train_df is missing precomputed future return column "
                f"{self.future_log_ret_col!r}. Apply "
                f"add_precomputed_future_return(train_df, horizon={self.horizon}, "
                f"future_log_ret_col={self.future_log_ret_col!r}) "
                f"BEFORE calling fit_per_z_window."
            )

        if self.verbose:
            print(
                f"[BS-Stable-HIE z={z_window}] start: n_train={len(train_df)}, "
                f"n_features={len(feature_cols)}, alpha_out={self.alpha_out}, "
                f"B={self.block_config.n_subsamples}, L={self.block_config.block_length}, "
                f"top_k_main={self.block_config.main_top_k}, "
                f"sensitivity={self.block_config.sensitivity_grid_top_k}, "
                f"π={self.block_config.pi_threshold}, "
                f"future_log_ret_col={self.future_log_ret_col!r}"
            )

        t0 = time.time()
        per_subsample_records: list[dict] = []

        n_per_symbol = train_df.groupby(symbol_col).size().to_dict()

        # --- 1. Полный train HIE для full_train union (audit compatibility) ---
        if self.verbose:
            print(f"  [z={z_window}] computing full-train HIE union for audit ...")
        union_full_train = self._fit_union_for_sample(
            train_df, feature_cols,
            random_state_offset=random_state_offset + 100 * z_window,
        )

        # --- 2. Block subsampling loop ---
        for b in range(self.block_config.n_subsamples):
            t_b = time.time()
            idx = block_subsample_indices_per_symbol(
                df=train_df,
                symbol_col=symbol_col,
                block_length=self.block_config.block_length,
                subsample_fraction=self.block_config.subsample_fraction,
                rng=self._rng,
                overlap=self.block_config.overlap_blocks,
            )
            sub = train_df.loc[idx].reset_index(drop=True)

            union_b = self._fit_union_for_sample(
                sub, feature_cols,
                random_state_offset=random_state_offset + 1000 * z_window + 7 * b,
            )

            union_b_idx = union_b.set_index("feature")

            for top_k in self.block_config.sensitivity_grid_top_k:
                topk_features = union_b.head(top_k)["feature"].tolist()
                for f in topk_features:
                    per_subsample_records.append({
                        "subsample_id": b,
                        "z_window": z_window,
                        "alpha_out": self.alpha_out,
                        "top_k": top_k,
                        "feature": f,
                        "union_rank": int(union_b_idx.loc[f, "union_rank"]),
                        "best_rank": float(union_b_idx.loc[f, "best_rank"]),
                        "best_hie_norm": float(union_b_idx.loc[f, "best_hie_norm"]),
                        "best_p_value": float(union_b_idx.loc[f, "best_p_value"]),
                        "regime_coverage_topk": int(union_b_idx.loc[f, "regime_coverage_topk"]),
                        "best_regime_source": str(union_b_idx.loc[f, "best_regime_source"]),
                        "subsample_size": int(len(sub)),
                        # Bin validity diagnostics — proof-of-validity HIE on subsample
                        "entry_n_samples": _safe_int(union_b_idx.loc[f, "entry_n_samples"]),
                        "entry_n_bins_used": _safe_int(union_b_idx.loc[f, "entry_n_bins_used"]),
                        "entry_min_bin_size_used": _safe_int(union_b_idx.loc[f, "entry_min_bin_size_used"]),
                        "entry_min_action_count_in_bin": _safe_int(union_b_idx.loc[f, "entry_min_action_count_in_bin"]),
                        "entry_reason": _safe_str(union_b_idx.loc[f, "entry_reason"]),
                        "exit_n_samples": _safe_int(union_b_idx.loc[f, "exit_n_samples"]),
                        "exit_n_bins_used": _safe_int(union_b_idx.loc[f, "exit_n_bins_used"]),
                        "exit_min_bin_size_used": _safe_int(union_b_idx.loc[f, "exit_min_bin_size_used"]),
                        "exit_min_action_count_in_bin": _safe_int(union_b_idx.loc[f, "exit_min_action_count_in_bin"]),
                        "exit_reason": _safe_str(union_b_idx.loc[f, "exit_reason"]),
                    })

            if self.verbose and (b + 1) % max(1, self.block_config.n_subsamples // 10) == 0:
                elapsed = time.time() - t0
                est_total = elapsed / (b + 1) * self.block_config.n_subsamples
                print(
                    f"  [z={z_window}] subsample {b+1}/{self.block_config.n_subsamples} "
                    f"done in {time.time()-t_b:.1f}s; "
                    f"elapsed {elapsed:.0f}s, est_total {est_total:.0f}s"
                )

        per_subsample_df = pd.DataFrame(per_subsample_records)

        # --- 3. Aggregate frequency_top_K per feature ---
        ranking_rows: list = []
        for top_k in self.block_config.sensitivity_grid_top_k:
            d_k = per_subsample_df[per_subsample_df["top_k"] == top_k]
            for f in feature_cols:
                rows_f = d_k[d_k["feature"] == f]
                n_app = int(len(rows_f))
                freq = n_app / self.block_config.n_subsamples

                # Bin validity diagnostics — медианы поперёк subsamples
                # где feature попал в top-K. Игнорируем sentinel -1 (отсутствие).
                def _valid_median(col: str) -> float:
                    if not n_app:
                        return float("nan")
                    vals = rows_f[col]
                    vals = vals[vals > 0]
                    return float(vals.median()) if len(vals) else float("nan")

                def _valid_min(col: str) -> float:
                    if not n_app:
                        return float("nan")
                    vals = rows_f[col]
                    vals = vals[vals > 0]
                    return float(vals.min()) if len(vals) else float("nan")

                ranking_rows.append({
                    "z_window": z_window,
                    "alpha_out": self.alpha_out,
                    "top_k": top_k,
                    "feature": f,
                    "frequency_top_k": float(freq),
                    "n_appearances": n_app,
                    "n_subsamples_total": int(self.block_config.n_subsamples),
                    "mean_union_rank": float(rows_f["union_rank"].mean()) if n_app else np.nan,
                    "median_union_rank": float(rows_f["union_rank"].median()) if n_app else np.nan,
                    "mean_best_rank": float(rows_f["best_rank"].mean()) if n_app else np.nan,
                    "mean_best_hie_norm": float(rows_f["best_hie_norm"].mean()) if n_app else np.nan,
                    "median_best_hie_norm": float(rows_f["best_hie_norm"].median()) if n_app else np.nan,
                    "median_regime_coverage_topk": (
                        float(rows_f["regime_coverage_topk"].median()) if n_app else np.nan
                    ),
                    # Bin validity aggregates — proof что HIE не работает на шуме.
                    "median_entry_n_samples": _valid_median("entry_n_samples"),
                    "median_entry_n_bins_used": _valid_median("entry_n_bins_used"),
                    "median_entry_min_bin_size_used": _valid_median("entry_min_bin_size_used"),
                    "median_entry_min_action_count_in_bin": _valid_median("entry_min_action_count_in_bin"),
                    "median_exit_n_samples": _valid_median("exit_n_samples"),
                    "median_exit_n_bins_used": _valid_median("exit_n_bins_used"),
                    "median_exit_min_bin_size_used": _valid_median("exit_min_bin_size_used"),
                    "median_exit_min_action_count_in_bin": _valid_median("exit_min_action_count_in_bin"),
                    "min_entry_min_bin_size_used": _valid_min("entry_min_bin_size_used"),
                    "min_exit_min_bin_size_used": _valid_min("exit_min_bin_size_used"),
                    "min_entry_min_action_count_in_bin": _valid_min("entry_min_action_count_in_bin"),
                    "min_exit_min_action_count_in_bin": _valid_min("exit_min_action_count_in_bin"),
                    "is_stable_at_pi": bool(freq >= self.block_config.pi_threshold),
                })
        ranking_df = pd.DataFrame(ranking_rows)

        # --- 4. Stability selection bound (Meinshausen-Bühlmann 2010, Theorem 1) ---
        # E[#{j : V_j ≥ π} ∩ {θ_j = 0}] ≤ q² / ((2π - 1) · p)
        # Используется как theoretical motivation; формальная гарантия требует
        # exchangeability, которая в non-stationary финансовых данных не выполняется
        # напрямую. CPSS (Shah-Samworth 2013) даёт более тугие bounds через
        # complementary pairs, но здесь НЕ реализован — обозначен как related work.
        p_total = len(feature_cols)
        mb_stability_bounds: dict = {}
        denom = 2.0 * self.block_config.pi_threshold - 1.0
        for top_k in self.block_config.sensitivity_grid_top_k:
            if denom > 0 and p_total > 0:
                mb_stability_bounds[int(top_k)] = float((top_k ** 2) / (denom * p_total))
            else:
                mb_stability_bounds[int(top_k)] = float("inf")

        # --- 5. Stable features per top_k, ordered by frequency desc ---
        stable_features_by_top_k: dict[int, list[str]] = {}
        for top_k in self.block_config.sensitivity_grid_top_k:
            d_k = ranking_df[(ranking_df["top_k"] == top_k) & (ranking_df["is_stable_at_pi"])]
            d_k_sorted = d_k.sort_values(
                [
                    "frequency_top_k",
                    "mean_best_hie_norm",
                    "mean_best_rank",
                    "feature",
                ],
                ascending=[False, False, True, True],
            )
            stable_features_by_top_k[int(top_k)] = d_k_sorted["feature"].tolist()

        compute_seconds = time.time() - t0

        # Aggregated bin validity diagnostics at main_top_k для логирования
        d_main = ranking_df[
            (ranking_df["top_k"] == self.block_config.main_top_k)
            & (ranking_df["n_appearances"] > 0)
        ]
        if len(d_main):
            med_entry_minbin = float(d_main["median_entry_min_bin_size_used"].median())
            med_exit_minbin = float(d_main["median_exit_min_bin_size_used"].median())
            med_entry_minact = float(d_main["median_entry_min_action_count_in_bin"].median())
            med_exit_minact = float(d_main["median_exit_min_action_count_in_bin"].median())
        else:
            med_entry_minbin = med_exit_minbin = float("nan")
            med_entry_minact = med_exit_minact = float("nan")

        if self.verbose:
            print(
                f"[BS-Stable-HIE z={z_window}] DONE in {compute_seconds:.0f}s. "
                f"Stable at top_k={self.block_config.main_top_k} & π={self.block_config.pi_threshold}: "
                f"{len(stable_features_by_top_k[self.block_config.main_top_k])} features. "
                f"MB-stability E[FP] bound at top_k={self.block_config.main_top_k}: "
                f"{mb_stability_bounds[self.block_config.main_top_k]:.3f}. "
                f"Bin diagnostics across selected top-K subsamples: "
                f"median min_bin_size_used entry={med_entry_minbin:.0f} exit={med_exit_minbin:.0f}, "
                f"median min_action_count_in_bin entry={med_entry_minact:.0f} exit={med_exit_minact:.0f} "
                f"(HIE thresholds were min_bin_size={self.hie_config.min_bin_size}, "
                f"min_action_count_per_bin={self.hie_config.min_action_count_per_bin})."
            )

        return {
            "ranking_df": ranking_df,
            "per_subsample_df": per_subsample_df,
            "union_full_train": union_full_train,
            "stable_features_by_top_k": stable_features_by_top_k,
            "config_used": {
                "block": vars(self.block_config),
                "hie": vars(self.hie_config),
                "alpha_out": self.alpha_out,
                "horizon": self.horizon,
                "trade_cost": self.trade_cost,
                "future_log_ret_col": self.future_log_ret_col,
                "z_window": int(z_window),
                "n_train": int(len(train_df)),
                "n_features": int(p_total),
                "rows_per_symbol": {str(k): int(v) for k, v in n_per_symbol.items()},
            },
            "mb_stability_bound_by_top_k": mb_stability_bounds,
            "bin_validity_summary": {
                "median_entry_min_bin_size_used_at_main_top_k": med_entry_minbin,
                "median_exit_min_bin_size_used_at_main_top_k": med_exit_minbin,
                "median_entry_min_action_count_in_bin_at_main_top_k": med_entry_minact,
                "median_exit_min_action_count_in_bin_at_main_top_k": med_exit_minact,
                "hie_threshold_min_bin_size": int(self.hie_config.min_bin_size),
                "hie_threshold_min_action_count_per_bin": int(self.hie_config.min_action_count_per_bin),
            },
            "compute_seconds": float(compute_seconds),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_union_for_sample(
        self,
        sample_df: pd.DataFrame,
        feature_cols: list[str],
        random_state_offset: int,
    ) -> pd.DataFrame:
        """Считает entry HIE, exit HIE, и union ranking для одного sample."""
        entry_scores = self._fit_hie_for_regime(
            sample_df, feature_cols,
            prev_position=0,
            random_state_offset=random_state_offset + 1,
        )
        exit_scores = self._fit_hie_for_regime(
            sample_df, feature_cols,
            prev_position=1,
            random_state_offset=random_state_offset + 2,
        )
        union = build_hie_union(
            entry_scores, exit_scores,
            stability_top_k=self.block_config.main_top_k,
        )
        return union

    def _fit_hie_for_regime(
        self,
        sub_df: pd.DataFrame,
        feature_cols: list,
        prev_position: int,
        random_state_offset: int,
    ) -> pd.DataFrame:
        """Запускает HIE selector на одном sample для одного prev_position.

        Использует precomputed future_log_ret из колонки `self.future_log_ret_col`
        в `sub_df`, чтобы избежать bogus future return через block boundaries
        после subsampling.
        """
        cf = make_direct_counterfactual_log_binary_success_from_precomputed(
            sub_df,
            feature_cols=feature_cols,
            trade_cost=self.trade_cost,
            alpha_out=self.alpha_out,
            prev_position=prev_position,
            future_log_ret_col=self.future_log_ret_col,
            action_col_name="raw_action",
            random_state=self.block_config.random_state + random_state_offset,
            horizon=self.horizon,
        )

        cfg = FeatureSelectionConfig(
            n_bins=self.hie_config.n_bins,
            n_bootstrap=self.hie_config.n_bootstrap,
            min_bin_size=self.hie_config.min_bin_size,
            min_action_count_per_bin=self.hie_config.min_action_count_per_bin,
            random_state=self.block_config.random_state + random_state_offset,
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
        scores["regime"] = "entry" if prev_position == 0 else "exit"
        return scores


# =====================================================================
# Global aggregation across z_windows
# =====================================================================


def aggregate_global_stability(
    ranking_dfs_by_z: dict[int, pd.DataFrame],
    main_top_k: int,
    pi_threshold: float,
) -> pd.DataFrame:
    """
    Global stability across z_windows: для каждого признака — в скольких
    z_windows он стабилен (frequency_top_k >= π at main_top_k).

    Полезно как глобальный финальный отбор, аналогичный build_global_stability
    из существующего pipeline.
    """
    rows: list[dict] = []
    for z, df in ranking_dfs_by_z.items():
        d = df[df["top_k"] == main_top_k]
        for _, r in d.iterrows():
            rows.append({
                "z_window": int(z),
                "feature": r["feature"],
                "frequency_top_k": float(r["frequency_top_k"]),
                "is_stable_at_pi": bool(r["is_stable_at_pi"]),
                "mean_best_hie_norm": float(r["mean_best_hie_norm"]),
                "median_union_rank": float(r["median_union_rank"]),
            })
    long_df = pd.DataFrame(rows)

    agg = (
        long_df
        .groupby("feature", as_index=False)
        .agg(
            n_z_windows_seen=("z_window", "nunique"),
            z_windows_seen=("z_window", lambda s: "|".join(map(str, sorted(s.unique())))),
            n_z_windows_stable=("is_stable_at_pi", "sum"),
            mean_frequency=("frequency_top_k", "mean"),
            max_frequency=("frequency_top_k", "max"),
            mean_best_hie_norm_global=("mean_best_hie_norm", "mean"),
            median_union_rank_global=("median_union_rank", "median"),
        )
    )
    agg["is_global_stable_ge1_z"] = agg["n_z_windows_stable"] >= 1
    agg["is_global_stable_ge2_z"] = agg["n_z_windows_stable"] >= 2

    agg = agg.sort_values(
        [
            "n_z_windows_stable",
            "mean_frequency",
            "mean_best_hie_norm_global",
            "median_union_rank_global",
            "feature",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    agg["global_stable_hie_rank"] = np.arange(1, len(agg) + 1)
    agg["main_top_k"] = int(main_top_k)
    agg["pi_threshold"] = float(pi_threshold)
    return agg
