"""
Stage 1 screening: BS-stable HIE hybrid vs correlation baseline на 4 bandit
алгоритмах с default-параметрами.

Цель скрининга: для каждого (algorithm, feature_set, z_window) оценить минимаксную
DRO-метрику качества policy на validation, чтобы выбрать candidate (z, set)
пары для Stage 2 HPO. Скрининг не оптимизирует гиперпараметры, использует
defaults; цель — фильтр feature sets.

Принятые design choices (см. обсуждение):
  - threshold_mode = "none" (нет execution inertia, bandit policy самостоятельно
    отвечает за выбор raw_action; устраняет confounding HPO с execution-layer).
  - bandit_update_action_source = "executed". В режиме "all_bars" это
    post-hoc action masking / shielding (Alshiekh et al. 2018): bandit
    обучается на executed_action после constraint shielding. В режиме
    "decision_only" (текущий default) этот параметр не оказывает влияния
    на forced-singleton барах — bandit там вообще не вызывается, см. ниже.
  - bandit_update_policy = "decision_only" (РЕКОМЕНДУЕТСЯ для нашего минимального
    state setup). Constrained bandit с execution-continuation semantics:
        feasible_set = {1} (MIN_HOLD active)  -> bandit не вызывается;
                                                 pending update НЕ queued;
                                                 executed_action = 1 forced;
        feasible_set = {0} (COOLDOWN active)  -> bandit не вызывается;
                                                 pending update НЕ queued;
                                                 executed_action = 0 forced;
        feasible_set = {0, 1}                  -> bandit делает свободный выбор;
                                                 pending update queued.
    Forced-singleton бары влияют на equity/PnL/trade_log (через execute_transition),
    но posterior bandit'а накапливается только на decision-point барах. Это
    устраняет искажение posterior от forced repeated updates на min_hold/cooldown.
    Теоретическое обоснование: в constrained bandit (Pacchiano et al. 2021)
    regret на барах с |feasible_set|=1 тождественно равен нулю — альтернативного
    действия нет; такие бары являются execution-continuation, не decision point.
  - State context минимальный: state_in_position only. Bandit не различает
    forced и свободное состояние в контексте, но в режиме decision_only это
    не проблема, потому что bandit на forced барах вообще не consultируется.
    Минимальный state сохранён для прямой формулировки HIE vs corr теста без
    добавления MDP-like state-rich features.
  - DRO objective synchronized между screening и HPO (α-смесь, α=DRO_ALPHA):
    seed_dro_score = α*min(mean_symbol(val_h1), mean_symbol(val_h2))
                   + (1-α)*0.5*(mean_symbol(val_h1) + mean_symbol(val_h2))
    dro_objective = median_seed(seed_dro_score)
    При α=1.0 — классический temporal maximin DRO (CVaR(0.5) на 2-point uniform
    mixture). При α=0.5 — mean-CVaR mixture (Rockafellar-Uryasev 2000): worst-half
    эффективный вес 0.75, best-half 0.25 — optimizer учитывает обе половины.
    Pure min(h1, h2), median across symbols, per-symbol maximin сохранены как
    diagnostic для sensitivity analysis.
  - 3 актива (BTC, ETH, SOL), 3 z_windows (24, 48, 72).
  - 4 алгоритма × 2 feature_sets × 3 z_windows = 24 screening configs.
  - 30 seeds для TS алгоритмов, 1 seed для UCB.

Constraint semantics clarification (см. backtest/functions/state_function.py
и backtest_class._apply_constraints):
  - min_hold_bars=4: minimum **elapsed bars since entry** = 4. Forced-hold
    blocks ровно (min_hold_bars-1)=3 decision bars after entry, на 4-м bar
    exit allowed.
  - cooldown_bars=2: minimum **elapsed bars since last trade** = 2. Forced-flat
    blocks (cooldown_bars-1)=1 decision bar after exit, на 2-м bar re-entry
    allowed.
  - В дипломной описывать как "minimum elapsed bars since trade", не как
    "число заблокированных decision bars", чтобы избежать off-by-one
    путаницы.

Запуск:
    cd D:\\PythonProjects\\BTCTrading
    python feature_selection/screening_bs_stable_vs_corr.py

Выходы в feature_selection/bs_stable_hie_outputs_alpha0p5_train_only/
screening_{execution|raw}_action[_decision_only]/  (имя зависит от
BANDIT_UPDATE_ACTION_SOURCE и BANDIT_UPDATE_POLICY, см. OUTPUT_DIR):
    screening_results_per_run.csv     — все runs (algo × set × z × seed × symbol)
    screening_aggregated.csv          — per (algo × set × z), median across seeds
    screening_minimax_dro.csv         — main objective: per (algo × set × z),
                                        median_seed(α*min + (1-α)*mean) — α-смесь
                                        DRO score; pure_min_half_median рядом
                                        как diagnostic (что было бы при α=1.0)
    screening_summary.csv             — top-1 (set × z) per algorithm
    screening_meta.json               — конфигурация запуска

Compute оценка: ~30-60 минут на 3 активах. Per-run backtest ~10-20 секунд,
24 configs × (30+30+1+1) seeds = ~1500 runs total.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [
    PROJECT_ROOT,
    PROJECT_ROOT / "data_processing",
    PROJECT_ROOT / "feature_selection",
    PROJECT_ROOT / "backtest",
    PROJECT_ROOT / "mab",
]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# =====================================================================
# Configuration
# =====================================================================

SEED_BASE = 142
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
OHLCV_RELATIVE_PATH = r"data\klines_data\crypto_240m_bybit_TEST.parquet"

Z_WINDOWS = [24, 48, 72]
ALPHA_OUT = 0.5
HORIZON = 10
TRADE_COST = 0.0025

VAL_BARS = 2000
TEST_BARS = 2000
# EMBARGO_BARS = 0: в online adaptive bandit с pending-update механизмом embargo
# избыточен. Pending в _run_symbol_phase добавляет update в очередь только если
# `due_index < len(df_symbol)`, что гарантирует, что reward для последних H=10
# train-баров не созревает внутри train и эти updates не применяются. Поэтому
# никакие val/test цены не используются для train-фазы. Дополнительный embargo
# делает pipeline только консервативнее (теряет ещё 10 train rows у границы)
# без устранения какого-либо leakage.
# Для feature selection (BS-HIE, corr) labels считаются train-only через
# add_precomputed_future_return на train_df ПОСЛЕ split: последние H rows train
# отбрасываются через dropна, val/test цены не задействованы.
EMBARGO_BARS = 0

# Backtest execution settings (action masking)
MIN_HOLD_BARS = 4
COOLDOWN_BARS = 2
START_CAPITAL = 100.0
POSITION_SIZE = 0.10

# Threshold and update semantics (post-обсуждение решения)
THRESHOLD_MODE = "none"                       # disabled (no execution inertia)
CONFIDENCE_THRESHOLD = 0.0                    # ignored when mode=none
BANDIT_UPDATE_ACTION_SOURCE = "raw"      # action masking semantics raw/executed

# Bandit update policy — decision_only skips bandit consultation and pending-update
# queueing on forced-singleton bars (where |feasible_set|=1 due to MIN_HOLD or
# COOLDOWN). Bandit posterior accumulates ONLY on decision-point bars. Legacy
# behavior ("all_bars") preserves pre-decision_only screening runs.
BANDIT_UPDATE_POLICY = "all_bars"        # "all_bars" | "decision_only"

# DRO objective: α-смесь min и mean между двумя halves validation.
#   seed_dro_score = DRO_ALPHA * min(h1, h2) + (1 - DRO_ALPHA) * 0.5 * (h1 + h2)
# α=1.0 — жёсткий worst-case (классический DRO / CVaR(0.5) на 2-point distr.);
# α=0.5 — robust mean (mean-CVaR mixture, Rockafellar-Uryasev 2000): худшая
#         половина получает эффективный вес 0.75, лучшая 0.25;
# α=0.0 — простое среднее (без робастности к нестационарности).
# Цель α<1: оптимизатор перестаёт игнорировать лучшую половину, когда худшая
# фиксированно доминирует, при этом сохраняет робастный наклон в сторону worst-half.
# Pure min(h1,h2) сохраняется как diagnostic в столбце seed_dro_min_aggregate_half.
DRO_ALPHA = 0.5

# Minimal state (preservation focused hypothesis scope)
STATE_FEATURE_COLUMNS = ["state_in_position"]

# Bandit defaults — синхронизированы с твоим v2 ipynb:
# screening_hybrid_corr_alpha0p5_defaults_top1_balanced_state1_11features.ipynb
DEFAULT_MEMORY_HORIZON_BARS = 325
DEFAULT_DISCOUNT_FACTOR = 1.0 - 1.0 / DEFAULT_MEMORY_HORIZON_BARS    # ≈ 0.99692
DEFAULT_WINDOW_SIZE = DEFAULT_MEMORY_HORIZON_BARS
DEFAULT_LAMBDA_PRIOR = 1.0
DEFAULT_NOISE_STD = 0.03
DEFAULT_UCB_ALPHA = 0.10
DEFAULT_REWARD_CLIP = 0.10
REWARD_CLIP = DEFAULT_REWARD_CLIP

BANDIT_DEFAULT_CONFIGS = {
    "discounted_lints": {
        "bandit_type": "discounted_lints",
        "discount_factor": DEFAULT_DISCOUNT_FACTOR,
        "lambda_prior": DEFAULT_LAMBDA_PRIOR,
        "noise_std": DEFAULT_NOISE_STD,
        "reward_clip": DEFAULT_REWARD_CLIP,
    },
    "discounted_linucb": {
        "bandit_type": "discounted_linucb",
        "discount_factor": DEFAULT_DISCOUNT_FACTOR,
        "lambda_prior": DEFAULT_LAMBDA_PRIOR,
        "ucb_alpha": DEFAULT_UCB_ALPHA,
        "reward_clip": DEFAULT_REWARD_CLIP,
    },
    "sw_lints": {
        "bandit_type": "sw_lints",
        "window_size": DEFAULT_WINDOW_SIZE,
        "lambda_prior": DEFAULT_LAMBDA_PRIOR,
        "noise_std": DEFAULT_NOISE_STD,
        "reward_clip": DEFAULT_REWARD_CLIP,
    },
    "sw_linucb": {
        "bandit_type": "sw_linucb",
        "window_size": DEFAULT_WINDOW_SIZE,
        "lambda_prior": DEFAULT_LAMBDA_PRIOR,
        "ucb_alpha": DEFAULT_UCB_ALPHA,
        "reward_clip": DEFAULT_REWARD_CLIP,
    },
}
TS_ALGORITHMS = ["discounted_lints", "sw_lints"]
UCB_ALGORITHMS = ["discounted_linucb", "sw_linucb"]

# Number of seeds
N_SEEDS_TS = 30
N_SEEDS_UCB = 1

# Output directory naming:
#   - "executed" + "all_bars"       -> screening_execution_action          (legacy)
#   - "raw"      + "all_bars"       -> screening_raw_action                (legacy)
#   - "executed" + "decision_only"  -> screening_execution_action_decision_only
#   - "raw"      + "decision_only"  -> screening_raw_action_decision_only
# Suffix prevents overwriting existing legacy "all_bars" screening results.
_action_label = {"executed": "execution", "raw": "raw"}[BANDIT_UPDATE_ACTION_SOURCE]
_policy_suffix = "" if BANDIT_UPDATE_POLICY == "all_bars" else f"_{BANDIT_UPDATE_POLICY}"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "feature_selection"
    / "bs_stable_hie_outputs_alpha0p5_train_only"
    / f"screening_{_action_label}_action{_policy_suffix}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature sets source (from BS-stable HIE outputs)
FEATURE_SETS_PATH = (
    PROJECT_ROOT
    / "feature_selection"
    / "bs_stable_hie_outputs_alpha0p5_train_only"
    / "feature_sets_hybrid_corr_bs_stable_hie_alpha0p5.json"
)


# =====================================================================
# Project imports
# =====================================================================

from data_processing.functions.klines_dataloader import KlinesDataLoader
from data_processing.functions.stream_indicators import stream_TA_lib
from data_processing.functions.transform_indicators import transform_indicators_df
from data_processing.functions.rolling_z_score_clip import rolling_z_score_clip_df
from backtest.backtest_class import Backtesting

# Note: add_precomputed_future_return и future_log_ret_true НЕ требуются для
# screening, потому что reward вычисляется in-flight в Backtesting._delayed_reward
# на непрерывных train/val (без block subsampling). Precomputed future return
# нужен только для HIE feature selection на block subsamples.

# Indicators config (mirror BS-stable runner)
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
# Helper functions
# =====================================================================


def load_ohlcv() -> pd.DataFrame:
    loader = KlinesDataLoader(symbols=SYMBOLS)
    df = loader.load_data(
        download_path=OHLCV_RELATIVE_PATH,
        analyse_data=True,
        cleaning=True,
    )
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["symbol"].isin(SYMBOLS)].copy()
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def compute_features_for_z_window(
    ohlcv_df: pd.DataFrame, z_window: int
) -> tuple[pd.DataFrame, list]:
    parts = []
    for sym, g in ohlcv_df.groupby("symbol", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True).copy()
        ind = stream_TA_lib(g, meta_cols=META_COLS, **CONFIG_FOR_INDICATORS)
        transformed = transform_indicators_df(ind, meta_cols=META_COLS)
        zdf = rolling_z_score_clip_df(
            transformed, meta_cols=META_COLS,
            window=z_window, clip_value=5.0, shift_by_one=True,
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
        test_start = n - TEST_BARS
        val_start = n - TEST_BARS - VAL_BARS
        train_end = val_start - EMBARGO_BARS
        phase = np.array(["unused_gap"] * n, dtype=object)
        phase[:train_end] = "train"
        if EMBARGO_BARS > 0:
            phase[train_end:val_start] = "embargo"
        phase[val_start:test_start] = "val"
        phase[test_start:] = "test"
        g["phase"] = phase
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def compute_per_symbol_return_pct(balance_list: list, start_capital: float) -> float:
    """Final return % over phase."""
    if not balance_list:
        return float("nan")
    final_value = float(balance_list[-1])
    return 100.0 * (final_value / start_capital - 1.0)


def split_half_returns(
    times_val: list, balance_val: list, start_capital: float
) -> tuple[float, float]:
    """Compute return_pct для val_first_half и val_second_half отдельно.

    Половинная граница: `boundary = n // 2 - 1`. Для VAL_BARS=2000 это даёт
    ровные половины:
        h1: decisions 0..999  (1000 decision bars), return от start_capital
            до balance_val[999].
        h2: decisions 999..1999  (1000 decision bars), return от
            balance_val[999] до balance_val[-1].
    Граничный bar (999) формально включён в обе половины: его close — это
    end_h1 и одновременно start_h2. Это стандартный compound-return split.

    Note: с mid = n//2 (как было раньше) h1 имел 1001 decisions vs h2 999 —
    asymmetric. Текущая boundary=n//2-1 устраняет этот off-by-one.
    """
    if not balance_val or len(balance_val) < 4:
        return float("nan"), float("nan")

    n = len(balance_val)
    boundary = n // 2 - 1

    # First half: from start_capital to balance[boundary]
    start_h1 = float(start_capital)
    end_h1 = float(balance_val[boundary])
    ret_h1 = 100.0 * (end_h1 / start_h1 - 1.0) if start_h1 > 0 else float("nan")

    # Second half: from balance[boundary] to balance[-1]
    start_h2 = float(balance_val[boundary])
    end_h2 = float(balance_val[-1])
    ret_h2 = (
        100.0 * (end_h2 / start_h2 - 1.0)
        if start_h2 > 0
        else float("nan")
    )

    return ret_h1, ret_h2


# =====================================================================
# Single run
# =====================================================================


def run_single_screening(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    bandit_config: dict,
    seed: int,
    z_window: int,
    set_name: str,
    algorithm: str,
) -> dict:
    """Один прогон скрининга: train + val на одной seed."""
    cfg = dict(bandit_config)
    cfg["seed"] = int(seed)
    cfg["n_features"] = len(feature_cols) + len(STATE_FEATURE_COLUMNS)
    cfg["actions"] = [0, 1]

    bt = Backtesting(
        meta_cols=META_COLS,
        feature_columns=feature_cols,
        config_for_bandit=cfg,
        trade_cost=TRADE_COST,
        seed=int(seed),
        update_on_validation=True,
        horizon=HORIZON,
        min_hold_bars=MIN_HOLD_BARS,
        cooldown_bars=COOLDOWN_BARS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        alpha_out=ALPHA_OUT,
        state_feature_columns=STATE_FEATURE_COLUMNS,
        use_symbol_seed_offset=True,
        threshold_mode=THRESHOLD_MODE,
        bandit_update_action_source=BANDIT_UPDATE_ACTION_SOURCE,
        bandit_update_policy=BANDIT_UPDATE_POLICY,
    )

    # Run train (with bandit updates) then val (with bandit updates).
    # API: bt.backtest(dataframe_train, dataframe_val, symbols, start_capital, position_size).
    try:
        bt.backtest(
            dataframe_train=train_df,
            dataframe_val=val_df,
            symbols=SYMBOLS,
            start_capital=START_CAPITAL,
            position_size=POSITION_SIZE,
        )
    except Exception as e:
        return {
            "z_window": int(z_window),
            "set_name": set_name,
            "algorithm": algorithm,
            "seed": int(seed),
            "error": str(e),
            "ok": False,
        }

    # Aggregate metrics per symbol
    per_symbol = []
    for sym in SYMBOLS:
        bal = bt.balance_val[sym]
        if not bal:
            continue
        return_pct_full = compute_per_symbol_return_pct(bal, START_CAPITAL)
        ret_h1, ret_h2 = split_half_returns(
            bt.times_val[sym], bal, START_CAPITAL
        )
        min_half = min(ret_h1, ret_h2)

        # Правильные trade метрики:
        # - exposure_bars: количество decision bars где actions==1 (in long)
        # - n_trade_events: количество BUY/SELL transitions из trade_log_val
        n_decisions = int(len(bt.actions_val[sym]))
        exposure_bars = sum(1 for a in bt.actions_val[sym] if a == 1)
        exposure_ratio = exposure_bars / n_decisions if n_decisions > 0 else 0.0
        trade_events = bt.trade_log_val.get(sym, [])
        n_trade_events = int(len(trade_events))
        n_buys = sum(1 for x in trade_events if x.get("event") == "BUY")
        n_sells = sum(1 for x in trade_events if x.get("event") == "SELL")

        # Constraint diagnostics из decision_log_val.
        # В режиме "all_bars": constraint_applied = post-hoc override raw->executed
        #                       (raw_executed_mismatch ratio относится к force overrides).
        # В режиме "decision_only": forced-singleton bars не вызывают bandit,
        #                       поэтому raw_action == executed_action by construction.
        #                       Главные новые метрики:
        #                         forced_singleton_ratio  = доля баров с |feasible|=1
        #                         decision_point_ratio    = доля баров с |feasible|=2
        #                         bandit_consulted_ratio  = доля баров где bandit
        #                                                   реально делал выбор
        #                         pending_update_queued_ratio = доля баров с pending
        #                                                       update queued
        decisions = bt.decision_log_val.get(sym, [])
        if decisions:
            n_constraint_applied = sum(
                1 for d in decisions if d.get("constraint_applied", False)
            )
            n_min_hold_applied = sum(
                1 for d in decisions if d.get("constraint_type") == "min_hold"
            )
            n_cooldown_applied = sum(
                1 for d in decisions if d.get("constraint_type") == "cooldown"
            )
            # Legacy mismatch — в decision_only всегда 0 (raw_action forced equals executed)
            n_raw_executed_mismatch = sum(
                1 for d in decisions
                if d.get("raw_action") != d.get("executed_action")
            )
            n_blocked_exit = sum(
                1 for d in decisions
                if d.get("raw_action") == 0 and d.get("executed_action") == 1
                and d.get("constraint_type") == "min_hold"
            )
            n_blocked_entry = sum(
                1 for d in decisions
                if d.get("raw_action") == 1 and d.get("executed_action") == 0
                and d.get("constraint_type") == "cooldown"
            )
            # decision_only-aware diagnostics
            n_forced_singleton = sum(
                1 for d in decisions if d.get("feasible_set_size", 2) == 1
            )
            n_decision_points = sum(
                1 for d in decisions if d.get("is_decision_point", True)
            )
            n_bandit_consulted = sum(
                1 for d in decisions if d.get("bandit_consulted", True)
            )
            n_pending_update_queued = sum(
                1 for d in decisions if d.get("pending_update_queued", True)
            )
            denom = len(decisions)
            constraint_applied_ratio = n_constraint_applied / denom
            min_hold_ratio = n_min_hold_applied / denom
            cooldown_ratio = n_cooldown_applied / denom
            raw_executed_mismatch_ratio = n_raw_executed_mismatch / denom
            blocked_exit_ratio = n_blocked_exit / denom
            blocked_entry_ratio = n_blocked_entry / denom
            forced_singleton_ratio = n_forced_singleton / denom
            decision_point_ratio = n_decision_points / denom
            bandit_consulted_ratio = n_bandit_consulted / denom
            pending_update_queued_ratio = n_pending_update_queued / denom
        else:
            constraint_applied_ratio = min_hold_ratio = cooldown_ratio = 0.0
            raw_executed_mismatch_ratio = blocked_exit_ratio = blocked_entry_ratio = 0.0
            forced_singleton_ratio = decision_point_ratio = 0.0
            bandit_consulted_ratio = pending_update_queued_ratio = 0.0

        per_symbol.append({
            "symbol": sym,
            "val_return_pct_full": return_pct_full,
            "val_return_pct_h1": ret_h1,
            "val_return_pct_h2": ret_h2,
            "val_min_half_return_pct": min_half,
            # Trade metrics (правильное naming):
            "n_decisions_val": n_decisions,
            "exposure_bars_val": int(exposure_bars),
            "exposure_ratio_val": float(exposure_ratio),
            "n_trade_events_val": n_trade_events,
            "n_buys_val": int(n_buys),
            "n_sells_val": int(n_sells),
            # Legacy constraint diagnostics (informative in "all_bars" mode):
            "constraint_applied_ratio": float(constraint_applied_ratio),
            "min_hold_ratio": float(min_hold_ratio),
            "cooldown_ratio": float(cooldown_ratio),
            "raw_executed_mismatch_ratio": float(raw_executed_mismatch_ratio),
            "blocked_exit_ratio": float(blocked_exit_ratio),
            "blocked_entry_ratio": float(blocked_entry_ratio),
            # decision_only-aware diagnostics (NEW):
            "forced_singleton_ratio": float(forced_singleton_ratio),
            "decision_point_ratio": float(decision_point_ratio),
            "bandit_consulted_ratio": float(bandit_consulted_ratio),
            "pending_update_queued_ratio": float(pending_update_queued_ratio),
        })

    if not per_symbol:
        return {
            "z_window": int(z_window),
            "set_name": set_name,
            "algorithm": algorithm,
            "seed": int(seed),
            "error": "no per-symbol results",
            "ok": False,
        }

    psdf = pd.DataFrame(per_symbol)
    # Per-symbol aggregation для seed-level metric.
    # Используем MEAN across symbols (equal-weight aggregate, точно соответствует
    # equal-weight aggregate evaluation strategy across symbols), затем α-смесь
    # min и mean между двумя halves (см. DRO_ALPHA в config):
    #   seed_dro_score = α * min(h1, h2) + (1 - α) * 0.5 * (h1 + h2)
    # При α=0.5 это mean-CVaR mixture (Rockafellar-Uryasev 2000): worst-half
    # эффективный вес 0.75, best-half 0.25. При α=1.0 классический temporal
    # maximin DRO. Pure min(h1, h2) сохранён как diagnostic.
    mean_symbol_full = float(psdf["val_return_pct_full"].mean())
    mean_symbol_h1 = float(psdf["val_return_pct_h1"].mean())
    mean_symbol_h2 = float(psdf["val_return_pct_h2"].mean())
    seed_dro_min_aggregate_half = float(min(mean_symbol_h1, mean_symbol_h2))  # diagnostic
    seed_dro_score = float(
        DRO_ALPHA * seed_dro_min_aggregate_half
        + (1.0 - DRO_ALPHA) * 0.5 * (mean_symbol_h1 + mean_symbol_h2)
    )

    # Дополнительные diagnostic metrics:
    # 1) Median across symbols — robust к outlier symbol (информативно при
    #    сильной asymmetry per-symbol).
    median_symbol_full = float(psdf["val_return_pct_full"].median())
    median_symbol_h1 = float(psdf["val_return_pct_h1"].median())
    median_symbol_h2 = float(psdf["val_return_pct_h2"].median())
    # 2) Per-symbol maximin: median across symbols of per-symbol min_half.
    #    Более жёсткий variant DRO (more conservative).
    median_symbol_per_symbol_min_half = float(psdf["val_min_half_return_pct"].median())

    # Aggregation для trade/constraint diagnostics
    median_exposure_bars = float(psdf["exposure_bars_val"].median())
    median_exposure_ratio = float(psdf["exposure_ratio_val"].median())
    median_n_trade_events = float(psdf["n_trade_events_val"].median())
    mean_constraint_applied_ratio = float(psdf["constraint_applied_ratio"].mean())
    mean_min_hold_ratio = float(psdf["min_hold_ratio"].mean())
    mean_cooldown_ratio = float(psdf["cooldown_ratio"].mean())
    mean_raw_executed_mismatch_ratio = float(psdf["raw_executed_mismatch_ratio"].mean())
    mean_blocked_exit_ratio = float(psdf["blocked_exit_ratio"].mean())
    mean_blocked_entry_ratio = float(psdf["blocked_entry_ratio"].mean())
    # decision_only-aware aggregations (NEW):
    mean_forced_singleton_ratio = float(psdf["forced_singleton_ratio"].mean())
    mean_decision_point_ratio = float(psdf["decision_point_ratio"].mean())
    mean_bandit_consulted_ratio = float(psdf["bandit_consulted_ratio"].mean())
    mean_pending_update_queued_ratio = float(psdf["pending_update_queued_ratio"].mean())

    return {
        "z_window": int(z_window),
        "set_name": set_name,
        "algorithm": algorithm,
        "seed": int(seed),
        # MAIN DRO objective per seed: α-смесь min и mean (α=DRO_ALPHA).
        #   seed_dro_score = α * min(h1, h2) + (1 - α) * 0.5 * (h1 + h2)
        # Aligned с финальной equal-weight aggregate evaluation across symbols.
        "seed_dro_score": seed_dro_score,
        "dro_alpha": float(DRO_ALPHA),
        # DIAGNOSTIC: pure min(h1, h2) — что было бы при α=1.0 (hard worst-case).
        "seed_dro_min_aggregate_half": seed_dro_min_aggregate_half,
        # Equal-weight aggregate components (mean across symbols)
        "mean_symbol_val_full": mean_symbol_full,
        "mean_symbol_val_h1": mean_symbol_h1,
        "mean_symbol_val_h2": mean_symbol_h2,
        # Median diagnostic (robust к outlier symbols)
        "median_symbol_val_full": median_symbol_full,
        "median_symbol_val_h1": median_symbol_h1,
        "median_symbol_val_h2": median_symbol_h2,
        # Per-symbol maximin diagnostic (более жёсткий variant)
        "median_symbol_per_symbol_min_half": median_symbol_per_symbol_min_half,
        # Trade metrics
        "median_exposure_bars_val": median_exposure_bars,
        "median_exposure_ratio_val": median_exposure_ratio,
        "median_n_trade_events_val": median_n_trade_events,
        # Constraint diagnostics
        "mean_constraint_applied_ratio": mean_constraint_applied_ratio,
        "mean_min_hold_ratio": mean_min_hold_ratio,
        "mean_cooldown_ratio": mean_cooldown_ratio,
        "mean_raw_executed_mismatch_ratio": mean_raw_executed_mismatch_ratio,
        "mean_blocked_exit_ratio": mean_blocked_exit_ratio,
        "mean_blocked_entry_ratio": mean_blocked_entry_ratio,
        # decision_only-aware diagnostics (NEW):
        "mean_forced_singleton_ratio": mean_forced_singleton_ratio,
        "mean_decision_point_ratio": mean_decision_point_ratio,
        "mean_bandit_consulted_ratio": mean_bandit_consulted_ratio,
        "mean_pending_update_queued_ratio": mean_pending_update_queued_ratio,
        "per_symbol": psdf.to_dict("records"),
        "ok": True,
    }


# =====================================================================
# Main pipeline
# =====================================================================


def main():
    t_start = datetime.now()
    print("=" * 80)
    print("Screening BS-stable HIE hybrid vs corr baseline")
    print(f"Symbols: {SYMBOLS}")
    print(f"Z_windows: {Z_WINDOWS}")
    print(f"Algorithms: {list(BANDIT_DEFAULT_CONFIGS.keys())}")
    print(f"Seeds: TS={N_SEEDS_TS}, UCB={N_SEEDS_UCB}")
    print(f"Threshold mode: {THRESHOLD_MODE}")
    print(f"Bandit update action source: {BANDIT_UPDATE_ACTION_SOURCE}")
    print(f"Bandit update policy:        {BANDIT_UPDATE_POLICY}")
    print(f"State features: {STATE_FEATURE_COLUMNS}")
    print(f"DRO_ALPHA = {DRO_ALPHA} "
          f"(α=1 → hard min, α=0.5 → mean-CVaR mixture, α=0 → mean)")
    print(f"DRO objective: median_seed( {DRO_ALPHA}*min(h1,h2) + {1.0-DRO_ALPHA}*0.5*(h1+h2) )")
    print("=" * 80)

    # Load feature sets
    with open(FEATURE_SETS_PATH) as f:
        feature_sets = json.load(f)

    available_sets = sorted(feature_sets.keys())
    print(f"\nLoaded {len(available_sets)} feature sets from {FEATURE_SETS_PATH.name}:")
    for sn in available_sets:
        print(f"  {sn}: {len(feature_sets[sn])} features")

    # Load OHLCV
    ohlcv = load_ohlcv()
    print(f"\nOHLCV loaded: {ohlcv.shape}\n")

    # Loop over z_windows
    all_seed_results = []

    for z_window in Z_WINDOWS:
        print("=" * 80)
        print(f"z_window = {z_window}")
        print("=" * 80)

        zdf, feature_cols_all = compute_features_for_z_window(ohlcv, z_window)
        zdf = chronological_split_per_symbol(zdf)
        train_df = zdf[zdf["phase"] == "train"].copy().reset_index(drop=True)
        val_df = zdf[zdf["phase"] == "val"].copy().reset_index(drop=True)

        # Note: для screening бэктесту не нужен future_log_ret_true precompute,
        # потому что reward вычисляется in-flight через _delayed_reward, не через
        # counterfactual log. Backtesting класс сам делает shift на непрерывных
        # train/val (без block subsampling).

        print(f"  train: {len(train_df)} rows, val: {len(val_df)} rows")

        # Feature sets для этого z
        set_names = [
            f"z{z_window}_a0p5_corr_pruned_top10",
            f"z{z_window}_a0p5_hybrid_corr5_bsstablehie5_top10",
        ]
        for sn in set_names:
            if sn not in feature_sets:
                print(f"  WARN: set {sn} not in feature_sets, skipping")
                continue

            set_features = feature_sets[sn]
            missing = [f for f in set_features if f not in train_df.columns]
            if missing:
                print(f"  WARN: missing features in train_df for {sn}: {missing}")
                continue

            print(f"\n  Set: {sn} ({len(set_features)} features)")

            for algo_name, default_cfg in BANDIT_DEFAULT_CONFIGS.items():
                n_seeds = N_SEEDS_TS if algo_name in TS_ALGORITHMS else N_SEEDS_UCB
                t_algo_start = time.time()
                print(f"    Algorithm: {algo_name}  (seeds: {n_seeds})")

                for seed_offset in range(n_seeds):
                    seed = SEED_BASE + seed_offset * 100 + z_window
                    res = run_single_screening(
                        train_df=train_df,
                        val_df=val_df,
                        feature_cols=set_features,
                        bandit_config=default_cfg,
                        seed=seed,
                        z_window=z_window,
                        set_name=sn,
                        algorithm=algo_name,
                    )
                    if res.get("ok"):
                        all_seed_results.append(res)
                    else:
                        print(f"      seed {seed} ERROR: {res.get('error', 'unknown')}")

                elapsed = time.time() - t_algo_start
                print(f"      done in {elapsed:.1f}s ({elapsed/max(1, n_seeds):.1f}s/seed)")

    # ================================================================
    # Aggregation
    # ================================================================

    if not all_seed_results:
        print("\nNo successful runs. Exiting.")
        return

    print("\n" + "=" * 80)
    print("Aggregation")
    print("=" * 80)

    # Per-run table (per seed × symbol)
    per_run_rows = []
    for r in all_seed_results:
        for sym_row in r["per_symbol"]:
            per_run_rows.append({
                "z_window": r["z_window"],
                "set_name": r["set_name"],
                "algorithm": r["algorithm"],
                "seed": r["seed"],
                "symbol": sym_row["symbol"],
                "val_return_pct_full": sym_row["val_return_pct_full"],
                "val_return_pct_h1": sym_row["val_return_pct_h1"],
                "val_return_pct_h2": sym_row["val_return_pct_h2"],
                "val_min_half_return_pct": sym_row["val_min_half_return_pct"],
                "n_decisions_val": sym_row["n_decisions_val"],
                "exposure_bars_val": sym_row["exposure_bars_val"],
                "exposure_ratio_val": sym_row["exposure_ratio_val"],
                "n_trade_events_val": sym_row["n_trade_events_val"],
                "n_buys_val": sym_row["n_buys_val"],
                "n_sells_val": sym_row["n_sells_val"],
                "constraint_applied_ratio": sym_row["constraint_applied_ratio"],
                "min_hold_ratio": sym_row["min_hold_ratio"],
                "cooldown_ratio": sym_row["cooldown_ratio"],
                # Legacy all_bars diagnostics. В decision_only режиме эти величины
                # by construction нулевые (raw=executed на decision points, никогда
                # не блокируется на forced bars — туда bandit не consultируется).
                "raw_executed_mismatch_ratio": sym_row["raw_executed_mismatch_ratio"],
                "blocked_exit_ratio": sym_row["blocked_exit_ratio"],
                "blocked_entry_ratio": sym_row["blocked_entry_ratio"],
                # decision_only-aware per-symbol diagnostics (более информативны):
                #   forced_singleton_ratio   — доля баров с |feasible_set|=1
                #   decision_point_ratio     — доля баров с |feasible_set|>=2
                #   bandit_consulted_ratio   — доля баров где bandit принимал решение
                #   pending_update_queued_ratio — доля баров с queued posterior update
                "forced_singleton_ratio":      sym_row.get("forced_singleton_ratio", float("nan")),
                "decision_point_ratio":        sym_row.get("decision_point_ratio", float("nan")),
                "bandit_consulted_ratio":      sym_row.get("bandit_consulted_ratio", float("nan")),
                "pending_update_queued_ratio": sym_row.get("pending_update_queued_ratio", float("nan")),
            })
    per_run_df = pd.DataFrame(per_run_rows)
    per_run_df.to_csv(OUTPUT_DIR / "screening_results_per_run.csv", index=False)
    print(f"  Saved per-run table: {len(per_run_df)} rows")

    # Per (algo × set × z × seed): aggregate (mean) across symbols + seed_dro
    seed_level_rows = []
    for r in all_seed_results:
        seed_level_rows.append({
            "z_window": r["z_window"],
            "set_name": r["set_name"],
            "algorithm": r["algorithm"],
            "seed": r["seed"],
            # MAIN DRO objective per seed (α-смесь)
            "seed_dro_score": r["seed_dro_score"],
            "dro_alpha": r["dro_alpha"],
            # Diagnostic: pure min(h1, h2)
            "seed_dro_min_aggregate_half": r["seed_dro_min_aggregate_half"],
            "mean_symbol_val_full": r["mean_symbol_val_full"],
            "mean_symbol_val_h1": r["mean_symbol_val_h1"],
            "mean_symbol_val_h2": r["mean_symbol_val_h2"],
            "median_symbol_val_full": r["median_symbol_val_full"],
            "median_symbol_val_h1": r["median_symbol_val_h1"],
            "median_symbol_val_h2": r["median_symbol_val_h2"],
            "median_symbol_per_symbol_min_half": r["median_symbol_per_symbol_min_half"],
            "median_exposure_bars_val": r["median_exposure_bars_val"],
            "median_exposure_ratio_val": r["median_exposure_ratio_val"],
            "median_n_trade_events_val": r["median_n_trade_events_val"],
            "mean_constraint_applied_ratio": r["mean_constraint_applied_ratio"],
            "mean_min_hold_ratio": r["mean_min_hold_ratio"],
            "mean_cooldown_ratio": r["mean_cooldown_ratio"],
            "mean_raw_executed_mismatch_ratio": r["mean_raw_executed_mismatch_ratio"],
            "mean_blocked_exit_ratio": r["mean_blocked_exit_ratio"],
            "mean_blocked_entry_ratio": r["mean_blocked_entry_ratio"],
            # decision_only-aware (NEW)
            "mean_forced_singleton_ratio": r.get("mean_forced_singleton_ratio", 0.0),
            "mean_decision_point_ratio": r.get("mean_decision_point_ratio", 1.0),
            "mean_bandit_consulted_ratio": r.get("mean_bandit_consulted_ratio", 1.0),
            "mean_pending_update_queued_ratio": r.get("mean_pending_update_queued_ratio", 1.0),
        })
    seed_level = pd.DataFrame(seed_level_rows)
    seed_level.to_csv(OUTPUT_DIR / "screening_aggregated.csv", index=False)
    print(f"  Saved seed-level aggregation: {len(seed_level)} rows")

    # Main DRO objective per (algo × set × z):
    #   median_seed(seed_dro_score)
    # где seed_dro_score = α * min(h1, h2) + (1 - α) * 0.5 * (h1 + h2),
    # h_k = mean_symbol(val_h_k). α=DRO_ALPHA (см. config).
    # Это main objective и для screening, и для HPO.
    # Pure min(h1, h2) сохранён как diagnostic: pure_min_half_median (что было бы
    # при α=1.0 — классический worst-case DRO).
    minimax_dro = (
        seed_level.groupby(["z_window", "set_name", "algorithm"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            dro_minimax_median=("seed_dro_score", "median"),
            dro_minimax_mean=("seed_dro_score", "mean"),
            dro_minimax_std=("seed_dro_score", "std"),
            dro_minimax_q25=("seed_dro_score", lambda s: float(np.percentile(s, 25))),
            dro_minimax_q75=("seed_dro_score", lambda s: float(np.percentile(s, 75))),
            # Diagnostic: pure min(h1,h2) — что было бы при α=1.0 (hard worst-case)
            pure_min_half_median=("seed_dro_min_aggregate_half", "median"),
            pure_min_half_mean=("seed_dro_min_aggregate_half", "mean"),
            # Equal-weight aggregate components (main scale)
            full_mean_aggregate=("mean_symbol_val_full", "median"),
            h1_mean_aggregate=("mean_symbol_val_h1", "median"),
            h2_mean_aggregate=("mean_symbol_val_h2", "median"),
            # Median diagnostic (robust к outlier symbols)
            full_median=("median_symbol_val_full", "median"),
            h1_median=("median_symbol_val_h1", "median"),
            h2_median=("median_symbol_val_h2", "median"),
            # Per-symbol maximin diagnostic (более жёсткий)
            per_symbol_minhalf_median=("median_symbol_per_symbol_min_half", "median"),
            median_exposure_bars=("median_exposure_bars_val", "median"),
            median_exposure_ratio=("median_exposure_ratio_val", "median"),
            median_n_trade_events=("median_n_trade_events_val", "median"),
            mean_constraint_applied_ratio=("mean_constraint_applied_ratio", "mean"),
            mean_raw_executed_mismatch_ratio=("mean_raw_executed_mismatch_ratio", "mean"),
            mean_blocked_exit_ratio=("mean_blocked_exit_ratio", "mean"),
            mean_blocked_entry_ratio=("mean_blocked_entry_ratio", "mean"),
            # decision_only-aware (NEW)
            mean_forced_singleton_ratio=("mean_forced_singleton_ratio", "mean"),
            mean_decision_point_ratio=("mean_decision_point_ratio", "mean"),
            mean_bandit_consulted_ratio=("mean_bandit_consulted_ratio", "mean"),
            mean_pending_update_queued_ratio=("mean_pending_update_queued_ratio", "mean"),
        )
        .sort_values(
            ["algorithm", "dro_minimax_median"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )
    minimax_dro.to_csv(OUTPUT_DIR / "screening_minimax_dro.csv", index=False)
    print(f"  Saved minimax DRO table: {len(minimax_dro)} rows")
    print()
    print("Main DRO objective per (algorithm, set_name, z_window):")
    print(f"  seed_dro_score = {DRO_ALPHA}*min(h1,h2) + {1.0-DRO_ALPHA}*0.5*(h1+h2)   "
          f"[α-mixture, α=DRO_ALPHA={DRO_ALPHA}]")
    print("  dro_minimax_median = median_seed(seed_dro_score)")
    print("  pure_min_half_median = median_seed(min(h1,h2))   [diagnostic, α=1.0 baseline]")
    print("  (sorted by dro_minimax_median desc within each algorithm)")
    print(minimax_dro[
        ["algorithm", "set_name", "z_window", "n_seeds",
         "dro_minimax_median", "h1_mean_aggregate", "h2_mean_aggregate",
         "full_mean_aggregate",
         "median_n_trade_events", "median_exposure_ratio",
         "mean_constraint_applied_ratio", "mean_blocked_exit_ratio",
         "mean_blocked_entry_ratio"]
    ].to_string(index=False))

    # Top-1 (set × z) per algorithm by minimax DRO
    top_per_algo = (
        minimax_dro.sort_values(["algorithm", "dro_minimax_median"], ascending=[True, False])
        .groupby("algorithm", as_index=False)
        .first()
    )
    top_per_algo.to_csv(OUTPUT_DIR / "screening_summary.csv", index=False)
    print("\n" + "=" * 80)
    print("Top-1 (set × z) per algorithm by main DRO objective:")
    print("=" * 80)
    print(top_per_algo[
        ["algorithm", "set_name", "z_window", "dro_minimax_median",
         "h1_mean_aggregate", "h2_mean_aggregate", "full_mean_aggregate",
         "median_n_trade_events"]
    ].to_string(index=False))

    # Hybrid vs corr comparison per algorithm (DRO objective)
    pivot = minimax_dro.pivot_table(
        index=["algorithm", "z_window"],
        columns="set_name",
        values="dro_minimax_median",
    )
    # Find columns with corr and hybrid patterns
    print("\n" + "=" * 80)
    print("Hybrid vs Corr на minimax DRO objective:")
    print("=" * 80)
    hybrid_vs_corr_rows = []
    for (algo, z), row in pivot.iterrows():
        # Bug fix: ищем колонки которые относятся именно к данному z_window,
        # потому что pivot index содержит все 6 set_names (по 2 на каждый z),
        # и `next(...if "corr_pruned" in c)` без фильтра по z ловил бы первый
        # corr_pruned (всегда z24), давая NaN для (algo, z>=48) rows.
        z_prefix = f"z{z}_"
        corr_col = next(
            (c for c in row.index if "corr_pruned" in c and c.startswith(z_prefix)),
            None,
        )
        hybrid_col = next(
            (c for c in row.index if "hybrid" in c and c.startswith(z_prefix)),
            None,
        )
        if corr_col and hybrid_col:
            corr_val = float(row[corr_col]) if not pd.isna(row[corr_col]) else None
            hybrid_val = float(row[hybrid_col]) if not pd.isna(row[hybrid_col]) else None
            diff = (
                hybrid_val - corr_val if (corr_val is not None and hybrid_val is not None)
                else None
            )
            hybrid_vs_corr_rows.append({
                "algorithm": algo,
                "z_window": z,
                "corr_dro": corr_val,
                "hybrid_dro": hybrid_val,
                "hybrid_minus_corr": diff,
            })
    cmp_df = pd.DataFrame(hybrid_vs_corr_rows)
    cmp_df.to_csv(OUTPUT_DIR / "screening_hybrid_vs_corr.csv", index=False)
    print(cmp_df.to_string(index=False))

    # Save run meta
    t_end = datetime.now()
    meta = {
        "started_at": t_start.isoformat(),
        "finished_at": t_end.isoformat(),
        "duration_seconds": (t_end - t_start).total_seconds(),
        "symbols": SYMBOLS,
        "z_windows": Z_WINDOWS,
        "algorithms": list(BANDIT_DEFAULT_CONFIGS.keys()),
        "n_seeds_ts": N_SEEDS_TS,
        "n_seeds_ucb": N_SEEDS_UCB,
        "threshold_mode": THRESHOLD_MODE,
        "bandit_update_action_source": BANDIT_UPDATE_ACTION_SOURCE,
        "bandit_update_policy": BANDIT_UPDATE_POLICY,
        "bandit_update_semantics": (
            f"Update policy: {BANDIT_UPDATE_POLICY}. "
            + ("Constrained-bandit semantics: posterior updates are queued ONLY on "
               "bars with |feasible_set|>=2 (decision points). Forced-singleton bars "
               "(MIN_HOLD or COOLDOWN active) affect equity but do not consult the "
               "bandit and do not queue pending updates."
               if BANDIT_UPDATE_POLICY == "decision_only"
               else "Legacy 'all_bars' mode: post-hoc action masking / shielding "
                    "(Alshiekh et al. 2018 style). Bandit is consulted on every bar; "
                    "execution constraints override raw_action post-selection; bandit "
                    "updates use executed_action after constraint shielding.")
        ),
        "state_feature_columns": STATE_FEATURE_COLUMNS,
        "state_limitation_note": (
            "bandit context = state_in_position only; bandit не видит "
            "bars_in_position / time_since_last_trade. В режиме decision_only "
            "это не приводит к искажению posterior, так как на forced-singleton "
            "барах (|feasible_set|=1) bandit вообще не consultируется и pending "
            "update не queued — posterior накапливается только на decision-point "
            "барах. Forced бары влияют на equity/PnL через execute_transition. "
            "Это применяется одинаково к hybrid и corr feature sets."
        ),
        "dro_alpha": float(DRO_ALPHA),
        "dro_objective_formula": (
            f"seed_dro_score = {DRO_ALPHA}*min(mean_symbol(val_h1), mean_symbol(val_h2)) "
            f"+ {1.0 - DRO_ALPHA}*0.5*(mean_symbol(val_h1) + mean_symbol(val_h2)); "
            f"dro_objective = median_seed(seed_dro_score)"
        ),
        "dro_objective_interpretation": (
            f"α-смесь жёсткого worst-case min(h1,h2) и среднего 0.5*(h1+h2) с α="
            f"{DRO_ALPHA}. При α=1.0 классический temporal maximin DRO (равно "
            f"CVaR(0.5) на 2-point uniform mixture). При α=0.5 это mean-CVaR "
            f"mixture (Rockafellar-Uryasev 2000): худшая половина эффективный "
            f"вес 0.75, лучшая 0.25 — optimizer учитывает обе половины при "
            f"сохранении робастного наклона. При α=0.0 — простое среднее (без "
            f"робастности). Aligned с финальной equal-weight aggregate "
            f"evaluation strategy across symbols; synchronized между screening "
            f"и HPO через DRO_ALPHA константу."
        ),
        "dro_diagnostics": {
            "pure_min_half_median": (
                "diagnostic: median_seed(min(h1, h2)) — что было бы при α=1.0, "
                "т.е. классический hard worst-case DRO baseline для сравнения"
            ),
            "median_symbol_aggregate_minhalf": (
                "median across symbols variant: робастный к outlier symbols"
            ),
            "per_symbol_minhalf_median": (
                "per-symbol maximin variant: более жёсткий, требует чтобы каждый symbol "
                "не проваливался в одной из половин (сохранён как diagnostic, не main)"
            ),
        },
        "constraint_semantics_note": (
            "min_hold_bars=4 means min elapsed bars since entry = 4 "
            "(forced-hold blocks 3 decision bars after entry); "
            "cooldown_bars=2 means min elapsed bars since last trade = 2 "
            "(forced-flat blocks 1 decision bar after exit)"
        ),
        "horizon": HORIZON,
        "trade_cost": TRADE_COST,
        "val_bars": VAL_BARS,
        "embargo_bars": EMBARGO_BARS,
        "min_hold_bars": MIN_HOLD_BARS,
        "cooldown_bars": COOLDOWN_BARS,
        "position_size": POSITION_SIZE,
        "start_capital": START_CAPITAL,
        "reward_clip": REWARD_CLIP,
        "alpha_out": ALPHA_OUT,
        "feature_sets_source": str(FEATURE_SETS_PATH),
        "bandit_default_configs": BANDIT_DEFAULT_CONFIGS,
    }
    with open(OUTPUT_DIR / "screening_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"DONE. Total duration: {(t_end - t_start).total_seconds():.0f}s")
    print(f"Outputs: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
