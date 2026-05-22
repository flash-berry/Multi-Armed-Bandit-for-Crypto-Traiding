"""
Stage 2 HPO - DRO temporal-maximin objective for non-stationary CMAB.

Pipeline
--------
For each (algorithm, method_group) selected from screening top-1 (z):
  1) Run Optuna study with TPESampler + constraints_func.
     Primary objective (α-смесь min и mean, α=DRO_ALPHA):
         seed_dro_score    = α * min(mean_symbol(val_h1), mean_symbol(val_h2))
                           + (1 - α) * 0.5 * (mean_symbol(val_h1) + mean_symbol(val_h2))
         dro_minimax_median = median_seed(seed_dro_score)
     α=1.0 → классический temporal maximin DRO; α=0.5 → mean-CVaR mixture
     (Rockafellar-Uryasev 2000): worst-half эффективный вес 0.75, best-half 0.25;
     α=0.0 → простое среднее. Pure min(h1,h2) сохраняется как diagnostic
     (dro_pure_min_median). DRO_ALPHA должен совпадать с одноимённой константой
     в screening_bs_stable_vs_corr.py.
     Constraints — MINIMAL SET (Optuna convention: value <= 0 feasible):
         (1) median_min_symbol_trades  >= MIN_CLOSED_TRADES_PER_SYMBOL (50)
         (2) finite objective
     Drawdown, profit factor, aggregate trade count are NOT constraints; they
     are computed as DIAGNOSTICS in trial_results_all.csv. Rationale:
       (a) Transaction costs enter realised log-returns, so excessive turnover
           is expected to hurt DRO; this is an empirical tendency, not a hard
           guarantee — pathological cases are detectable through the diagnostic
           columns.
       (b) Drawdown is NOT directly constrained and is reported as a path-risk
           diagnostic. DRO is endpoint-aware (worst-half mean return), not
           path-aware: large intra-half drawdowns that recover by the half's
           endpoint are not penalized through DRO alone.
       (c) Unprofitable trading depresses half-returns and is captured by DRO
           through negative worst-half values; here too, the relationship is
           empirical not enforced.
     The single retained constraint (min-symbol-trades) protects against
     single-symbol degeneracy in the 3-symbol experiment.
     A "screening default trial" (mem=325, lambda=1.0, noise=0.03 / ucb_alpha=0.10)
     is enqueued as trial #0 for every study.

  2) HPO selector (primary):
         best_main = argmax dro_minimax_median over feasible completed trials.
     (Constraint-aware. If no feasible trial exists, best is fallback-infeasible.)

  3) TS confirmation (only for Thompson Sampling bandits):
         - take top-N feasible trials by main DRO;
         - re-run each on 25 fresh confirmation seeds;
         - recompute DRO + same constraints on confirmation seeds;
         - confirmation selector = argmax confirmation_dro_minimax_median over
           confirmation-feasible trials. Fallback: argmax over all confirmation
           trials, labeled with confirmation_feasible=False.
     UCB selector: best_main (UCB is deterministic, confirmation skipped).

Synchronization with screening (screening_bs_stable_vs_corr.py)
---------------------------------------------------------------
  - threshold_mode="none", confidence_threshold=0.0 (no execution inertia).
  - bandit_update_action_source="executed" (post-hoc action masking semantics).
  - state_feature_columns=["state_in_position"].
  - DRO formula identical (α-смесь на equal-weight aggregate, DRO_ALPHA same).
  - 3 symbols (BTC, ETH, SOL), z_windows from screening top-1 per (algo, method).

Numerics
--------
  - Profit factor is computed from CLOSED trades (BUY/SELL pairs) on net log-pnl
    (gross_log_ret - 2*|log(1-trade_cost)|) as a DIAGNOSTIC only (not a
    constraint in this version). Per-symbol PF can be +inf if all closed trades
    are profitable; we cap that at PF_INF_CAP (10.0) before aggregation, purely
    to prevent +inf from poisoning median computation across symbols.
  - Returns are normalized to START_CAPITAL for val_full and val_h1, and to
    val_h1 ending balance for val_h2 (compound semantics, identical to
    screening's split_half_returns).
  - Total trades counted at the aggregate level (sum across all 3 symbols per
    seed, then median across seeds).
  - min_symbol_trades = per-seed min across symbols, then median across seeds
    (degeneracy guard ensuring no symbol is fully ignored by the policy).

Machine split (one line at top of file)
---------------------------------------
    Machine A:  RUN_PRESET = "A_discounted"  -> discounted_lints + discounted_linucb
    Machine B:  RUN_PRESET = "B_sliding"     -> sw_lints + sw_linucb
    Single PC:  RUN_PRESET = "ALL"           -> all 4 algorithms

All Optuna studies are in-memory (no persistence layer). All artifacts are
written as local CSV files into
    hyperparameters_optimization/stage2_dro_minimax_outputs/<RUN_LABEL>/
where <RUN_LABEL> = <RUN_PRESET>_<YYYYmmdd_HHMMSS>.

Run:
    cd D:\\PythonProjects\\BTCTrading
    python hyperparameters_optimization/stage2_hpo_dro_minimax.py
"""

from __future__ import annotations

import gc
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError as exc:
    raise ImportError("Optuna is not installed. Install: pip install optuna") from exc

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

from data_processing.functions.klines_dataloader import KlinesDataLoader
from data_processing.functions.stream_indicators import stream_TA_lib
from data_processing.functions.transform_indicators import transform_indicators_df
from data_processing.functions.rolling_z_score_clip import rolling_z_score_clip_df
from backtest.backtest_class import Backtesting


# =====================================================================
# Machine split (single string at the top)
# =====================================================================
MACHINE_PRESETS = {
    "A_discounted": ["discounted_lints", "discounted_linucb"],
    "B_sliding":    ["sw_lints", "sw_linucb"],
    "ALL":          ["discounted_lints", "discounted_linucb", "sw_lints", "sw_linucb"],
}
RUN_PRESET = "ALL"  # second machine: change to "B_sliding"

# =====================================================================
# Core protocol (synced with screening_bs_stable_vs_corr.py)
# =====================================================================
OPTUNA_SAMPLER_SEED = 20260520
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
ACTIONS = [0, 1]
OHLCV_RELATIVE_PATH = r"data\klines_data\crypto_240m_bybit_TEST.parquet"

HORIZON = 10
TRADE_COST = 0.0025
VAL_BARS = 2000
TEST_BARS = 2000
EMBARGO_BARS = 0

START_CAPITAL = 100.0
POSITION_SIZE = 0.10
MIN_HOLD_BARS = 4
COOLDOWN_BARS = 2
UPDATE_ON_VALIDATION = True
REWARD_CLIP = 0.10

CONFIDENCE_THRESHOLD = 0.0
THRESHOLD_MODE = "none"
BANDIT_UPDATE_ACTION_SOURCE = "executed"
# Bandit update policy — decision_only skips bandit consultation and pending-update
# queueing on forced-singleton bars (|feasible_set|=1 due to MIN_HOLD/COOLDOWN).
# MUST match screening's BANDIT_UPDATE_POLICY for consistent feature-pair selection.
BANDIT_UPDATE_POLICY = "decision_only"        # "all_bars" | "decision_only"
STATE_FEATURES = ["state_in_position"]
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
# Smoke test mode (set SMOKE_TEST=True to do a fast feasibility check)
# =====================================================================
# Run smoke test FIRST with SMOKE_TEST=True to verify that MIN_CLOSED_TRADES_PER_SYMBOL
# allows feasible trials in every study. The minimal constraint set now contains
# only this constraint (+ finite objective sanity); drawdown, profit factor, and
# aggregate trade count are diagnostics only. If smoke shows 0% feasible rate in
# some study, the bandit becomes too inactive on that symbol — either reduce the
# constraint floor or re-check feature pair quality. After successful smoke
# (>=1 feasible trial per study), set SMOKE_TEST=False and re-run for the full HPO.
SMOKE_TEST = True

FULL_N_TRIALS = 50
FULL_TS_SEEDS_PER_TRIAL = [3142, 3143, 3144, 3145, 3146]
FULL_UCB_SEEDS_PER_TRIAL = [3142]
FULL_N_STARTUP_TRIALS = 10

SMOKE_N_TRIALS = 16
SMOKE_TS_SEEDS_PER_TRIAL = [3142, 3143]
SMOKE_UCB_SEEDS_PER_TRIAL = [3142]
SMOKE_N_STARTUP_TRIALS = 4

if SMOKE_TEST:
    N_TRIALS = SMOKE_N_TRIALS
    TS_SEEDS_PER_TRIAL = SMOKE_TS_SEEDS_PER_TRIAL
    UCB_SEEDS_PER_TRIAL = SMOKE_UCB_SEEDS_PER_TRIAL
    N_STARTUP_TRIALS = SMOKE_N_STARTUP_TRIALS
else:
    N_TRIALS = FULL_N_TRIALS
    TS_SEEDS_PER_TRIAL = FULL_TS_SEEDS_PER_TRIAL
    UCB_SEEDS_PER_TRIAL = FULL_UCB_SEEDS_PER_TRIAL
    N_STARTUP_TRIALS = FULL_N_STARTUP_TRIALS

# =====================================================================
# Constraints — minimal set (used identically in HPO and confirmation)
# =====================================================================
#
# Methodological choice: we apply a SINGLE substantive constraint
# (MIN_CLOSED_TRADES_PER_SYMBOL) plus a sanity check (finite objective).
# Drawdown, profit factor, aggregate trade frequency are MONITORED as
# diagnostics in trial_results_all.csv but NOT used to filter trials:
#   - Transaction costs enter realised log-returns, so excessive turnover
#     is expected to hurt DRO empirically. This is monitored as a diagnostic,
#     not enforced as a constraint.
#   - Drawdown is NOT directly constrained. It is reported as a path-risk
#     diagnostic. DRO is endpoint-aware (worst-half mean return), not
#     path-aware: large intra-half drawdowns that recover by the half's
#     endpoint are not penalized by the DRO objective alone.
#   - Unprofitable trading shows up as negative half-returns; here too,
#     the relationship is empirical not enforced.
#
# Rationale for avoiding profit factor / drawdown thresholds:
#   - PF does NOT differentiate equity curves with identical PF but opposite
#     dynamics (many small wins + one big loss vs many small losses + one big
#     win both give PF=2). PF systematically biased toward mean-reverting
#     strategies; threshold could pre-judge HIE-vs-corr comparison.
#   - DD threshold is path-aware but DRO is endpoint-based; they measure
#     different things. Adding DD threshold filters policies that endpoint-
#     recover after deep drawdowns even though they DRO-rank well. This
#     filters legitimate strategies.
#   - Aggregate trade-count bounds: MIN floor is achieved through
#     MIN_CLOSED_TRADES_PER_SYMBOL × 3 symbols; MAX is monitored as
#     diagnostic but not enforced (overtrading penalty surfaces through
#     transaction cost in realised returns).
#
# Why min-symbol-trades remains as a hard constraint:
#   - Statistical floor: >= 50 closed trades per symbol gives stable PF/DRO
#     median estimates.
#   - Degeneracy protection: without it, bandit could find "trade only one
#     symbol, hold cash on others" — single-symbol DRO would not represent
#     the intended 3-symbol experiment. This is the only constraint where
#     diagnostic monitoring is insufficient.
#
INVALID_SCORE = -1_000_000.0
MIN_CLOSED_TRADES_PER_SYMBOL = 50

# DRO objective: α-смесь min и mean между двумя halves validation.
#   seed_dro_score = DRO_ALPHA * min(h1, h2) + (1 - DRO_ALPHA) * 0.5 * (h1 + h2)
# α=1.0 — жёсткий worst-case (классический DRO / CVaR(0.5) на 2-point distr.);
# α=0.5 — robust mean (mean-CVaR mixture, Rockafellar-Uryasev 2000): худшая
#         половина получает эффективный вес 0.75, лучшая 0.25;
# α=0.0 — простое среднее.
# Должно соответствовать DRO_ALPHA в screening_bs_stable_vs_corr.py для
# consistency между screening selection (top z_window per algo×method_group)
# и HPO objective.
# Pure min(h1, h2) сохраняется как diagnostic столбец seed_dro_min_aggregate_half.
DRO_ALPHA = 0.5

# PF_INF_CAP=10.0: per-symbol PF is computed as diagnostic. If a symbol has
# 0 losing closed trades (PF would be +inf), we cap at 10.0 before taking
# medians, only to prevent +inf from poisoning median aggregation. This is
# diagnostic-only; PF no longer participates in feasibility decisions.
PF_INF_CAP = 10.0

# Defaults enqueued as trial 0 (anchor to screening baseline)
SCREENING_DEFAULT_MEMORY_HORIZON_BARS = 325
SCREENING_DEFAULT_LAMBDA_PRIOR = 1.0
SCREENING_DEFAULT_NOISE_STD = 0.03
SCREENING_DEFAULT_UCB_ALPHA = 0.10
ENQUEUE_DEFAULT_TRIAL = True

# Confirmation (TS only). In smoke mode confirmation is disabled because the
# smoke test only checks feasibility of the constraint thresholds; full-blown
# confirmation re-evaluation on 25 fresh seeds is wasted compute during smoke.
RUN_TS_CONFIRMATION = (not SMOKE_TEST)
TOP_N_CONFIRMATION = 3
CONFIRMATION_SEEDS = list(range(8000, 8025))   # 25 fresh seeds (full run only)

# =====================================================================
# Search space (memory window 175..450; lambda 0.5..30 log)
# =====================================================================
SEARCH_SPACE = {
    "discounted": {
        # gamma = 1 - 1/H_mem; H_mem in [175, 450] -> one_minus_gamma in [1/450, 1/175]
        "one_minus_gamma_low":  1.0 / 450.0,
        "one_minus_gamma_high": 1.0 / 175.0,
        "lambda_prior_low": 0.5,
        "lambda_prior_high": 30.0,
    },
    "sliding": {
        "window_size_low": 175,
        "window_size_high": 450,
        "window_size_step": 25,
        "lambda_prior_low": 0.5,
        "lambda_prior_high": 30.0,
    },
    "ts": {
        "noise_std_low": 0.001,
        "noise_std_high": 0.04,
    },
    "ucb": {
        "ucb_alpha_low": 0.005,
        "ucb_alpha_high": 0.20,
    },
}
TS_BANDITS = {"discounted_lints", "sw_lints"}
UCB_BANDITS = {"discounted_linucb", "sw_linucb"}
DISCOUNTED_BANDITS = {"discounted_lints", "discounted_linucb"}
SLIDING_BANDITS = {"sw_lints", "sw_linucb"}
BANDIT_TYPE_MAP = {a: a for a in (TS_BANDITS | UCB_BANDITS)}

# =====================================================================
# Paths (in-memory Optuna; outputs are local CSV/JSON only)
# =====================================================================
FEATURE_SELECTION_ROOT = PROJECT_ROOT / "feature_selection" / "bs_stable_hie_outputs_alpha0p5_train_only"
FEATURE_SETS_PATH = FEATURE_SELECTION_ROOT / "feature_sets_hybrid_corr_bs_stable_hie_alpha0p5.json"

# Screening variant — selects WHICH screening run feeds HPO feature pairs.
# Derived automatically from BANDIT_UPDATE_ACTION_SOURCE and BANDIT_UPDATE_POLICY
# above; folder naming matches the layout produced by screening_bs_stable_vs_corr.py:
#   ("executed", "all_bars")      -> screening_execution_action            (legacy)
#   ("raw",      "all_bars")      -> screening_raw_action                  (legacy)
#   ("executed", "decision_only") -> screening_execution_action_decision_only
#   ("raw",      "decision_only") -> screening_raw_action_decision_only
# To switch HPO to a different screening variant, change the two parameters above
# (BANDIT_UPDATE_ACTION_SOURCE and BANDIT_UPDATE_POLICY).
_action_label = {"executed": "execution", "raw": "raw"}[BANDIT_UPDATE_ACTION_SOURCE]
_policy_suffix = "" if BANDIT_UPDATE_POLICY == "all_bars" else f"_{BANDIT_UPDATE_POLICY}"
SCREENING_VARIANT = f"{_action_label}_action{_policy_suffix}"
SCREENING_MINIMAX_DRO_CSV = (
    FEATURE_SELECTION_ROOT / f"screening_{SCREENING_VARIANT}" / "screening_minimax_dro.csv"
)
# Consistency check — file must exist before HPO can build feature pairs.
assert SCREENING_MINIMAX_DRO_CSV.exists(), (
    f"Screening output not found: {SCREENING_MINIMAX_DRO_CSV}\n"
    f"Make sure you ran screening_bs_stable_vs_corr.py with "
    f"BANDIT_UPDATE_ACTION_SOURCE={BANDIT_UPDATE_ACTION_SOURCE!r} and "
    f"BANDIT_UPDATE_POLICY={BANDIT_UPDATE_POLICY!r}, or change these to match "
    f"an existing screening folder."
)

HPO_ROOT = PROJECT_ROOT / "hyperparameters_optimization"
_RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_LABEL = f"{RUN_PRESET}_smoke_{_RUN_TS}" if SMOKE_TEST else f"{RUN_PRESET}_{_RUN_TS}"
OUTPUT_ROOT = HPO_ROOT / "stage2_dro_minimax_outputs"
OUTPUT_DIR = OUTPUT_ROOT / RUN_LABEL
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
STUDIES_DIR = OUTPUT_DIR / "studies"
for d in [OUTPUT_DIR, DIAGNOSTICS_DIR, STUDIES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ALGORITHMS_TO_RUN = MACHINE_PRESETS[RUN_PRESET]


# =====================================================================
# Build FEATURE_PAIRS: 8 pairs (4 algo x 2 method) auto-selected from screening
# =====================================================================
def _method_group_from_set_name(set_name: str) -> str:
    if "corr_pruned" in set_name:
        return "corr"
    if "hybrid" in set_name:
        return "hybrid"
    raise ValueError(f"Unknown method group in set_name={set_name!r}")


def build_feature_pairs() -> list[dict]:
    """Load 8 feature pairs from screening output.

    For each (algorithm, method_group) in ALGORITHMS_TO_RUN x {corr, hybrid},
    pick the z_window with the highest dro_minimax_median in screening.
    """
    if not SCREENING_MINIMAX_DRO_CSV.exists():
        raise FileNotFoundError(f"Screening output not found: {SCREENING_MINIMAX_DRO_CSV}")
    if not FEATURE_SETS_PATH.exists():
        raise FileNotFoundError(f"Feature sets JSON not found: {FEATURE_SETS_PATH}")

    with open(FEATURE_SETS_PATH, "r", encoding="utf-8") as f:
        all_feature_sets = json.load(f)

    df = pd.read_csv(SCREENING_MINIMAX_DRO_CSV)
    df["method_group"] = df["set_name"].apply(_method_group_from_set_name)
    df = df[df["algorithm"].isin(ALGORITHMS_TO_RUN)].copy()
    if df.empty:
        raise ValueError(f"No screening rows for algorithms {ALGORITHMS_TO_RUN}")

    top1 = (
        df.sort_values(["algorithm", "method_group", "dro_minimax_median"], ascending=[True, True, False])
          .groupby(["algorithm", "method_group"], as_index=False)
          .first()
    )

    pairs = []
    for _, row in top1.iterrows():
        set_name = row["set_name"]
        if set_name not in all_feature_sets:
            raise KeyError(f"Feature set {set_name} not in {FEATURE_SETS_PATH.name}")
        pairs.append({
            "algorithm": row["algorithm"],
            "method_group": row["method_group"],
            "set_name": set_name,
            "z_window": int(row["z_window"]),
            "features": list(all_feature_sets[set_name]),
            "screening_dro_minimax_median": float(row["dro_minimax_median"]),
            "screening_h1_mean_aggregate": float(row["h1_mean_aggregate"]),
            "screening_h2_mean_aggregate": float(row["h2_mean_aggregate"]),
            "screening_full_mean_aggregate": float(row["full_mean_aggregate"]),
            "screening_median_n_trade_events": float(row["median_n_trade_events"]),
        })
    return pairs


# =====================================================================
# Data pipeline
# =====================================================================
def process_indicators_for_z_window(ohlcv: pd.DataFrame, z_window: int) -> pd.DataFrame:
    """Feature preprocessing pipeline aligned 1:1 with screening_bs_stable_vs_corr.py
    (compute_features_for_z_window).

    Pipeline:
      1. stream_TA_lib (per symbol) — raw technical indicators
      2. transform_indicators_df (per symbol) — log-returns, ratios, etc.
      3. rolling_z_score_clip_df (per symbol) — z-score with clip and shift
      4. concat across symbols
      5. replace(+/-inf -> NaN) + dropna on feature+meta columns combined
         (final cleanup after ALL transformations, NOT per-stage)

    Note: no per-stage dropna intermediate. This matches screening exactly so
    that HPO sees identical row boundaries / rolling windows / train-val splits
    given the same OHLCV input.
    """
    parts = []
    for sym in SYMBOLS:
        g = ohlcv[ohlcv["symbol"] == sym].sort_values("timestamp").reset_index(drop=True).copy()
        ind = stream_TA_lib(df=g, meta_cols=META_COLS, **CONFIG_FOR_INDICATORS)
        transformed = transform_indicators_df(df=ind, meta_cols=META_COLS)
        zdf = rolling_z_score_clip_df(
            df=transformed, meta_cols=META_COLS,
            window=z_window, clip_value=5.0, shift_by_one=True,
        )
        parts.append(zdf)
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    feature_cols = [c for c in out.columns if c not in META_COLS]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=feature_cols + META_COLS).reset_index(drop=True)
    return out


def split_train_val_test(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_parts, val_parts, test_parts = [], [], []
    for sym in SYMBOLS:
        g = df[df["symbol"] == sym].sort_values("timestamp").reset_index(drop=True).copy()
        n = len(g)
        if n <= VAL_BARS + TEST_BARS + EMBARGO_BARS:
            raise ValueError(f"Too few rows for {sym}: n={n}")
        test_start = n - TEST_BARS
        val_start = test_start - VAL_BARS
        train_end = max(val_start - EMBARGO_BARS, 0)
        train_parts.append(g.iloc[:train_end].copy())
        val_parts.append(g.iloc[val_start:test_start].copy())
        test_parts.append(g.iloc[test_start:].copy())
    return {
        "train": pd.concat(train_parts, ignore_index=True).sort_values(["symbol","timestamp"]).reset_index(drop=True),
        "val":   pd.concat(val_parts,   ignore_index=True).sort_values(["symbol","timestamp"]).reset_index(drop=True),
        "test":  pd.concat(test_parts,  ignore_index=True).sort_values(["symbol","timestamp"]).reset_index(drop=True),
    }


# =====================================================================
# Per-symbol metrics — full val + val_first / val_second halves
# =====================================================================
def _store(bt, mode: str, name: str):
    if mode not in {"train", "val"}:
        raise ValueError(mode)
    return getattr(bt, f"{name}_{mode}")


_BAR_SECONDS = 4 * 3600  # 4-hour bars (240-minute interval)


def extract_trade_diagnostics(trade_log: list[dict]) -> dict:
    """Closed-trade data from BUY/SELL pairs in a trade log.

    Returns a dict of numpy arrays (each indexed by closed trade):
      pnls_after_cost           — net log PnL = log(exit/entry) + 2*log(1 - trade_cost)
      holding_bars              — bars elapsed between BUY and SELL of the same trade
      time_between_entries_bars — bars between consecutive BUY events
    """
    empty = {
        "pnls_after_cost":           np.array([], dtype=float),
        "holding_bars":              np.array([], dtype=float),
        "time_between_entries_bars": np.array([], dtype=float),
    }
    if not trade_log:
        return empty
    events = pd.DataFrame(trade_log)
    if events.empty or "event" not in events.columns:
        return empty
    events = events.sort_values("timestamp").copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])

    pnls, holding, buy_ts_seq = [], [], []
    open_entry_price = None
    open_entry_ts = None
    for _, row in events.iterrows():
        if row["event"] == "BUY":
            open_entry_price = float(row["price"])
            open_entry_ts = row["timestamp"]
            buy_ts_seq.append(open_entry_ts)
        elif row["event"] == "SELL" and open_entry_price is not None:
            exit_price = float(row["price"])
            exit_ts = row["timestamp"]
            gross = np.log(exit_price / open_entry_price)
            net = gross + 2.0 * np.log(1.0 - TRADE_COST)
            pnls.append(net)
            hold_bars = (exit_ts - open_entry_ts).total_seconds() / _BAR_SECONDS
            holding.append(hold_bars)
            open_entry_price = None
            open_entry_ts = None
    tbi = []
    for i in range(1, len(buy_ts_seq)):
        diff_bars = (buy_ts_seq[i] - buy_ts_seq[i - 1]).total_seconds() / _BAR_SECONDS
        tbi.append(diff_bars)
    return {
        "pnls_after_cost":           np.asarray(pnls, dtype=float),
        "holding_bars":              np.asarray(holding, dtype=float),
        "time_between_entries_bars": np.asarray(tbi, dtype=float),
    }


def _holding_stats(holding: np.ndarray) -> dict:
    """Per-symbol holding-time statistics, including share of trades closed
    before / after HORIZON bars (reward maturity boundary)."""
    n = len(holding)
    if n == 0:
        return {
            "median_holding_bars": float("nan"),
            "mean_holding_bars":   float("nan"),
            "p25_holding_bars":    float("nan"),
            "p75_holding_bars":    float("nan"),
            "share_trades_closed_before_horizon": float("nan"),
            "share_trades_closed_after_horizon":  float("nan"),
        }
    return {
        "median_holding_bars": float(np.median(holding)),
        "mean_holding_bars":   float(np.mean(holding)),
        "p25_holding_bars":    float(np.percentile(holding, 25)),
        "p75_holding_bars":    float(np.percentile(holding, 75)),
        "share_trades_closed_before_horizon": float((holding < HORIZON).mean()),
        "share_trades_closed_after_horizon":  float((holding >= HORIZON).mean()),
    }


def _time_between_entries(tbi: np.ndarray) -> float:
    return float(np.median(tbi)) if len(tbi) > 0 else float("nan")


def _profit_factor_from_pnls(pnls: np.ndarray) -> float:
    """Per-symbol profit factor from closed-trade net PnL.

    PF = sum(positive pnl) / abs(sum(negative pnl)). If there are no negative
    closed trades but at least one positive: returns +inf (will be capped at
    PF_INF_CAP downstream before taking medians). If there are no trades at
    all: returns NaN.
    """
    n = len(pnls)
    if n == 0:
        return float("nan")
    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss = float(abs(pnls[pnls < 0].sum()))
    if gross_loss > 0:
        return gross_profit / gross_loss
    return float("inf") if gross_profit > 0 else 1.0


def compute_symbol_metrics_val(bt, sym: str) -> dict:
    """Full-validation metrics for one symbol.

    return_pct = compound return from START_CAPITAL to balance[-1] (not balance[0]).
    """
    balance = np.asarray(_store(bt, "val", "balance").get(sym, []), dtype=float)
    actions = np.asarray(_store(bt, "val", "actions").get(sym, []), dtype=float)
    trade_log = _store(bt, "val", "trade_log").get(sym, [])
    decision_log_records = _store(bt, "val", "decision_log").get(sym, [])

    empty_hold = _holding_stats(np.array([], dtype=float))
    if len(balance) == 0:
        return {"symbol": sym, "mode": "val", "n_decisions": 0,
                "return_pct": float("nan"), "drawdown_abs": float("nan"),
                "trades": 0, "profit_factor": float("nan"), "is_active": False,
                "executed_action_1_ratio": float("nan"),
                "constraint_applied_ratio": float("nan"),
                "median_time_between_entries": float("nan"),
                **empty_hold}

    end_val = float(balance[-1])
    return_pct = 100.0 * (end_val / float(START_CAPITAL) - 1.0)  # normalized to START_CAPITAL
    running_max = np.maximum.accumulate(balance)
    drawdown = balance / (running_max + 1e-12) - 1.0
    max_dd_pct = float(drawdown.min() * 100.0)

    diag = extract_trade_diagnostics(trade_log)
    pnls = diag["pnls_after_cost"]
    holding = diag["holding_bars"]
    tbi = diag["time_between_entries_bars"]
    n_trades = int(len(pnls))
    pf = _profit_factor_from_pnls(pnls)
    hold_stats = _holding_stats(holding)
    median_tbi = _time_between_entries(tbi)

    # Decision-only-aware diagnostics from decision_log:
    #   constraint_applied_ratio: legacy — share of bars where constraint_type is set
    #   forced_singleton_ratio:   share of bars with |feasible_set| == 1
    #   decision_point_ratio:     share of bars with |feasible_set| >= 2
    #   bandit_consulted_ratio:   share of bars where select_action was called
    #   pending_update_queued_ratio: share of bars where delayed-update was queued
    constraint_applied_ratio = float("nan")
    forced_singleton_ratio = float("nan")
    decision_point_ratio = float("nan")
    bandit_consulted_ratio = float("nan")
    pending_update_queued_ratio = float("nan")
    if decision_log_records:
        dl = pd.DataFrame(decision_log_records)
        if "constraint_applied" in dl.columns:
            constraint_applied_ratio = float(dl["constraint_applied"].astype(bool).mean())
        if "feasible_set_size" in dl.columns:
            sizes = pd.to_numeric(dl["feasible_set_size"], errors="coerce")
            forced_singleton_ratio = float((sizes == 1).mean())
            decision_point_ratio = float((sizes >= 2).mean())
        if "is_decision_point" in dl.columns:
            # If is_decision_point present, prefer it (more direct)
            decision_point_ratio = float(dl["is_decision_point"].astype(bool).mean())
            forced_singleton_ratio = 1.0 - decision_point_ratio
        if "bandit_consulted" in dl.columns:
            bandit_consulted_ratio = float(dl["bandit_consulted"].astype(bool).mean())
        if "pending_update_queued" in dl.columns:
            pending_update_queued_ratio = float(dl["pending_update_queued"].astype(bool).mean())

    return {
        "symbol": sym,
        "mode": "val",
        "n_decisions": int(len(actions)),
        "return_pct": float(return_pct),
        "drawdown_abs": abs(max_dd_pct),
        "trades": n_trades,
        "profit_factor": pf,
        "is_active": n_trades > 0,
        "executed_action_1_ratio": float(np.mean(actions == 1)) if len(actions) else float("nan"),
        "constraint_applied_ratio": constraint_applied_ratio,
        # decision-only-aware diagnostics (NEW)
        "forced_singleton_ratio": forced_singleton_ratio,
        "decision_point_ratio": decision_point_ratio,
        "bandit_consulted_ratio": bandit_consulted_ratio,
        "pending_update_queued_ratio": pending_update_queued_ratio,
        "median_time_between_entries": median_tbi,
        **hold_stats,
    }


def compute_symbol_metrics_half(bt, sym: str, period: str, start_idx: int, end_idx: int) -> dict:
    """Slice metrics for val_first / val_second halves (post-hoc, no rerun).

    val_first return is normalized to START_CAPITAL.
    val_second return is normalized to the val_h1 ending balance (compound,
    identical to screening's split_half_returns).
    """
    balance = np.asarray(_store(bt, "val", "balance").get(sym, []), dtype=float)
    if len(balance) == 0 or end_idx <= start_idx:
        return {"symbol": sym, "mode": period, "return_pct": float("nan"),
                "trades": 0, "drawdown_abs": float("nan"),
                "profit_factor": float("nan"), "is_active": False}

    bal_slice = balance[start_idx:end_idx]
    times = pd.to_datetime(pd.Series(_store(bt, "val", "times").get(sym, []))).iloc[start_idx:end_idx]
    start_ts, end_ts = times.iloc[0], times.iloc[-1]

    if period == "val_first":
        start_val = float(START_CAPITAL)
    else:  # val_second
        start_val = float(bal_slice[0])   # = balance[boundary] = val_h1 ending balance
    end_val = float(bal_slice[-1])
    return_pct = 100.0 * (end_val / start_val - 1.0) if start_val > 0 else float("nan")

    running_max = np.maximum.accumulate(bal_slice)
    drawdown = bal_slice / (running_max + 1e-12) - 1.0
    max_dd_pct = float(drawdown.min() * 100.0)

    trade_log = _store(bt, "val", "trade_log").get(sym, [])
    events = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    if not events.empty and "timestamp" in events.columns:
        events = events.copy()
        events["timestamp"] = pd.to_datetime(events["timestamp"])
        events = events[(events["timestamp"] >= start_ts) & (events["timestamp"] <= end_ts)]
    diag = extract_trade_diagnostics(events.to_dict("records") if not events.empty else [])
    pnls = diag["pnls_after_cost"]
    holding = diag["holding_bars"]
    tbi = diag["time_between_entries_bars"]
    n_trades = int(len(pnls))
    pf = _profit_factor_from_pnls(pnls)
    hold_stats = _holding_stats(holding)
    median_tbi = _time_between_entries(tbi)

    return {
        "symbol": sym,
        "mode": period,
        "return_pct": float(return_pct),
        "drawdown_abs": abs(max_dd_pct),
        "trades": n_trades,
        "profit_factor": pf,
        "is_active": n_trades > 0,
        "median_time_between_entries": median_tbi,
        **hold_stats,
    }


# =====================================================================
# Seed summary + aggregate across seeds
# =====================================================================
def _pf_cap_series(s: pd.Series) -> pd.Series:
    """Cap +inf at PF_INF_CAP; map -inf -> NaN -> 0.0 (no negative PF defined)."""
    return (s.replace(np.inf, PF_INF_CAP).replace(-np.inf, np.nan).fillna(0.0))


def summarize_seed(symbol_metrics: pd.DataFrame, seed: int) -> dict:
    """Aggregate one seed's per-symbol metrics into a single seed-level summary.

    Key fields (used by HPO objective / constraints):
        seed_dro_score              = α*min(h1, h2) + (1-α)*0.5*(h1+h2)  [α=DRO_ALPHA]
            -- MAIN objective: α-смесь temporal worst-case и среднего между
               двумя val halves (см. DRO_ALPHA в config). При α=1 это классический
               temporal maximin DRO; при α=0.5 mean-CVaR mixture.
        seed_dro_min_aggregate_half = min(mean_symbol_h1, mean_symbol_h2)
            -- DIAGNOSTIC: pure worst-case (что было бы при α=1.0).
        median_profit_factor        = median across symbols of capped PF
        total_trades                = sum across symbols (aggregate)
        min_symbol_trades           = min across symbols (degeneracy probe)
        median_drawdown_abs         = median across symbols of |max_drawdown_%|
    """
    val = symbol_metrics[symbol_metrics["mode"].eq("val")].copy()
    val_first = symbol_metrics[symbol_metrics["mode"].eq("val_first")].copy()
    val_second = symbol_metrics[symbol_metrics["mode"].eq("val_second")].copy()
    if val.empty:
        raise ValueError("No val rows in symbol_metrics")

    # Cap +inf -> PF_INF_CAP, then map remaining non-finite to 0.0 for median aggregation
    val["pf_clean"] = _pf_cap_series(val["profit_factor"])

    mean_symbol_h1 = float(val_first["return_pct"].mean()) if not val_first.empty else float("nan")
    mean_symbol_h2 = float(val_second["return_pct"].mean()) if not val_second.empty else float("nan")
    if np.isfinite(mean_symbol_h1) and np.isfinite(mean_symbol_h2):
        seed_dro_pure_min = float(min(mean_symbol_h1, mean_symbol_h2))         # diagnostic
        seed_dro = float(
            DRO_ALPHA * seed_dro_pure_min
            + (1.0 - DRO_ALPHA) * 0.5 * (mean_symbol_h1 + mean_symbol_h2)
        )                                                                       # MAIN objective
    else:
        seed_dro_pure_min = float("nan")
        seed_dro = float("nan")

    # Holding-time + cycle diagnostics (median across symbols of per-symbol stats)
    def _med_or_nan(col):
        if col not in val.columns:
            return float("nan")
        ser = pd.to_numeric(val[col], errors="coerce").dropna()
        return float(ser.median()) if len(ser) > 0 else float("nan")

    return {
        "seed": int(seed),
        "n_symbols": int(val["symbol"].nunique()),
        # MAIN DRO objective per seed: α-смесь (α=DRO_ALPHA)
        "seed_dro_score": seed_dro,
        "dro_alpha": float(DRO_ALPHA),
        # DIAGNOSTIC: pure min(h1, h2) — что было бы при α=1.0
        "seed_dro_min_aggregate_half": seed_dro_pure_min,
        "mean_symbol_h1_return_pct": mean_symbol_h1,
        "mean_symbol_h2_return_pct": mean_symbol_h2,
        # Diagnostic full-val metrics
        "mean_symbol_full_return_pct": float(val["return_pct"].mean()),
        "median_symbol_full_return_pct": float(val["return_pct"].median()),
        # Risk
        "median_drawdown_abs": float(val["drawdown_abs"].median()),
        "max_drawdown_abs": float(val["drawdown_abs"].max()),
        # PF (median across symbols of capped per-symbol PF)
        "median_profit_factor": float(val["pf_clean"].median()),
        # Trades
        "total_trades": int(val["trades"].sum()),          # aggregate sum across symbols
        "min_symbol_trades": int(val["trades"].min()),     # degeneracy probe (per-seed min)
        # Action diagnostics
        "mean_executed_action_1_ratio": float(val["executed_action_1_ratio"].mean()),
        "mean_constraint_applied_ratio": float(val["constraint_applied_ratio"].mean()),
        # decision-only-aware (median across symbols)
        "mean_forced_singleton_ratio":      _med_or_nan("forced_singleton_ratio"),
        "mean_decision_point_ratio":        _med_or_nan("decision_point_ratio"),
        "mean_bandit_consulted_ratio":      _med_or_nan("bandit_consulted_ratio"),
        "mean_pending_update_queued_ratio": _med_or_nan("pending_update_queued_ratio"),
        # Holding-time & cycle diagnostics (median across symbols)
        "median_holding_bars": _med_or_nan("median_holding_bars"),
        "mean_holding_bars":   _med_or_nan("mean_holding_bars"),
        "p25_holding_bars":    _med_or_nan("p25_holding_bars"),
        "p75_holding_bars":    _med_or_nan("p75_holding_bars"),
        "share_trades_closed_before_horizon": _med_or_nan("share_trades_closed_before_horizon"),
        "share_trades_closed_after_horizon":  _med_or_nan("share_trades_closed_after_horizon"),
        "median_time_between_entries": _med_or_nan("median_time_between_entries"),
    }


def aggregate_seeds(seed_summaries: list[dict]) -> dict:
    """Median-across-seeds aggregation of per-seed summaries."""
    s = pd.DataFrame(seed_summaries)
    return {
        "n_seeds": int(s["seed"].nunique()),
        "seeds_used": "|".join(str(int(v)) for v in sorted(s["seed"].unique())),
        "dro_alpha": float(DRO_ALPHA),
        # Primary DRO objective = median across seeds of seed_dro_score (α-смесь)
        "dro_minimax_median": float(s["seed_dro_score"].median()),
        "dro_minimax_mean":   float(s["seed_dro_score"].mean()),
        "dro_minimax_std":    float(s["seed_dro_score"].std(ddof=1)) if len(s) > 1 else 0.0,
        "dro_minimax_min":    float(s["seed_dro_score"].min()),
        "dro_minimax_max":    float(s["seed_dro_score"].max()),
        # Diagnostic: pure min(h1, h2) baseline — что было бы при α=1.0
        "dro_pure_min_median": float(s["seed_dro_min_aggregate_half"].median()),
        "dro_pure_min_mean":   float(s["seed_dro_min_aggregate_half"].mean()),
        "dro_pure_min_std":    float(s["seed_dro_min_aggregate_half"].std(ddof=1)) if len(s) > 1 else 0.0,
        # Half-level diagnostics (median across seeds of equal-weight aggregate per half)
        "h1_mean_aggregate_median": float(s["mean_symbol_h1_return_pct"].median()),
        "h2_mean_aggregate_median": float(s["mean_symbol_h2_return_pct"].median()),
        "full_mean_aggregate_median": float(s["mean_symbol_full_return_pct"].median()),
        # Constraint inputs
        "median_max_drawdown_abs":   float(s["median_drawdown_abs"].median()),
        "median_profit_factor":      float(s["median_profit_factor"].median()),
        "median_total_trades":       float(s["total_trades"].median()),
        "median_min_symbol_trades":  float(s["min_symbol_trades"].median()),
        # Holding-time & cycle diagnostics — median across seeds of per-seed median across symbols
        "median_holding_bars_overall": float(s["median_holding_bars"].median())
            if "median_holding_bars" in s.columns else float("nan"),
        "mean_holding_bars_overall":   float(s["mean_holding_bars"].median())
            if "mean_holding_bars" in s.columns else float("nan"),
        "p25_holding_bars_overall":    float(s["p25_holding_bars"].median())
            if "p25_holding_bars" in s.columns else float("nan"),
        "p75_holding_bars_overall":    float(s["p75_holding_bars"].median())
            if "p75_holding_bars" in s.columns else float("nan"),
        "share_trades_closed_before_horizon_overall": float(s["share_trades_closed_before_horizon"].median())
            if "share_trades_closed_before_horizon" in s.columns else float("nan"),
        "share_trades_closed_after_horizon_overall":  float(s["share_trades_closed_after_horizon"].median())
            if "share_trades_closed_after_horizon" in s.columns else float("nan"),
        "median_time_between_entries_overall": float(s["median_time_between_entries"].median())
            if "median_time_between_entries" in s.columns else float("nan"),
        # decision-only-aware aggregations (median across seeds of per-seed median across symbols)
        "forced_singleton_ratio_overall":      float(s["mean_forced_singleton_ratio"].median())
            if "mean_forced_singleton_ratio" in s.columns else float("nan"),
        "decision_point_ratio_overall":        float(s["mean_decision_point_ratio"].median())
            if "mean_decision_point_ratio" in s.columns else float("nan"),
        "bandit_consulted_ratio_overall":      float(s["mean_bandit_consulted_ratio"].median())
            if "mean_bandit_consulted_ratio" in s.columns else float("nan"),
        "pending_update_queued_ratio_overall": float(s["mean_pending_update_queued_ratio"].median())
            if "mean_pending_update_queued_ratio" in s.columns else float("nan"),
    }


# =====================================================================
# DRO objective + Optuna constraints_func (2-tuple, minimal constraint set)
# =====================================================================
def dro_objective_from_agg(agg: dict) -> tuple[float, dict, tuple[float, ...]]:
    """Return (score, details, constraints_tuple).

    Objective: dro_minimax_median (to maximize).
    Constraints (Optuna convention: value <= 0 feasible, value > 0 violation):
      (1) min_symbol_trades: median_min_symbol_trades >= MIN_CLOSED_TRADES_PER_SYMBOL
      (2) finite objective

    DD / PF / total-trades are NOT constraints; they are reported as
    diagnostics in `details` (observed_*). Transaction costs enter realised
    returns, so excessive turnover is expected to hurt DRO empirically, but
    turnover is monitored — not constrained. Drawdown is reported as a
    path-risk diagnostic (DRO objective is endpoint-aware, not path-aware).
    See module docstring for the full methodological rationale.
    """
    obj = float(agg["dro_minimax_median"])
    pure_min = float(agg.get("dro_pure_min_median", float("nan")))
    dd  = float(agg["median_max_drawdown_abs"])
    pf  = float(agg["median_profit_factor"])
    trades = float(agg["median_total_trades"])
    min_sym = float(agg["median_min_symbol_trades"])

    BIG = 1_000_000.0
    cv_min_sym = (MIN_CLOSED_TRADES_PER_SYMBOL - min_sym) if np.isfinite(min_sym) else BIG
    cv_finite  = 0.0 if np.isfinite(obj) else BIG

    constraints_tuple = (cv_min_sym, cv_finite)
    valid = bool(all(c <= 0.0 for c in constraints_tuple))

    details = {
        "valid": valid,
        "objective_metric_name": "dro_minimax_median",
        "objective_score": obj if np.isfinite(obj) else INVALID_SCORE,
        "constraints_values": list(constraints_tuple),
        "cv_min_symbol_trades": float(cv_min_sym),
        "cv_finite_objective":  float(cv_finite),
        # Diagnostic-only (NOT used for feasibility decisions)
        "observed_dro_minimax_median":      obj,
        "observed_dro_pure_min_median":     pure_min,   # α=1.0 baseline
        "dro_alpha":                         float(DRO_ALPHA),
        "observed_median_max_drawdown_abs": dd,
        "observed_median_profit_factor":    pf,
        "observed_median_total_trades":     trades,
        "observed_median_min_symbol_trades": min_sym,
    }
    return float(details["objective_score"]), details, constraints_tuple


def optuna_constraints_func(frozen_trial):
    """Read constraint tuple stored in trial.user_attrs; default to all-violating."""
    values = frozen_trial.user_attrs.get("constraints_values", None)
    expected_len = 2
    if values is None:
        return (1_000_000.0,) * expected_len
    cleaned = []
    for v in values:
        try:
            v = float(v)
            if not np.isfinite(v):
                v = 1_000_000.0
        except Exception:
            v = 1_000_000.0
        cleaned.append(v)
    if len(cleaned) != expected_len:
        return (1_000_000.0,) * expected_len
    return tuple(cleaned)


def is_feasible(constraints_tuple: tuple[float, ...]) -> bool:
    return all(c <= 0.0 for c in constraints_tuple)


# =====================================================================
# Bandit params: suggest / default / rebuild
# =====================================================================
def suggest_bandit_params(trial: optuna.Trial, algorithm: str) -> dict:
    params: dict = {"bandit_type": BANDIT_TYPE_MAP[algorithm], "reward_clip": REWARD_CLIP}
    family_key = "discounted" if algorithm in DISCOUNTED_BANDITS else "sliding"
    params["lambda_prior"] = trial.suggest_float(
        "lambda_prior",
        SEARCH_SPACE[family_key]["lambda_prior_low"],
        SEARCH_SPACE[family_key]["lambda_prior_high"],
        log=True,
    )
    if algorithm in DISCOUNTED_BANDITS:
        omg = trial.suggest_float(
            "one_minus_gamma",
            SEARCH_SPACE["discounted"]["one_minus_gamma_low"],
            SEARCH_SPACE["discounted"]["one_minus_gamma_high"],
            log=True,
        )
        params["discount_factor"] = 1.0 - omg
        trial.set_user_attr("memory_horizon_bars", 1.0 / omg)
    else:
        ws = trial.suggest_int(
            "window_size",
            SEARCH_SPACE["sliding"]["window_size_low"],
            SEARCH_SPACE["sliding"]["window_size_high"],
            step=SEARCH_SPACE["sliding"]["window_size_step"],
        )
        params["window_size"] = ws
        trial.set_user_attr("memory_horizon_bars", float(ws))

    if algorithm in TS_BANDITS:
        params["noise_std"] = trial.suggest_float(
            "noise_std",
            SEARCH_SPACE["ts"]["noise_std_low"],
            SEARCH_SPACE["ts"]["noise_std_high"],
            log=True,
        )
    else:
        params["ucb_alpha"] = trial.suggest_float(
            "ucb_alpha",
            SEARCH_SPACE["ucb"]["ucb_alpha_low"],
            SEARCH_SPACE["ucb"]["ucb_alpha_high"],
            log=True,
        )
    return params


def default_trial_params(algorithm: str) -> dict:
    """Screening defaults to enqueue as trial 0 (memory_horizon=325, lambda=1.0)."""
    if algorithm in DISCOUNTED_BANDITS:
        p = {
            "one_minus_gamma": 1.0 / float(SCREENING_DEFAULT_MEMORY_HORIZON_BARS),
            "lambda_prior": float(SCREENING_DEFAULT_LAMBDA_PRIOR),
        }
    else:
        p = {
            "window_size": int(SCREENING_DEFAULT_MEMORY_HORIZON_BARS),
            "lambda_prior": float(SCREENING_DEFAULT_LAMBDA_PRIOR),
        }
    if algorithm in TS_BANDITS:
        p["noise_std"] = float(SCREENING_DEFAULT_NOISE_STD)
    else:
        p["ucb_alpha"] = float(SCREENING_DEFAULT_UCB_ALPHA)
    return p


def rebuild_bandit_params_from_trial_row(row: pd.Series) -> dict:
    """Rebuild Backtesting-ready bandit config from a saved trial_results row."""
    algorithm = row["bandit_name"]
    p = {
        "bandit_type": BANDIT_TYPE_MAP[algorithm],
        "lambda_prior": float(row["param_lambda_prior"]),
        "reward_clip": REWARD_CLIP,
    }
    if algorithm in DISCOUNTED_BANDITS:
        p["discount_factor"] = float(row["param_discount_factor"])
    else:
        p["window_size"] = int(row["param_window_size"])
    if algorithm in TS_BANDITS:
        p["noise_std"] = float(row["param_noise_std"])
    else:
        p["ucb_alpha"] = float(row["param_ucb_alpha"])
    return p


# =====================================================================
# Single backtest for one seed -> per-symbol metrics + seed summary
# =====================================================================
def run_backtest_for_seed(
    feature_pair: dict,
    bandit_params: dict,
    seed: int,
    datasets_by_z: dict[int, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, dict]:
    z = int(feature_pair["z_window"])
    features = list(feature_pair["features"])
    train_df = datasets_by_z[z]["train"]
    val_df = datasets_by_z[z]["val"]

    cfg = dict(bandit_params)
    cfg["n_features"] = len(features) + len(STATE_FEATURES)
    cfg["actions"] = ACTIONS
    cfg["seed"] = int(seed)

    bt = Backtesting(
        meta_cols=META_COLS,
        feature_columns=features,
        config_for_bandit=cfg,
        trade_cost=TRADE_COST,
        seed=int(seed),
        update_on_validation=UPDATE_ON_VALIDATION,
        horizon=HORIZON,
        min_hold_bars=MIN_HOLD_BARS,
        cooldown_bars=COOLDOWN_BARS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        alpha_out=0.5,
        state_feature_columns=STATE_FEATURES,
        use_symbol_seed_offset=True,
        threshold_mode=THRESHOLD_MODE,
        bandit_update_action_source=BANDIT_UPDATE_ACTION_SOURCE,
        bandit_update_policy=BANDIT_UPDATE_POLICY,
    )
    bt.backtest(
        dataframe_train=train_df,
        dataframe_val=val_df,
        symbols=SYMBOLS,
        start_capital=START_CAPITAL,
        position_size=POSITION_SIZE,
    )

    rows = []
    for sym in SYMBOLS:
        rows.append(compute_symbol_metrics_val(bt, sym))
        bal = _store(bt, "val", "balance").get(sym, [])
        if len(bal) >= 4:
            boundary = len(bal) // 2 - 1
            rows.append(compute_symbol_metrics_half(bt, sym, "val_first",  0,        boundary + 1))
            rows.append(compute_symbol_metrics_half(bt, sym, "val_second", boundary, len(bal)))

    symbol_metrics = pd.DataFrame(rows)
    seed_summary = summarize_seed(symbol_metrics, seed=seed)
    return symbol_metrics, seed_summary


def append_jsonl(path: Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


# =====================================================================
# Selectors
#   HPO main:     argmax dro_minimax_median over feasible completed trials
#                 (fallback: argmax over all completed trials, infeasible flag)
#   TS confirm:   argmax confirmation_dro_minimax_median over CONFIRMATION-COMPLETE
#                 AND confirmation-feasible trials (priority: complete -> feasible
#                 -> score; explicit fallback flags expose partial/infeasible cases)
#   UCB final:    HPO main best (UCB is deterministic, no confirmation)
# =====================================================================
def pick_best_feasible_or_fallback(
    trials_with_scores: list[tuple[Any, float, bool]],
) -> tuple[Any, bool]:
    """Generic feasible-first argmax with explicit fallback.

    Args:
        trials_with_scores: list of (item, score, is_feasible) tuples.
    Returns:
        (selected_item, is_fallback_infeasible).
        If any item is feasible: returns argmax score among feasible, is_fallback=False.
        Else if any item exists: returns argmax score among all, is_fallback=True.
        Else: returns (None, False).
    """
    feasible = [(it, sc) for (it, sc, ok) in trials_with_scores if ok]
    if feasible:
        best_item = max(feasible, key=lambda t: t[1])[0]
        return best_item, False
    fallback = [(it, sc) for (it, sc, _) in trials_with_scores]
    if fallback:
        best_item = max(fallback, key=lambda t: t[1])[0]
        return best_item, True
    return None, False


def pick_best_complete_feasible_or_fallback(
    trials_with_scores: list[tuple[Any, float, bool, bool]],
) -> tuple[Any, bool, bool]:
    """Three-tier selector for confirmation: complete -> feasible -> score.

    Used for picking confirmation winner where confirmation must run on all
    CONFIRMATION_SEEDS to be considered "complete". Methodological rationale:
    confirmation_dro_minimax_median is taken as a median across seeds, so
    trials evaluated on N=25 seeds are not directly comparable to trials
    evaluated on N=24 seeds. We prefer complete trials strictly.

    Args:
        trials_with_scores: list of (item, score, is_feasible, is_complete) tuples.

    Returns:
        (selected_item, is_fallback_partial_or_infeasible, is_fallback_infeasible).

    Selection priority:
        1) complete=True AND feasible=True   -> argmax score; flags (False, False)
        2) complete=True only                -> argmax score; flags (True,  True)
           (no feasible-complete trial; we pick complete-but-infeasible)
        3) any item                          -> argmax score; flags (True,  True)
           (no complete trial at all; we pick partial)
    Empty input -> (None, False, False).
    """
    if not trials_with_scores:
        return None, False, False
    complete_feasible = [(it, sc) for (it, sc, ok, comp) in trials_with_scores if comp and ok]
    if complete_feasible:
        best_item = max(complete_feasible, key=lambda t: t[1])[0]
        return best_item, False, False
    complete_only = [(it, sc) for (it, sc, _, comp) in trials_with_scores if comp]
    if complete_only:
        best_item = max(complete_only, key=lambda t: t[1])[0]
        return best_item, True, True
    # fallback to partial trials
    partial_feasible = [(it, sc) for (it, sc, ok, _) in trials_with_scores if ok]
    if partial_feasible:
        best_item = max(partial_feasible, key=lambda t: t[1])[0]
        return best_item, True, False
    fallback = [(it, sc) for (it, sc, _, _) in trials_with_scores]
    best_item = max(fallback, key=lambda t: t[1])[0]
    return best_item, True, True


# =====================================================================
# Main
# =====================================================================
def main():
    t_start = datetime.now()

    mode_tag = "SMOKE" if SMOKE_TEST else "FULL"
    print("=" * 100)
    print(f"Stage 2 HPO [{mode_tag}] - DRO maximin | preset = {RUN_PRESET} | run_label = {RUN_LABEL}")
    print("=" * 100)
    if SMOKE_TEST:
        print(">>> SMOKE TEST mode: reduced budget for feasibility verification.")
        print(f"    Trials: {SMOKE_N_TRIALS}, TS seeds: {SMOKE_TS_SEEDS_PER_TRIAL}, "
              f"UCB seeds: {SMOKE_UCB_SEEDS_PER_TRIAL}, confirmation: disabled.")
        print(f"    Goal: verify MIN_CLOSED_TRADES_PER_SYMBOL ({MIN_CLOSED_TRADES_PER_SYMBOL}) "
              "allows feasible trials. Drawdown/PF/trade-count are diagnostics only.")
        print(f"    After successful smoke (>=1 feasible per study) set SMOKE_TEST=False.")
        print("=" * 100)
    print(f"Algorithms:               {ALGORITHMS_TO_RUN}")
    print(f"Symbols:                  {SYMBOLS}")
    print(f"State features:           {STATE_FEATURES}")
    print(f"Confidence threshold:     {CONFIDENCE_THRESHOLD}  (threshold_mode={THRESHOLD_MODE!r})")
    print(f"Update action source:     {BANDIT_UPDATE_ACTION_SOURCE!r}")
    print(f"Update policy:            {BANDIT_UPDATE_POLICY!r}")
    print(f"Screening variant:        {SCREENING_VARIANT!r}  (path: {SCREENING_MINIMAX_DRO_CSV.parent.name})")
    print(f"DRO_ALPHA:                {DRO_ALPHA}  "
          f"(α=1 → hard min, α=0.5 → mean-CVaR mixture, α=0 → mean)")
    print(f"DRO objective:            median_seed("
          f"{DRO_ALPHA}*min(h1,h2) + {1.0-DRO_ALPHA}*0.5*(h1+h2))   "
          f"[h_k = mean_symbol(val_h_k)]")
    print(f"Diagnostic (parallel):    median_seed(min(h1,h2))   "
          f"[pure α=1.0 baseline — saved as dro_pure_min_median]")
    print(f"Constraints (2-tuple, minimal set):")
    print(f"   (1) median_min_symbol_trades  >= {MIN_CLOSED_TRADES_PER_SYMBOL}  (statistical floor + degeneracy guard)")
    print(f"   (2) finite objective")
    print(f"Diagnostics computed (NOT enforced): drawdown, profit_factor, aggregate trade count")
    print(f"PF_INF_CAP (diagnostic only):  {PF_INF_CAP}")
    print(f"N trials per study:        {N_TRIALS}  (startup {N_STARTUP_TRIALS})")
    print(f"TS seeds:                  {TS_SEEDS_PER_TRIAL}")
    print(f"UCB seeds:                 {UCB_SEEDS_PER_TRIAL}")
    print(f"Confirmation:              {('top-' + str(TOP_N_CONFIRMATION) + ' x ' + str(len(CONFIRMATION_SEEDS)) + ' fresh seeds (TS only)') if RUN_TS_CONFIRMATION else 'disabled'}")
    print(f"Output dir:                {OUTPUT_DIR}")
    print("=" * 100)

    # ---- Load FEATURE_PAIRS from screening top-1 per (algo, method)
    feature_pairs = build_feature_pairs()
    feature_pairs_table = pd.DataFrame([
        {
            "algorithm": p["algorithm"],
            "method_group": p["method_group"],
            "set_name": p["set_name"],
            "z_window": p["z_window"],
            "n_market_features": len(p["features"]),
            "n_state_features": len(STATE_FEATURES),
            "n_total_features": len(p["features"]) + len(STATE_FEATURES),
            "screening_dro": p["screening_dro_minimax_median"],
            "screening_h1": p["screening_h1_mean_aggregate"],
            "screening_h2": p["screening_h2_mean_aggregate"],
            "screening_full": p["screening_full_mean_aggregate"],
            "screening_median_trades": p["screening_median_n_trade_events"],
            "features": "|".join(p["features"]),
        }
        for p in feature_pairs
    ])
    feature_pairs_table.to_csv(OUTPUT_DIR / "feature_pairs_to_optimize.csv", index=False)
    print("\nFeature pairs selected from screening (top-1 z per algo x method):")
    print(feature_pairs_table[[
        "algorithm", "method_group", "z_window", "set_name", "screening_dro",
        "screening_h1", "screening_h2", "screening_median_trades"
    ]].to_string(index=False))

    # ---- Load OHLCV and prepare datasets per needed z_window
    print("\nLoading OHLCV...")
    loader = KlinesDataLoader(symbols=SYMBOLS)
    ohlcv = loader.load_data(download_path=OHLCV_RELATIVE_PATH, analyse_data=False, cleaning=True)
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
    ohlcv = ohlcv[ohlcv["symbol"].isin(SYMBOLS)].sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print(f"OHLCV: {ohlcv.shape}")

    needed_z = sorted({p["z_window"] for p in feature_pairs})
    datasets_by_z: dict[int, dict[str, pd.DataFrame]] = {}
    for z in needed_z:
        print(f"\nPreparing datasets for z_window={z}...")
        zdf = process_indicators_for_z_window(ohlcv, z)
        splits = split_train_val_test(zdf)
        datasets_by_z[z] = splits
        print(f"  train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # ---- Save run config snapshot
    run_config = {
        "created_at": t_start.isoformat(),
        "run_label": RUN_LABEL,
        "run_preset": RUN_PRESET,
        "smoke_test": SMOKE_TEST,
        "mode": "SMOKE" if SMOKE_TEST else "FULL",
        "algorithms_to_run": ALGORITHMS_TO_RUN,
        "symbols": SYMBOLS,
        "state_features": STATE_FEATURES,
        "n_trials_per_study": N_TRIALS,
        "ts_seeds_per_trial": TS_SEEDS_PER_TRIAL,
        "ucb_seeds_per_trial": UCB_SEEDS_PER_TRIAL,
        "n_startup_trials": N_STARTUP_TRIALS,
        "full_n_trials": FULL_N_TRIALS,
        "full_ts_seeds_per_trial": FULL_TS_SEEDS_PER_TRIAL,
        "full_ucb_seeds_per_trial": FULL_UCB_SEEDS_PER_TRIAL,
        "smoke_n_trials": SMOKE_N_TRIALS,
        "smoke_ts_seeds_per_trial": SMOKE_TS_SEEDS_PER_TRIAL,
        "smoke_ucb_seeds_per_trial": SMOKE_UCB_SEEDS_PER_TRIAL,
        "search_space": SEARCH_SPACE,
        "constraints": {
            "min_closed_trades_per_symbol": MIN_CLOSED_TRADES_PER_SYMBOL,
            "pf_inf_cap_diagnostic":        PF_INF_CAP,
            "convention": "constraint_value <= 0 is feasible; > 0 is violation",
            "tuple_order": ["min_symbol_trades", "finite_objective"],
            "note": (
                "Minimal constraint set: only median_min_symbol_trades >= 50 + finite "
                "objective. Drawdown, profit factor, aggregate trade count are computed "
                "as diagnostics in trial_results_all.csv but NOT used to filter trials. "
                "Rationale: transaction costs enter realised log-returns, so excessive "
                "turnover is expected to hurt DRO empirically; turnover is monitored "
                "diagnostically, not constrained. Drawdown is reported as a path-risk "
                "diagnostic — the DRO objective is endpoint-aware (worst-half mean "
                "return), not path-aware, so intra-half drawdowns that recover by the "
                "half's endpoint are not penalized by DRO alone. min_symbol_trades is "
                "the single retained hard constraint (statistical floor + protection "
                "against single-symbol degeneracy in 3-symbol experiment)."
            ),
        },
        "objective": {
            "name": "dro_minimax_median",
            "dro_alpha": float(DRO_ALPHA),
            "formula": (
                f"median_seed( {DRO_ALPHA} * min(mean_symbol(val_h1), mean_symbol(val_h2)) "
                f"+ {1.0 - DRO_ALPHA} * 0.5 * (mean_symbol(val_h1) + mean_symbol(val_h2)) )"
            ),
            "interpretation": (
                f"α-смесь min и mean между двумя val halves (α=DRO_ALPHA={DRO_ALPHA}). "
                f"α=1.0 → классический temporal maximin DRO (CVaR(0.5) на 2-point uniform "
                f"mixture); α=0.5 → mean-CVaR mixture (Rockafellar-Uryasev 2000): "
                f"worst-half эффективный вес 0.75, best-half 0.25; α=0.0 → простое "
                f"среднее. Должно совпадать с DRO_ALPHA в screening_bs_stable_vs_corr.py "
                f"для consistency между screening selection и HPO objective."
            ),
            "diagnostic_baseline": {
                "name": "dro_pure_min_median",
                "formula": "median_seed( min(mean_symbol(val_h1), mean_symbol(val_h2)) )",
                "purpose": "что было бы при α=1.0 (hard worst-case DRO baseline)",
            },
            "return_normalization": "val_full and val_h1 against START_CAPITAL; val_h2 against val_h1 ending balance (compound)",
        },
        "execution": {
            "start_capital":              START_CAPITAL,
            "position_size":              POSITION_SIZE,
            "min_hold_bars":              MIN_HOLD_BARS,
            "cooldown_bars":              COOLDOWN_BARS,
            "confidence_threshold":       CONFIDENCE_THRESHOLD,
            "threshold_mode":             THRESHOLD_MODE,
            "bandit_update_action_source": BANDIT_UPDATE_ACTION_SOURCE,
            "bandit_update_policy": BANDIT_UPDATE_POLICY,
            "screening_variant": SCREENING_VARIANT,
            "screening_minimax_dro_csv": str(SCREENING_MINIMAX_DRO_CSV),
            "update_on_validation":       UPDATE_ON_VALIDATION,
            "horizon":                    HORIZON,
            "trade_cost":                 TRADE_COST,
            "reward_clip":                REWARD_CLIP,
        },
        "confirmation": {
            "enabled":            RUN_TS_CONFIRMATION,
            "top_n":              TOP_N_CONFIRMATION,
            "confirmation_seeds": CONFIRMATION_SEEDS,
        },
        "feature_pairs": feature_pairs_table.to_dict("records"),
    }
    (OUTPUT_DIR / "optuna_stage2_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # ---- Containers
    all_trial_rows: list[dict] = []
    all_seed_rows: list[dict] = []
    all_symbol_frames: list[pd.DataFrame] = []
    study_best_rows: list[dict] = []
    error_rows: list[dict] = []

    trial_jsonl_path = OUTPUT_DIR / "trial_results.jsonl"
    error_jsonl_path = OUTPUT_DIR / "trial_errors.jsonl"

    # ===============================================================
    # Per-study Optuna loop
    # ===============================================================
    for pair_idx, feature_pair in enumerate(feature_pairs, start=1):
        algorithm = feature_pair["algorithm"]
        method_group = feature_pair["method_group"]
        set_name = feature_pair["set_name"]
        z_window = feature_pair["z_window"]
        features = feature_pair["features"]
        seeds_for_trial = TS_SEEDS_PER_TRIAL if algorithm in TS_BANDITS else UCB_SEEDS_PER_TRIAL

        study_name = f"{algorithm}__{method_group}__{set_name}"
        safe_dir = f"{pair_idx:02d}_{algorithm}_{method_group}"
        study_dir = STUDIES_DIR / safe_dir
        study_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 120)
        print(f"Study {pair_idx}/{len(feature_pairs)}: {study_name}")
        print(f"  z_window={z_window}, market features={len(features)}, "
              f"seeds_per_trial={seeds_for_trial}")
        print("=" * 120)

        sampler = optuna.samplers.TPESampler(
            seed=OPTUNA_SAMPLER_SEED + pair_idx,
            n_startup_trials=min(N_STARTUP_TRIALS, max(1, N_TRIALS // 4)),
            multivariate=True,
            group=True,
            constraints_func=optuna_constraints_func,
        )
        # In-memory Optuna; no persistence.
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
        )

        if ENQUEUE_DEFAULT_TRIAL:
            dp = default_trial_params(algorithm)
            print(f"  Enqueue screening default as trial #0: {dp}")
            study.enqueue_trial(dp)

        def objective(trial: optuna.Trial) -> float:
            bandit_params = suggest_bandit_params(trial, algorithm)
            seed_summaries = []
            symbol_metric_frames = []
            try:
                for seed in seeds_for_trial:
                    symbol_metrics, seed_summary = run_backtest_for_seed(
                        feature_pair=feature_pair,
                        bandit_params=bandit_params,
                        seed=int(seed),
                        datasets_by_z=datasets_by_z,
                    )
                    symbol_metrics.insert(0, "seed", int(seed))
                    symbol_metrics.insert(0, "trial_number", int(trial.number))
                    symbol_metrics.insert(0, "set_name", set_name)
                    symbol_metrics.insert(0, "method_group", method_group)
                    symbol_metrics.insert(0, "bandit_name", algorithm)
                    symbol_metric_frames.append(symbol_metrics)

                    seed_summary.update({
                        "trial_number": int(trial.number),
                        "bandit_name": algorithm,
                        "method_group": method_group,
                        "set_name": set_name,
                        "z_window": z_window,
                    })
                    seed_summaries.append(seed_summary)

                agg = aggregate_seeds(seed_summaries)
                score, details, constraints_tuple = dro_objective_from_agg(agg)

                trial_row = {
                    "trial_number": int(trial.number),
                    "study_name": study_name,
                    "bandit_name": algorithm,
                    "method_group": method_group,
                    "set_name": set_name,
                    "z_window": z_window,
                    "n_market_features": len(features),
                    "feature_list": "|".join(features),
                    "memory_horizon_bars": float(trial.user_attrs.get("memory_horizon_bars", float("nan"))),
                    "objective_score": score,
                    **details,
                    **agg,
                    **{f"param_{k}": v for k, v in bandit_params.items() if k != "actions"},
                    **{f"optuna_param_{k}": v for k, v in trial.params.items()},
                }

                all_trial_rows.append(trial_row)
                all_seed_rows.extend(seed_summaries)
                if symbol_metric_frames:
                    all_symbol_frames.append(pd.concat(symbol_metric_frames, ignore_index=True))

                append_jsonl(trial_jsonl_path, trial_row)
                pd.DataFrame(all_trial_rows).to_csv(OUTPUT_DIR / "trial_results_all.csv", index=False)
                pd.DataFrame(all_seed_rows).to_csv(OUTPUT_DIR / "trial_seed_level_summary_all.csv", index=False)
                if all_symbol_frames:
                    pd.concat(all_symbol_frames, ignore_index=True).to_csv(
                        DIAGNOSTICS_DIR / "trial_symbol_metrics_all.csv", index=False
                    )

                trial.set_user_attr("objective_score", score)
                trial.set_user_attr("constraints_values", list(constraints_tuple))
                for k, v in details.items():
                    trial.set_user_attr(k, v)
                for k, v in agg.items():
                    trial.set_user_attr(k, v)
                return score

            except Exception as e:
                err = {
                    "trial_number": int(trial.number),
                    "study_name": study_name,
                    "bandit_name": algorithm,
                    "method_group": method_group,
                    "set_name": set_name,
                    "error": repr(e),
                    "params": dict(trial.params),
                }
                error_rows.append(err)
                append_jsonl(error_jsonl_path, err)
                pd.DataFrame(error_rows).to_csv(OUTPUT_DIR / "trial_errors.csv", index=False)
                print("TRIAL ERROR:", err)
                return float(INVALID_SCORE)

        study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, gc_after_trial=True, show_progress_bar=False)

        # Save Optuna's native trial dataframe per study (diagnostic only)
        study_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        study_df.to_csv(study_dir / "optuna_trials_dataframe.csv", index=False)

        # ---- HPO selector: argmax DRO over feasible completed trials
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE
                     and t.value is not None and np.isfinite(float(t.value))]
        trials_with_scores = [
            (t, float(t.value), is_feasible(tuple(t.user_attrs.get("constraints_values", [1e9]))))
            for t in completed
        ]
        best_trial, is_fallback = pick_best_feasible_or_fallback(trials_with_scores)
        n_feasible = sum(1 for (_, _, ok) in trials_with_scores if ok)
        print(f"\n  [{study_name}] completed={len(completed)}, feasible={n_feasible}")

        if best_trial is None:
            print(f"  WARNING: no completed trials for {study_name}; skipping")
            continue

        sel_attrs = best_trial.user_attrs
        best_row = {
            "study_name": study_name,
            "bandit_name": algorithm,
            "method_group": method_group,
            "set_name": set_name,
            "z_window": z_window,
            "best_trial_number": int(best_trial.number),
            "best_objective_score": float(best_trial.value),
            "best_is_fallback_infeasible": bool(is_fallback),
            "best_is_feasible": bool(not is_fallback),
            "n_completed_trials": int(len(completed)),
            "n_feasible_trials": int(n_feasible),
            "feasible_share": float(n_feasible / len(completed)) if completed else 0.0,
            **{f"best_param_{k}": v for k, v in best_trial.params.items()},
            **{f"best_metric_{k}": v for k, v in sel_attrs.items() if k in {
                "dro_minimax_median", "dro_minimax_mean", "dro_minimax_std",
                "h1_mean_aggregate_median", "h2_mean_aggregate_median", "full_mean_aggregate_median",
                "median_max_drawdown_abs", "median_profit_factor",
                "median_total_trades", "median_min_symbol_trades", "memory_horizon_bars",
                # Holding-time + cycle diagnostics
                "median_holding_bars_overall", "mean_holding_bars_overall",
                "p25_holding_bars_overall", "p75_holding_bars_overall",
                "share_trades_closed_before_horizon_overall",
                "share_trades_closed_after_horizon_overall",
                "median_time_between_entries_overall",
                # decision-only diagnostics
                "forced_singleton_ratio_overall",
                "decision_point_ratio_overall",
                "bandit_consulted_ratio_overall",
                "pending_update_queued_ratio_overall",
            }},
        }
        study_best_rows.append(best_row)
        pd.DataFrame(study_best_rows).to_csv(OUTPUT_DIR / "study_best_summary.csv", index=False)
        with open(study_dir / "best_trial.json", "w", encoding="utf-8") as f:
            json.dump(best_row, f, ensure_ascii=False, indent=2, default=str)

        feas_label = "FEASIBLE" if not is_fallback else "INFEASIBLE (fallback)"
        print(f"  Best HPO trial: #{best_trial.number} DRO={best_trial.value:.3f} [{feas_label}]")
        print(f"    params: {best_trial.params}")
        print(f"    DD={sel_attrs.get('median_max_drawdown_abs', float('nan')):.2f}%, "
              f"PF={sel_attrs.get('median_profit_factor', float('nan')):.2f}, "
              f"trades={sel_attrs.get('median_total_trades', float('nan')):.0f}, "
              f"min_sym={sel_attrs.get('median_min_symbol_trades', float('nan')):.0f}")
        print(f"    hold_bars: median={sel_attrs.get('median_holding_bars_overall', float('nan')):.1f}, "
              f"mean={sel_attrs.get('mean_holding_bars_overall', float('nan')):.1f}, "
              f"[p25,p75]=[{sel_attrs.get('p25_holding_bars_overall', float('nan')):.1f}, "
              f"{sel_attrs.get('p75_holding_bars_overall', float('nan')):.1f}], "
              f"share_closed_before_H={sel_attrs.get('share_trades_closed_before_horizon_overall', float('nan')):.2f}, "
              f"after_H={sel_attrs.get('share_trades_closed_after_horizon_overall', float('nan')):.2f}, "
              f"median_tbi={sel_attrs.get('median_time_between_entries_overall', float('nan')):.1f}")
        print(f"    decision_only: forced_singleton={sel_attrs.get('forced_singleton_ratio_overall', float('nan')):.2f}, "
              f"decision_point={sel_attrs.get('decision_point_ratio_overall', float('nan')):.2f}, "
              f"bandit_consulted={sel_attrs.get('bandit_consulted_ratio_overall', float('nan')):.2f}, "
              f"pending_queued={sel_attrs.get('pending_update_queued_ratio_overall', float('nan')):.2f}")

        gc.collect()

    print("\n" + "=" * 120)
    print("Optuna main loop done.")
    print("=" * 120)

    # ===============================================================
    # FEASIBILITY SUMMARY (printed regardless of mode; especially useful in smoke)
    # ===============================================================
    trial_results_path = OUTPUT_DIR / "trial_results_all.csv"
    if trial_results_path.exists():
        tr = pd.read_csv(trial_results_path)
        valid_mask = tr["valid"].astype(str).str.lower().eq("true")
        tr_valid = tr[valid_mask]

        print("\n" + "=" * 120)
        print(f"FEASIBILITY SUMMARY [{mode_tag} mode]")
        print("=" * 120)
        print(f"Total completed trials: {len(tr)}")
        print(f"Feasible trials:        {len(tr_valid)}  ({100*len(tr_valid)/max(1,len(tr)):.1f}%)")
        print(f"Infeasible trials:      {len(tr) - len(tr_valid)}  "
              f"({100*(len(tr)-len(tr_valid))/max(1,len(tr)):.1f}%)")

        # Per-study breakdown
        per_study = (
            tr.assign(valid_bool=valid_mask)
              .groupby(["bandit_name", "method_group", "set_name", "z_window"], as_index=False)
              .agg(n_total=("trial_number", "count"),
                   n_feasible=("valid_bool", "sum"))
        )
        per_study["feasible_share"] = per_study["n_feasible"] / per_study["n_total"]

        # Average violation magnitude per constraint (among infeasible trials)
        viol_cols = ["cv_min_symbol_trades", "cv_finite_objective"]
        viol_cols = [c for c in viol_cols if c in tr.columns]
        infeas = tr[~valid_mask]
        viol_summary = {}
        for c in viol_cols:
            v = infeas[c].astype(float)
            n_violating = int((v > 0).sum())
            if n_violating > 0:
                viol_summary[c] = (n_violating, float(v[v > 0].mean()))
            else:
                viol_summary[c] = (0, 0.0)

        print("\nPer-study feasibility:")
        print(per_study.to_string(index=False))

        if viol_summary and len(infeas) > 0:
            print(f"\nConstraint violation breakdown (n_infeasible={len(infeas)}):")
            print(f"  {'constraint':<28} {'n_violating':>12} {'mean_excess':>14}")
            for c, (n, m) in viol_summary.items():
                share = 100.0 * n / max(1, len(infeas))
                print(f"  {c:<28} {n:>6} ({share:5.1f}%)  {m:>14.3f}")

        per_study.to_csv(DIAGNOSTICS_DIR / "feasibility_per_study.csv", index=False)
        print(f"\nFeasibility per-study CSV: {DIAGNOSTICS_DIR / 'feasibility_per_study.csv'}")

        # Smoke-specific actionable hint
        if SMOKE_TEST:
            if len(tr_valid) == 0:
                print("\n" + "!" * 100)
                print("SMOKE TEST RESULT: 0 FEASIBLE TRIALS across ALL studies.")
                print("With minimal constraint set (only MIN_CLOSED_TRADES_PER_SYMBOL),")
                print("zero feasible trials means bandit is degenerate (single-symbol or no trades).")
                print("Suggested actions:")
                print(f"  - lower MIN_CLOSED_TRADES_PER_SYMBOL (e.g. {MIN_CLOSED_TRADES_PER_SYMBOL} -> 30)")
                print("  - inspect best-trial diagnostics: maybe bandit fully ignores some symbol")
                print("  - check that screening defaults still produce >=50 closed trades per symbol")
                print("DO NOT proceed to full HPO until at least 1 feasible trial appears per study.")
                print("!" * 100)
            else:
                empty_studies = per_study[per_study["n_feasible"].eq(0)]
                if len(empty_studies) > 0:
                    print("\n" + "!" * 100)
                    print(f"SMOKE WARN: {len(empty_studies)}/{len(per_study)} studies have 0 feasible trials.")
                    print("Studies with no feasible trials:")
                    print(empty_studies.to_string(index=False))
                    print("Consider relaxing constraints for these studies before full HPO.")
                    print("!" * 100)
                else:
                    print("\n" + "=" * 100)
                    print(f"SMOKE TEST OK: every study has >=1 feasible trial.")
                    print(f"Set SMOKE_TEST=False and re-run for full HPO ({FULL_N_TRIALS} trials, "
                          f"{len(FULL_TS_SEEDS_PER_TRIAL)}/{len(FULL_UCB_SEEDS_PER_TRIAL)} TS/UCB seeds, "
                          f"confirmation enabled).")
                    print("=" * 100)

    # ===============================================================
    # DIAGNOSTIC ONLY: post-hoc multi-key ranking
    # (NOT used for primary best-trial selection; main selector above is
    #  argmax DRO among feasible trials)
    # ===============================================================
    trial_results_path = OUTPUT_DIR / "trial_results_all.csv"
    if trial_results_path.exists():
        trial_results = pd.read_csv(trial_results_path)
        diag_cols = ["valid", "objective_score", "dro_minimax_median",
                     "h1_mean_aggregate_median", "h2_mean_aggregate_median",
                     "median_max_drawdown_abs", "median_profit_factor",
                     "median_total_trades", "memory_horizon_bars"]
        diag_asc  = [False, False, False, False, False, True, False, True, True]
        diag_cols = [c for c in diag_cols if c in trial_results.columns]
        diag_asc  = diag_asc[:len(diag_cols)]
        diag_ranking = (
            trial_results
            .sort_values(["bandit_name", "method_group", *diag_cols],
                         ascending=[True, True, *diag_asc])
            .reset_index(drop=True)
        )
        diag_ranking["diagnostic_rank"] = (
            diag_ranking.groupby(["bandit_name", "method_group"]).cumcount() + 1
        )
        diag_ranking.to_csv(DIAGNOSTICS_DIR / "diagnostic_trial_ranking_all.csv", index=False)
        print("\n(Diagnostic-only) multi-key ranking saved to "
              f"{DIAGNOSTICS_DIR / 'diagnostic_trial_ranking_all.csv'}")

    # ===============================================================
    # TS Confirmation: top-N feasible per (algo, method) x 25 fresh seeds.
    # Re-evaluate same params with new seeds; recompute DRO + constraints.
    # Confirmation selector: argmax confirmation DRO over confirmation-feasible
    # trials (with explicit infeasible fallback).
    # ===============================================================
    if not RUN_TS_CONFIRMATION:
        print("\nConfirmation disabled (RUN_TS_CONFIRMATION=False); skipping.")
    elif not study_best_rows:
        print("\nNo HPO results; confirmation skipped.")
    else:
        confirmation_rows: list[dict] = []
        confirmation_seed_rows: list[dict] = []
        confirmation_symbol_frames: list[pd.DataFrame] = []

        sb = pd.DataFrame(study_best_rows)
        ts_studies = sb[sb["bandit_name"].isin(TS_BANDITS)].copy()

        if ts_studies.empty:
            print("\nNo TS studies in this preset; confirmation skipped.")
        else:
            print("\n" + "=" * 120)
            print(f"TS confirmation: top-{TOP_N_CONFIRMATION} feasible trials per study "
                  f"x {len(CONFIRMATION_SEEDS)} fresh seeds")
            print("=" * 120)
            trial_results = pd.read_csv(trial_results_path)
            # Robust bool parsing from CSV: pandas may read "False" as a non-empty
            # string, and a non-empty string is truthy. So astype(bool) would turn
            # "False" into True. Compare to "true" after lowercasing string repr.
            valid_mask = trial_results["valid"].astype(str).str.lower().eq("true")
            pair_lookup = {(p["algorithm"], p["method_group"]): p for p in feature_pairs}

            for _, study_row in ts_studies.iterrows():
                algorithm = study_row["bandit_name"]
                method_group = study_row["method_group"]
                set_name = study_row["set_name"]

                # Take top-N FEASIBLE trials by main DRO from this study
                cur = trial_results[
                    (trial_results["bandit_name"].eq(algorithm))
                    & (trial_results["method_group"].eq(method_group))
                    & valid_mask
                ].sort_values("objective_score", ascending=False).head(TOP_N_CONFIRMATION)

                if cur.empty:
                    print(f"\nStudy {algorithm}/{method_group}: no feasible trials; "
                          f"confirmation skipped for this study.")
                    continue

                feature_pair = pair_lookup[(algorithm, method_group)]
                print(f"\nStudy {algorithm}/{method_group}/{set_name}: "
                      f"confirming top-{len(cur)} feasible trials")

                for rank, (_, trow) in enumerate(cur.iterrows(), start=1):
                    original_trial = int(trow["trial_number"])
                    original_objective = float(trow["objective_score"])
                    bandit_params = rebuild_bandit_params_from_trial_row(trow)

                    seed_summaries = []
                    symbol_metric_frames = []
                    for seed in CONFIRMATION_SEEDS:
                        try:
                            symbol_metrics, seed_summary = run_backtest_for_seed(
                                feature_pair=feature_pair,
                                bandit_params=bandit_params,
                                seed=int(seed),
                                datasets_by_z=datasets_by_z,
                            )
                        except Exception as e:
                            print(f"  confirm seed={seed} ERROR: {e!r}")
                            continue
                        symbol_metrics.insert(0, "confirmation_seed", int(seed))
                        symbol_metrics.insert(0, "original_trial_number", original_trial)
                        symbol_metrics.insert(0, "confirmation_rank", rank)
                        symbol_metrics.insert(0, "set_name", set_name)
                        symbol_metrics.insert(0, "method_group", method_group)
                        symbol_metrics.insert(0, "bandit_name", algorithm)
                        symbol_metric_frames.append(symbol_metrics)

                        seed_summary.update({
                            "confirmation_seed": int(seed),
                            "original_trial_number": original_trial,
                            "confirmation_rank": rank,
                            "bandit_name": algorithm,
                            "method_group": method_group,
                            "set_name": set_name,
                            "z_window": feature_pair["z_window"],
                        })
                        seed_summaries.append(seed_summary)

                    if not seed_summaries:
                        print(f"  Top-{rank} trial #{original_trial}: all confirm seeds failed")
                        continue

                    conf_agg = aggregate_seeds(seed_summaries)
                    conf_score, conf_details, conf_constraints = dro_objective_from_agg(conf_agg)
                    conf_feasible = is_feasible(conf_constraints)
                    # Methodological note: confirmation_complete means the trial
                    # was evaluated on the FULL CONFIRMATION_SEEDS set. Partial
                    # trials are kept in the CSV for diagnostic purposes but
                    # complete-feasible trials strictly outrank them in winner
                    # selection (see pick_best_complete_feasible_or_fallback).
                    conf_complete = (len(seed_summaries) == len(CONFIRMATION_SEEDS))
                    n_seeds_missing = len(CONFIRMATION_SEEDS) - len(seed_summaries)

                    row = {
                        "bandit_name": algorithm,
                        "method_group": method_group,
                        "set_name": set_name,
                        "z_window": feature_pair["z_window"],
                        "original_trial_number": original_trial,
                        "confirmation_rank_by_main_dro": rank,
                        "original_objective_score": original_objective,
                        "confirmation_objective_score": conf_score,
                        "confirmation_feasible": bool(conf_feasible),
                        "confirmation_complete": bool(conf_complete),
                        "n_confirmation_seeds_expected": int(len(CONFIRMATION_SEEDS)),
                        "n_confirmation_seeds_used": int(len(seed_summaries)),
                        "n_confirmation_seeds_missing": int(n_seeds_missing),
                        "memory_horizon_bars": float(
                            1.0 / (1.0 - float(bandit_params["discount_factor"]))
                            if algorithm in DISCOUNTED_BANDITS
                            else int(bandit_params["window_size"])
                        ),
                        "n_confirmation_seeds": len(seed_summaries),
                        **{f"confirmation_{k}": v for k, v in conf_agg.items()},
                        **{f"param_{k}": v for k, v in bandit_params.items() if k != "actions"},
                    }
                    confirmation_rows.append(row)
                    confirmation_seed_rows.extend(seed_summaries)
                    if symbol_metric_frames:
                        confirmation_symbol_frames.append(pd.concat(symbol_metric_frames, ignore_index=True))

                    feas_tag = "FEAS" if conf_feasible else "INFEAS"
                    complete_tag = "COMPLETE" if conf_complete else f"PARTIAL({len(seed_summaries)}/{len(CONFIRMATION_SEEDS)})"
                    print(f"  Top-{rank} #{original_trial} orig_DRO={original_objective:.3f} "
                          f"conf_DRO={conf_score:.3f} [{feas_tag}/{complete_tag}] "
                          f"(DD={conf_agg['median_max_drawdown_abs']:.2f}%, "
                          f"PF={conf_agg['median_profit_factor']:.2f}, "
                          f"trades={conf_agg['median_total_trades']:.0f}, "
                          f"min_sym={conf_agg['median_min_symbol_trades']:.0f})")

                    # Incremental save
                    pd.DataFrame(confirmation_rows).to_csv(
                        OUTPUT_DIR / "ts_confirmation_results.csv", index=False)
                    pd.DataFrame(confirmation_seed_rows).to_csv(
                        OUTPUT_DIR / "ts_confirmation_seed_level.csv", index=False)
                    if confirmation_symbol_frames:
                        pd.concat(confirmation_symbol_frames, ignore_index=True).to_csv(
                            DIAGNOSTICS_DIR / "ts_confirmation_symbol_metrics.csv", index=False)

        # ---- Confirmation selector + final unified table
        # Selection priority: complete=True -> feasible=True -> argmax conf_DRO.
        # Partial trials (where some confirmation seeds failed) are kept in CSV
        # for diagnostic transparency but strictly outranked by complete-feasible
        # winners. If no complete trial exists for a (algo, method) group,
        # fallback flags are exposed explicitly.
        if confirmation_rows:
            conf_df = pd.DataFrame(confirmation_rows)
            conf_best_rows: list[dict] = []
            for (algorithm, method_group), part in conf_df.groupby(["bandit_name", "method_group"], sort=True):
                items = [
                    (r,
                     float(r["confirmation_objective_score"]),
                     bool(r["confirmation_feasible"]),
                     bool(r["confirmation_complete"]))
                    for _, r in part.iterrows()
                ]
                best, fallback_any, fallback_infeasible = pick_best_complete_feasible_or_fallback(items)
                if best is None:
                    continue
                rec = dict(best)
                rec["confirmation_is_fallback_any"] = bool(fallback_any)
                rec["confirmation_is_fallback_infeasible"] = bool(fallback_infeasible)
                conf_best_rows.append(rec)
            conf_best_df = pd.DataFrame(conf_best_rows)
            conf_best_df.to_csv(
                OUTPUT_DIR / "ts_best_trial_per_algorithm_method_after_confirmation.csv", index=False)
            print("\nTS confirmation winners (per algo, method):")
            cols_show = ["bandit_name", "method_group", "set_name", "original_trial_number",
                         "confirmation_objective_score",
                         "confirmation_complete", "confirmation_feasible",
                         "confirmation_is_fallback_any", "confirmation_is_fallback_infeasible",
                         "n_confirmation_seeds_used", "n_confirmation_seeds_missing",
                         "confirmation_dro_minimax_median",
                         "confirmation_median_max_drawdown_abs",
                         "confirmation_median_profit_factor",
                         "confirmation_median_total_trades",
                         "confirmation_median_min_symbol_trades"]
            cols_show = [c for c in cols_show if c in conf_best_df.columns]
            print(conf_best_df[cols_show].to_string(index=False))
            n_partial = int((~conf_best_df["confirmation_complete"].astype(bool)).sum()) \
                        if "confirmation_complete" in conf_best_df.columns else 0
            if n_partial > 0:
                print(f"\n  WARN: {n_partial}/{len(conf_best_df)} confirmation winners are PARTIAL "
                      f"(some confirmation seeds failed; see n_confirmation_seeds_missing).")
        else:
            conf_best_df = pd.DataFrame()

        # ---- Final unified selection: TS = confirmation winner; UCB = HPO best
        final_rows: list[dict] = []
        for _, row in pd.DataFrame(study_best_rows).iterrows():
            algorithm = row["bandit_name"]
            method_group = row["method_group"]
            if algorithm in TS_BANDITS:
                if not conf_best_df.empty:
                    match = conf_best_df[
                        (conf_best_df["bandit_name"].eq(algorithm))
                        & (conf_best_df["method_group"].eq(method_group))
                    ]
                else:
                    match = pd.DataFrame()
                if not match.empty:
                    c = match.iloc[0]
                    final_rows.append({
                        "selection_source":   "ts_confirmation",
                        "bandit_name":        algorithm,
                        "method_group":       method_group,
                        "set_name":           c["set_name"],
                        "z_window":           int(c["z_window"]),
                        "selected_trial_number": int(c["original_trial_number"]),
                        "selected_dro":       float(c["confirmation_objective_score"]),
                        "selected_feasible":  bool(c["confirmation_feasible"]),
                        "selected_complete":  bool(c.get("confirmation_complete", False)),
                        "is_fallback_any":    bool(c.get("confirmation_is_fallback_any", False)),
                        "is_fallback_infeasible": bool(c.get("confirmation_is_fallback_infeasible", False)),
                        "n_seeds_expected":   int(c.get("n_confirmation_seeds_expected", len(CONFIRMATION_SEEDS))),
                        "n_seeds_used":       int(c.get("n_confirmation_seeds_used", 0)),
                        "n_seeds_missing":    int(c.get("n_confirmation_seeds_missing", 0)),
                        "selected_dd":        float(c.get("confirmation_median_max_drawdown_abs", np.nan)),
                        "selected_pf":        float(c.get("confirmation_median_profit_factor", np.nan)),
                        "selected_trades":    float(c.get("confirmation_median_total_trades", np.nan)),
                        "selected_min_sym":   float(c.get("confirmation_median_min_symbol_trades", np.nan)),
                        "selected_memory":    float(c.get("memory_horizon_bars", np.nan)),
                        "n_selection_seeds":  int(c.get("n_confirmation_seeds", 0)),
                        "original_optuna_dro": float(c.get("original_objective_score", np.nan)),
                    })
                    continue
                # Fallback to HPO best if confirmation missing
            # UCB or fallback for TS without confirmation
            final_rows.append({
                "selection_source":   "hpo_best" if algorithm in UCB_BANDITS else "hpo_best_no_confirmation",
                "bandit_name":        algorithm,
                "method_group":       method_group,
                "set_name":           row["set_name"],
                "z_window":           int(row["z_window"]),
                "selected_trial_number": int(row["best_trial_number"]),
                "selected_dro":       float(row["best_objective_score"]),
                "selected_feasible":  bool(row.get("best_is_feasible", False)),
                "selected_complete":  True,  # HPO always runs all seeds_for_trial
                "is_fallback_any":    bool(row.get("best_is_fallback_infeasible", False)),
                "is_fallback_infeasible": bool(row.get("best_is_fallback_infeasible", False)),
                "n_seeds_expected":   int(len(UCB_SEEDS_PER_TRIAL) if algorithm in UCB_BANDITS else len(TS_SEEDS_PER_TRIAL)),
                "n_seeds_used":       int(len(UCB_SEEDS_PER_TRIAL) if algorithm in UCB_BANDITS else len(TS_SEEDS_PER_TRIAL)),
                "n_seeds_missing":    0,
                "selected_dd":        float(row.get("best_metric_median_max_drawdown_abs", np.nan)),
                "selected_pf":        float(row.get("best_metric_median_profit_factor", np.nan)),
                "selected_trades":    float(row.get("best_metric_median_total_trades", np.nan)),
                "selected_min_sym":   float(row.get("best_metric_median_min_symbol_trades", np.nan)),
                "selected_memory":    float(row.get("best_metric_memory_horizon_bars", np.nan)),
                "n_selection_seeds":  int(len(UCB_SEEDS_PER_TRIAL) if algorithm in UCB_BANDITS else len(TS_SEEDS_PER_TRIAL)),
                "original_optuna_dro": float(row["best_objective_score"]),
            })

        if final_rows:
            final_df = pd.DataFrame(final_rows)
            final_df.to_csv(
                OUTPUT_DIR / "final_selection_per_algorithm_method.csv", index=False)
            print("\nFinal unified selection (TS via confirmation, UCB via HPO best):")
            print(final_df.to_string(index=False))

    # ---- End
    t_end = datetime.now()
    duration = (t_end - t_start).total_seconds()
    print("\n" + "=" * 100)
    print(f"DONE in {duration:.0f}s ({duration/3600:.2f}h). Outputs: {OUTPUT_DIR}")
    print(f"DRO_ALPHA used: {DRO_ALPHA}")
    print("=" * 100)



if __name__ == "__main__":
    main()
