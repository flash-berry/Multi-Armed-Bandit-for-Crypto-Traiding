"""
HIE-only binary feature selection for contextual bandit trading pipelines.

This module intentionally removes HDD from the final selector. For the current
long/out crypto setup the final feature-selection target is binary success:

    reward = 1[payoff_continuous > 0]

where payoff_continuous is aligned with the delayed reward used in backtesting.
The selector estimates whether a feature creates bins in which the locally best
action has a higher success probability than expected under a bootstrap null.

Expected input columns:
    - numeric market feature columns;
    - raw_action in {0, 1};
    - reward: binary {0, 1}, or payoff that can be binarized with threshold 0;
    - optional timestamp/symbol/regime columns for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class FeatureSelectionConfig:
    n_bins: int = 15
    n_bootstrap: int = 100
    min_bin_size: int = 40
    min_action_count_per_bin: int = 20
    reward_positive_threshold: float = 0.0
    random_state: int = 42


class CausalBanditFeatureSelector:
    """HIE-only selector for binary success rewards.

    HIE idea:
        For each candidate feature x, split observations into quantile bins.
        In each bin b, estimate P(Y=1 | action=a, x in b) for each action.
        HIE observed score is the weighted average local-best success rate.
        We center it by a bootstrap null where bin labels are shuffled while
        action/reward pairs are preserved.

    Output columns are named with `hie_*` only; no HDD fields are produced.
    """

    def __init__(self, config: FeatureSelectionConfig | None = None):
        self.config = config or FeatureSelectionConfig()
        self.rng = np.random.default_rng(self.config.random_state)
        self.results_: pd.DataFrame | None = None
        self.bin_details_: dict[str, pd.DataFrame] = {}

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        action_col: str = "raw_action",
        reward_col: str = "reward",
    ) -> pd.DataFrame:
        self._validate_input(df, feature_cols, action_col, reward_col)

        work = df[[*feature_cols, action_col, reward_col]].copy()
        work = work.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        work[reward_col] = (work[reward_col].astype(float) > self.config.reward_positive_threshold).astype(int)

        actions = sorted(work[action_col].unique().tolist())
        if len(actions) < 2:
            raise ValueError("Нужно минимум 2 действия/arms для HIE.")

        rows = []
        self.bin_details_.clear()
        for feature in feature_cols:
            scored = self._score_feature(work, feature, action_col, reward_col, actions)
            rows.append(scored["summary"])
            self.bin_details_[feature] = scored["bin_details"]

        results = pd.DataFrame(rows)
        results["hie_rank"] = results["hie_norm"].rank(ascending=False, method="min")
        results = results.sort_values(
            ["hie_rank", "hie_p_value", "hie_norm", "hie_observed"],
            ascending=[True, True, False, False],
        ).reset_index(drop=True)
        self.results_ = results
        return results

    def select_features(
        self,
        top_k: int | None = None,
        p_value_threshold: float | None = None,
        require_positive: bool = True,
    ) -> list[str]:
        if self.results_ is None:
            raise RuntimeError("Сначала вызови fit(...).")
        r = self.results_.copy()
        if require_positive:
            r = r[r["hie_norm"] > 0]
        if p_value_threshold is not None:
            r = r[r["hie_p_value"] <= p_value_threshold]
        r = r.sort_values(["hie_rank", "hie_p_value", "hie_norm"], ascending=[True, True, False])
        if top_k is not None:
            r = r.head(top_k)
        return r["feature"].tolist()

    def plot_scores(self, top_n: int = 30, figsize: tuple[int, int] = (12, 7)):
        if self.results_ is None:
            raise RuntimeError("Сначала вызови fit(...).")
        plot_df = self.results_.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(plot_df))
        ax.barh(y, plot_df["hie_norm"], height=0.6, label="HIE normalized")
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["feature"])
        ax.set_xlabel("HIE score centered by bootstrap null")
        ax.set_title("HIE binary feature scores")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_feature_bins(self, feature: str, figsize: tuple[int, int] = (12, 6)):
        if feature not in self.bin_details_:
            raise ValueError(f"Нет bin details для feature={feature!r}.")
        d = self.bin_details_[feature].copy()
        fig, ax = plt.subplots(figsize=figsize)
        for action, part in d.groupby("action"):
            ax.plot(part["bin_id"], part["success_rate"], marker="o", label=f"action={action}")
        ax.set_xlabel("Feature quantile bin")
        ax.set_ylabel("P(success)")
        ax.set_title(f"Local action success rates by bins: {feature}")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def _score_feature(self, df, feature, action_col, reward_col, actions):
        cfg = self.config
        binned = self._assign_bins(df[feature], cfg.n_bins)
        tmp = df[[feature, action_col, reward_col]].copy()
        tmp["_bin"] = binned
        tmp = tmp.dropna(subset=["_bin"])
        tmp["_bin"] = tmp["_bin"].astype(int)

        bin_sizes = tmp["_bin"].value_counts().sort_index()
        valid_bins = bin_sizes[bin_sizes >= cfg.min_bin_size].index.tolist()
        tmp = tmp[tmp["_bin"].isin(valid_bins)].copy()

        if tmp.empty or tmp["_bin"].nunique() < 2:
            return {
                "summary": self._empty_summary(feature, reason="too_few_bins"),
                "bin_details": pd.DataFrame(),
            }

        global_values = self._arm_values(tmp, action_col, reward_col, actions)
        global_best_action = max(global_values, key=global_values.get)
        global_best_value = float(global_values[global_best_action])

        psi_obs, bin_details = self._hie_observed(tmp, action_col, reward_col, actions)
        hie_raw = psi_obs - global_best_value

        bins = tmp["_bin"].to_numpy()
        bin_ids, counts = np.unique(bins, return_counts=True)
        ar = tmp[[action_col, reward_col]].to_numpy()
        psi_null = np.empty(cfg.n_bootstrap, dtype=float)

        for s in range(cfg.n_bootstrap):
            shuffled_bin_labels = np.repeat(bin_ids, counts)
            self.rng.shuffle(shuffled_bin_labels)
            boot = pd.DataFrame(ar, columns=[action_col, reward_col])
            boot["_bin"] = shuffled_bin_labels
            psi_null[s], _ = self._hie_observed(boot, action_col, reward_col, actions)

        hie_null_mean = float(np.mean(psi_null))
        hie_norm = float(psi_obs - hie_null_mean)
        hie_p = float(np.mean(psi_obs <= psi_null))

        action_counts = tmp[action_col].value_counts().to_dict()
        summary = {
            "feature": feature,
            "n_samples": int(len(tmp)),
            "n_bins_used": int(tmp["_bin"].nunique()),
            "hie_raw": float(hie_raw),
            "hie_observed": float(psi_obs),
            "hie_null_mean": hie_null_mean,
            "hie_null_std": float(np.std(psi_null, ddof=1)) if len(psi_null) > 1 else 0.0,
            "hie_norm": hie_norm,
            "hie_p_value": hie_p,
            "global_best_action": global_best_action,
            "global_best_value": global_best_value,
            "global_action_0_success": float(global_values.get(0, np.nan)),
            "global_action_1_success": float(global_values.get(1, np.nan)),
            "action_0_count": int(action_counts.get(0, 0)),
            "action_1_count": int(action_counts.get(1, 0)),
            "min_bin_size_used": int(tmp.groupby("_bin").size().min()),
            "min_action_count_in_bin": int(self._min_action_count_in_bins(tmp, action_col, actions)),
            "reason": "ok",
        }
        return {"summary": summary, "bin_details": bin_details}

    def _hie_observed(self, df, action_col, reward_col, actions):
        n = len(df)
        rows = []
        score = 0.0
        for bin_id, g in df.groupby("_bin", sort=True):
            values = self._arm_values(g, action_col, reward_col, actions)
            local_best_action = max(values, key=values.get)
            local_best_value = values[local_best_action]
            weight = len(g) / n
            score += weight * local_best_value
            for action in actions:
                part = g[g[action_col] == action]
                rows.append({
                    "bin_id": int(bin_id),
                    "action": action,
                    "success_rate": float(values[action]),
                    "local_best_action": local_best_action,
                    "bin_size": int(len(g)),
                    "action_count_in_bin": int(len(part)),
                    "bin_weight": float(weight),
                })
        return float(score), pd.DataFrame(rows)

    def _arm_values(self, df, action_col, reward_col, actions):
        cfg = self.config
        out = {}
        global_fallback = float(df[reward_col].mean())
        for action in actions:
            r = df.loc[df[action_col] == action, reward_col]
            if len(r) < cfg.min_action_count_per_bin:
                out[action] = global_fallback
            else:
                out[action] = float(r.mean())
        return out

    @staticmethod
    def _min_action_count_in_bins(df, action_col, actions):
        min_count = np.inf
        for _, g in df.groupby("_bin"):
            counts = g[action_col].value_counts().to_dict()
            for action in actions:
                min_count = min(min_count, counts.get(action, 0))
        return 0 if np.isinf(min_count) else int(min_count)

    @staticmethod
    def _assign_bins(x: pd.Series, n_bins: int) -> pd.Series:
        x = x.astype(float)
        if x.nunique(dropna=True) <= 1:
            return pd.Series(np.nan, index=x.index)
        try:
            return pd.qcut(x, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=x.index)

    @staticmethod
    def _validate_input(df, feature_cols, action_col, reward_col):
        needed = [*feature_cols, action_col, reward_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"В dataframe отсутствуют колонки: {missing}")
        if len(feature_cols) == 0:
            raise ValueError("feature_cols не должен быть пустым.")

    @staticmethod
    def _empty_summary(feature: str, reason: str) -> dict:
        return {
            "feature": feature,
            "n_samples": 0,
            "n_bins_used": 0,
            "hie_raw": np.nan,
            "hie_observed": np.nan,
            "hie_null_mean": np.nan,
            "hie_null_std": np.nan,
            "hie_norm": -np.inf,
            "hie_p_value": 1.0,
            "global_best_action": None,
            "global_best_value": np.nan,
            "global_action_0_success": np.nan,
            "global_action_1_success": np.nan,
            "action_0_count": 0,
            "action_1_count": 0,
            "min_bin_size_used": 0,
            "min_action_count_in_bin": 0,
            "reason": reason,
        }


# ----------------------------------------------------------------------
# Counterfactual log builder aligned with backtesting reward semantics
# ----------------------------------------------------------------------


def side_cost_log(trade_cost: float) -> float:
    """One-side transaction cost in log-return units."""
    trade_cost = float(trade_cost)
    if not (0.0 <= trade_cost < 1.0):
        raise ValueError("trade_cost must be in [0, 1).")
    return float(-np.log(1.0 - trade_cost))


def make_randomized_counterfactual_log_binary_success(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    trade_cost: float,
    alpha_out: float,
    prev_position: int,
    action_col_name: str = "raw_action",
    random_state: int = 42,
    balanced_by_symbol: bool = True,
) -> pd.DataFrame:
    """
    Builds a randomized exploration-style log with one action per timestamp.

    prev_position=0:
        action 1 = enter long, payoff = future_log_ret - side_cost
        action 0 = stay flat,  payoff = -alpha_out * future_log_ret

    prev_position=1:
        action 1 = hold long,  payoff = future_log_ret
        action 0 = exit flat,  payoff = -alpha_out * future_log_ret - side_cost

    reward = 1[payoff_continuous > 0].

    The function uses one randomized action per row, not two deterministic rows.
    This mimics a uniform/balanced exploration log and keeps action/reward pairs
    suitable for HIE bootstrap diagnostics.
    """
    if prev_position not in (0, 1):
        raise ValueError("prev_position must be 0 or 1")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    rng = np.random.default_rng(random_state)
    parts = []
    c_log = side_cost_log(trade_cost)

    for sym, g in df.groupby("symbol", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True).copy()
        g["future_close"] = g["close"].shift(-horizon)
        g["future_log_ret"] = np.log(g["future_close"] / g["close"])
        g = g.replace([np.inf, -np.inf], np.nan)
        g = g.dropna(subset=["future_log_ret", *feature_cols]).copy()
        if g.empty:
            continue

        n = len(g)
        if balanced_by_symbol:
            actions = np.resize(np.array([0, 1], dtype=int), n)
            rng.shuffle(actions)
        else:
            actions = rng.integers(0, 2, size=n)
        g[action_col_name] = actions

        r = g["future_log_ret"].astype(float)
        a = g[action_col_name].astype(int)

        if prev_position == 0:
            payoff = np.where(a == 1, r - c_log, -float(alpha_out) * r)
            regime = "entry"
        else:
            payoff = np.where(a == 1, r, -float(alpha_out) * r - c_log)
            regime = "exit"

        g["payoff_continuous"] = payoff.astype(float)
        g["reward"] = (g["payoff_continuous"] > 0.0).astype(int)
        g["prev_position"] = int(prev_position)
        g["regime"] = regime
        g["alpha_out"] = float(alpha_out)
        g["horizon"] = int(horizon)
        g["trade_cost"] = float(trade_cost)
        g["side_cost_log"] = float(c_log)

        cols = [
            "timestamp", "symbol", *feature_cols,
            "future_log_ret", "payoff_continuous", "reward", action_col_name,
            "prev_position", "regime", "alpha_out", "horizon", "trade_cost", "side_cost_log",
        ]
        parts.append(g[cols])

    if not parts:
        raise ValueError("Counterfactual log is empty after future return calculation.")

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def make_direct_counterfactual_log_binary_success(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    trade_cost: float,
    alpha_out: float,
    prev_position: int,
    action_col_name: str = "raw_action",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Direct full-feedback counterfactual log.

    Для каждого timestamp/symbol создаёт обе potential-outcome строки:
        action=1 и action=0.

    reward = 1[payoff_continuous > 0]

    prev_position=0:
        action 1 = enter long, payoff = future_log_ret - side_cost
        action 0 = stay flat,  payoff = -alpha_out * future_log_ret

    prev_position=1:
        action 1 = hold long,  payoff = future_log_ret
        action 0 = exit flat,  payoff = -alpha_out * future_log_ret - side_cost
    """

    if prev_position not in (0, 1):
        raise ValueError("prev_position must be 0 or 1")

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    c_log = side_cost_log(trade_cost)
    parts = []

    for sym, g in df.groupby("symbol", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True).copy()

        g["future_close"] = g["close"].shift(-horizon)
        g["future_log_ret"] = np.log(g["future_close"] / g["close"])

        g = g.replace([np.inf, -np.inf], np.nan)
        g = g.dropna(subset=["future_log_ret", *feature_cols]).copy()

        if g.empty:
            continue

        base = g[["timestamp", "symbol", *feature_cols, "future_log_ret"]].copy()

        # action = 1
        long_rows = base.copy()
        long_rows[action_col_name] = 1

        if prev_position == 0:
            long_rows["payoff_continuous"] = long_rows["future_log_ret"] - c_log
            regime = "entry"
        else:
            long_rows["payoff_continuous"] = long_rows["future_log_ret"]
            regime = "exit"

        # action = 0
        out_rows = base.copy()
        out_rows[action_col_name] = 0

        if prev_position == 0:
            out_rows["payoff_continuous"] = -float(alpha_out) * out_rows["future_log_ret"]
        else:
            out_rows["payoff_continuous"] = -float(alpha_out) * out_rows["future_log_ret"] - c_log

        out = pd.concat([long_rows, out_rows], ignore_index=True)

        out["reward"] = (out["payoff_continuous"] > 0.0).astype(int)
        out["prev_position"] = int(prev_position)
        out["regime"] = regime
        out["alpha_out"] = float(alpha_out)
        out["horizon"] = int(horizon)
        out["trade_cost"] = float(trade_cost)
        out["side_cost_log"] = float(c_log)

        parts.append(out)

    if not parts:
        raise ValueError("Counterfactual log is empty after future return calculation.")

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
