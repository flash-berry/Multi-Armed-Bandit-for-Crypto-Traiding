import numpy as np
import pandas as pd

from collections import defaultdict, deque

from mab.nonstationary_linear_bandits import (
    DiscountedLinearTS,
    DiscountedLinearUCB,
    SlidingWindowLinearTS,
    SlidingWindowLinearUCB,
)
from backtest.functions.state_function import State
from backtest.functions.trade_function import count_entry_action, count_usdt_final


FULL_STATE_FEATURES = [
    "state_in_position",
    "state_log_bars_in_position",
    "state_unrealized_pnl",
    "state_drawdown",
    "state_log_time_since_last_trade",
    "state_last_action",
]

DEFAULT_STATE_FEATURES = [
    "state_in_position",
]


class Backtesting:
    """
    Stateful long/out backtesting wrapper for non-stationary linear bandits.

    Pipeline assumptions:
        - action=1 means long / enter-or-hold long;
        - action=0 means out / stay-flat-or-exit;
        - delayed reward is assigned after `horizon` bars to the bandit;
        - execution constraints (min_hold, cooldown) can override raw_action via
          post-hoc action masking / shielding (Alshiekh et al. 2018);
        - reward_log stores both raw and executed actions for diagnostics;
        - selected state features are appended to market features in the context.

    Bandit update action source (parameter `bandit_update_action_source`):
        "executed" (default, canonical safe-RL semantics):
            reward computed and bandit updated on executed_action — действие,
            реально выполненное execution layer'ом после constraints. Aligns
            learning signal с реализуемой стратегией.
        "raw" (legacy, counterfactual):
            reward computed и bandit updated on raw_action — действие, которое
            policy запросила. Counterfactual update без IPS-correction.

    Delayed reward semantics (action_for_update determined по
    bandit_update_action_source):
        prev_position=0, action=1: entry long      reward = log_ret - side_cost
        prev_position=0, action=0: stay flat       reward = -alpha_out * log_ret
        prev_position=1, action=1: hold long       reward = log_ret
        prev_position=1, action=0: exit to flat    reward = -alpha_out * log_ret - side_cost

    side_cost is represented in log-return units as -log(1 - trade_cost), which is
    approximately equal to trade_cost for small costs.
    """

    def __init__(
        self,
        meta_cols,
        feature_columns,
        config_for_bandit,
        trade_cost=0.0025,
        seed=42,
        update_on_validation=True,
        horizon=10,
        min_hold_bars=2,
        cooldown_bars=1,
        confidence_threshold=0.02,
        alpha_out=0.5,
        state_feature_columns=None,
        use_symbol_seed_offset=True,
        threshold_mode="none",
        bandit_update_action_source="executed",
        bandit_update_policy="all_bars",
    ):
        """
        threshold_mode:
            "none" (default, методологически рекомендуемый): confidence_threshold
                полностью отключён. Bandit policy самостоятельно отвечает за выбор
                raw_action. Execution layer применяет только универсальные constraints:
                transaction cost (в reward и PnL), min_hold_bars, cooldown_bars.
                Turnover контролируется через transaction cost в reward и через HPO
                constraint на total_trades.

                Этот режим устраняет confounding между bandit-policy hyperparameters
                и execution-inertia hyperparameter: HPO больше не может «спасать»
                плохую policy через калибровку execution-inertia, и интерпретация
                bandit performance становится чистой.

            "uncertainty_normalized": порог применяется к
                normalized_edge = (mean_1 - mean_0) / sqrt(uncertainty_0^2 + uncertainty_1^2),
                то есть в единицах standard deviations of posterior mean difference
                (t-stat/Sharpe-like quantity). Используется только в ablation
                экспериментах для изучения эффекта inertia на out-of-sample.

            "absolute" (legacy): порог применяется к |score_1 - score_0|
                (sampled scores в TS, mean + alpha*uncertainty в UCB).
                Сохраняется для обратной совместимости с прежними HPO результатами,
                где threshold был в search space. Использовать только для
                воспроизведения прежних результатов; не для новых experiments.

        Recommended:
            Для новых HPO/evaluation runs использовать threshold_mode="none"
            и НЕ включать confidence_threshold в Optuna search space.
            Если нужна inertia ablation — отдельный experiment с
            threshold_mode="uncertainty_normalized" и фиксированным набором
            threshold values, отдельным от main HPO.
        """
        self.seed = int(seed)
        self.meta_cols = list(meta_cols)
        self.feature_columns = list(feature_columns)
        self.config_for_bandit = dict(config_for_bandit)
        self.trade_cost = float(trade_cost)
        self.side_cost_log = float(-np.log(1.0 - self.trade_cost))
        self.update_on_validation = bool(update_on_validation)

        self.horizon = int(horizon)
        self.min_hold_bars = int(min_hold_bars)
        self.cooldown_bars = int(cooldown_bars)
        self.confidence_threshold = float(confidence_threshold)
        self.alpha_out = float(alpha_out)
        self.use_symbol_seed_offset = bool(use_symbol_seed_offset)
        if threshold_mode not in ("none", "uncertainty_normalized", "absolute"):
            raise ValueError(
                f"Unknown threshold_mode: {threshold_mode!r}. "
                "Allowed: 'none', 'uncertainty_normalized', 'absolute'."
            )
        self.threshold_mode = str(threshold_mode)

        if bandit_update_action_source not in ("executed", "raw"):
            raise ValueError(
                f"Unknown bandit_update_action_source: {bandit_update_action_source!r}. "
                "Allowed: 'executed' (default, action masking semantics — bandit обучается "
                "на действии, которое реально было выполнено после constraints; canonical "
                "safe-RL action masking, Alshiekh et al. 2018), 'raw' (legacy counterfactual)."
            )
        self.bandit_update_action_source = str(bandit_update_action_source)

        if bandit_update_policy not in ("all_bars", "decision_only"):
            raise ValueError(
                f"Unknown bandit_update_policy: {bandit_update_policy!r}. "
                "Allowed: 'all_bars' (default, legacy): bandit consulted on every bar; "
                "in forced-singleton bars (MIN_HOLD/COOLDOWN active) raw action still "
                "produced and pending update queued. 'decision_only' (recommended for "
                "minimal-state setup): in forced-singleton bars bandit is NOT consulted "
                "and no pending update is queued; executed action equals the single "
                "feasible action; bandit posterior accumulates ONLY on decision-point "
                "bars where the feasible set has >= 2 actions. Theoretical basis: "
                "constrained-bandit regret is zero on bars with |feasible_set|=1 "
                "(no alternative action), so such bars are execution-continuation, "
                "not decision points. Pending rewards from prior decision-point bars "
                "are still processed normally on every bar (no change to delayed-reward "
                "queue resolution)."
            )
        self.bandit_update_policy = str(bandit_update_policy)

        self.state_feature_columns = (
            list(state_feature_columns) if state_feature_columns is not None else list(DEFAULT_STATE_FEATURES)
        )
        unknown_state = [c for c in self.state_feature_columns if c not in FULL_STATE_FEATURES]
        if unknown_state:
            raise ValueError(f"Unknown state_feature_columns: {unknown_state}. Allowed: {FULL_STATE_FEATURES}")

        self.symbols = []
        self.bandit = {}

        self.actions_train = {}
        self.raw_actions_train = {}
        self.rewards_train = {}
        self.balance_train = {}
        self.times_train = {}
        self.close_train = {}

        self.actions_val = {}
        self.raw_actions_val = {}
        self.rewards_val = {}
        self.balance_val = {}
        self.times_val = {}
        self.close_val = {}

        self.trade_log_train = {}
        self.trade_log_val = {}

        self.decision_log_train = {}
        self.decision_log_val = {}

        self.reward_log_train = {}
        self.reward_log_val = {}

        self.bandit_diagnostics_train = {}
        self.bandit_diagnostics_val = {}

    # ------------------------------------------------------------------
    # Validation / context
    # ------------------------------------------------------------------

    def _validate_dataframe(self, df, name):
        missing_meta = [col for col in self.meta_cols if col not in df.columns]
        if missing_meta:
            raise ValueError(f"В {name} отсутствуют meta columns: {missing_meta}")

        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"В {name} отсутствуют feature columns: {missing_features}")

        values = df[self.feature_columns].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError(f"В {name} feature_columns есть NaN")
        if np.isinf(values).any():
            raise ValueError(f"В {name} feature_columns есть inf")

    @staticmethod
    def _state_context_dict(state_obj, current_price):
        arr = state_obj.context(float(current_price))
        return {name: float(value) for name, value in zip(FULL_STATE_FEATURES, arr)}

    def _make_context(self, row, state_obj):
        market_context = row[self.feature_columns].astype(float).to_numpy(dtype=np.float64)
        state_dict = self._state_context_dict(state_obj, float(row["close"]))
        state_context = np.array([state_dict[c] for c in self.state_feature_columns], dtype=np.float64)
        return np.concatenate([market_context, state_context], axis=0), state_dict

    # ------------------------------------------------------------------
    # Portfolio / execution
    # ------------------------------------------------------------------

    def _portfolio_value(self, cash, asset_qty, close_price):
        if asset_qty is None:
            return cash
        return cash + count_usdt_final(
            final_asset_quantity=asset_qty,
            close_price=close_price,
            trade_cost=self.trade_cost,
        )

    def _apply_confidence_threshold(self, raw_action, score_info, prev_action):
        """
        Apply confidence threshold to decide whether raw_action overrides prev_action.

        Three modes are supported (selected via self.threshold_mode):

        1. "none" (default, методологически рекомендуемый):
            threshold полностью отключён. raw_action всегда проходит дальше как есть.
            Bandit policy сама контролирует выбор; execution layer применяет только
            min_hold/cooldown/cost. Это убирает confounding между bandit-policy
            hyperparameters и execution-inertia hyperparameter в HPO.

        2. "uncertainty_normalized" (для ablation экспериментов):
            normalized_edge = (mean_1 - mean_0) / sqrt(unc_0^2 + unc_1^2)
            где unc_a = sqrt(x.T A_inv[a] x) — posterior std of mean estimate for arm a.
            normalized_edge — это t-stat-like / Sharpe-like quantity, инвариантная
            к масштабу posterior. weak = |normalized_edge| < confidence_threshold.

        3. "absolute" (legacy, backward-compatible с прежними HPO):
            edge = score_1 - score_0   (sampled scores в TS, mean+alpha*unc в UCB)
            weak = |edge| < confidence_threshold.

        Returns:
            (final_action, edge_used_for_decision, threshold_applied)
            где threshold_applied = True если raw_action был перезаписан prev_action.
        """
        if self.threshold_mode == "none":
            # Threshold полностью отключён. Возвращаем raw_action и edge для логирования.
            scores = score_info.get("scores", {})
            if scores:
                edge_for_decision = float(scores.get(1, 0.0)) - float(scores.get(0, 0.0))
            else:
                edge_for_decision = 0.0
            return raw_action, edge_for_decision, False

        if self.threshold_mode == "uncertainty_normalized":
            means = score_info.get("means")
            uncertainty = score_info.get("uncertainty")
            if means is None or uncertainty is None:
                # Bandit не выдаёт means/uncertainty — деградация в abs режим
                scores = score_info["scores"]
                edge_for_decision = float(scores[1] - scores[0])
            else:
                mean_diff = float(means[1] - means[0])
                combined_std = float(np.sqrt(
                    float(uncertainty[0]) ** 2 + float(uncertainty[1]) ** 2
                ))
                # combined_std может быть очень малым на ранних шагах — guard от деления на 0
                edge_for_decision = mean_diff / max(combined_std, 1e-12)
        else:
            # "absolute" legacy mode
            scores = score_info["scores"]
            edge_for_decision = float(scores[1] - scores[0])

        weak_edge = abs(edge_for_decision) < self.confidence_threshold
        if weak_edge:
            final_action = prev_action
            applied = raw_action != prev_action
            return final_action, edge_for_decision, applied

        return raw_action, edge_for_decision, False

    def _apply_constraints(self, action, state_obj):
        if state_obj.in_position == 1 and state_obj.bars_in_position < self.min_hold_bars:
            forced_action = 1
            constraint_type = "min_hold" if action != forced_action else None
            return forced_action, constraint_type

        if state_obj.in_position == 0 and state_obj.time_since_last_trade < self.cooldown_bars:
            forced_action = 0
            constraint_type = "cooldown" if action != forced_action else None
            return forced_action, constraint_type

        return action, None

    def _compute_feasible_set(self, state_obj):
        """Return (feasible_actions, singleton_constraint_type).

        Mirrors `_apply_constraints` logic but computed BEFORE action selection.
        Used by `bandit_update_policy="decision_only"` to identify
        execution-continuation bars (|feasible|=1) vs decision points (|feasible|=2).

        Returns:
            feasible_actions: tuple — (1,) or (0,) or (0, 1)
            singleton_constraint_type: "min_hold" | "cooldown" | None
                — non-None only when feasible_actions is singleton
        """
        if state_obj.in_position == 1 and state_obj.bars_in_position < self.min_hold_bars:
            return (1,), "min_hold"
        if state_obj.in_position == 0 and state_obj.time_since_last_trade < self.cooldown_bars:
            return (0,), "cooldown"
        return (0, 1), None

    def _execute_transition(
        self,
        sym,
        timestamp,
        current_price,
        current_position,
        next_position,
        cash,
        assets,
        entry_price,
        state,
        trade_log_store,
        position_size,
    ):
        if current_position == 0 and next_position == 1:
            position_value = cash[sym] * position_size
            assets[sym] = count_entry_action(
                position_value=position_value,
                close_price=current_price,
                trade_cost=self.trade_cost,
            )
            cash[sym] -= position_value
            entry_price[sym] = current_price
            state[sym].enter(current_price)

            trade_log_store[sym].append({
                "timestamp": timestamp,
                "symbol": sym,
                "event": "BUY",
                "price": current_price,
                "cash_after": cash[sym],
                "asset_qty": assets[sym],
                "entry_price": current_price,
                "trade_cost": self.trade_cost,
                "portfolio_value": self._portfolio_value(cash[sym], assets[sym], current_price),
            })

        elif current_position == 1 and next_position == 0:
            exit_value = count_usdt_final(
                final_asset_quantity=assets[sym],
                close_price=current_price,
                trade_cost=self.trade_cost,
            )
            pnl_log_before_cost = (
                np.log(current_price / entry_price[sym])
                if entry_price[sym] is not None
                else 0.0
            )

            cash[sym] += exit_value
            assets[sym] = None
            state[sym].exit()

            trade_log_store[sym].append({
                "timestamp": timestamp,
                "symbol": sym,
                "event": "SELL",
                "price": current_price,
                "entry_price": entry_price[sym],
                "pnl_log_before_cost": pnl_log_before_cost,
                "cash_after": cash[sym],
                "asset_qty": 0.0,
                "trade_cost": self.trade_cost,
                "portfolio_value": cash[sym],
            })
            entry_price[sym] = None

        return cash, assets, entry_price, state

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _delayed_reward(self, start_price, end_price, raw_action, prev_position, return_details=False):
        log_ret = float(np.log(end_price / start_price))
        switched = int(prev_position) != int(raw_action)

        if raw_action == 1:
            cost = self.side_cost_log if switched else 0.0
            reward = log_ret - cost
            regime = "entry" if prev_position == 0 else "hold_long"
            opportunity_component = 0.0
        elif raw_action == 0:
            cost = self.side_cost_log if switched else 0.0
            opportunity_component = -self.alpha_out * log_ret
            reward = opportunity_component - cost
            regime = "exit" if prev_position == 1 else "stay_flat"
        else:
            raise ValueError(f"Неизвестный raw_action: {raw_action}")

        reward = float(reward)
        details = {
            "future_log_ret": log_ret,
            "reward": reward,
            "reward_positive": bool(reward > 0.0),
            "cost_applied": float(cost),
            "side_cost_log": self.side_cost_log,
            "switched": bool(switched),
            "decision_regime": regime,
            "opportunity_component": float(opportunity_component),
            "alpha_out": self.alpha_out,
            "horizon": self.horizon,
        }
        if return_details:
            return reward, log_ret, details
        return reward, log_ret

    # ------------------------------------------------------------------
    # Stores / run loop
    # ------------------------------------------------------------------

    def _get_phase_stores(self, phase_name):
        if phase_name == "train":
            return {
                "actions": self.actions_train,
                "raw_actions": self.raw_actions_train,
                "rewards": self.rewards_train,
                "balance": self.balance_train,
                "times": self.times_train,
                "close": self.close_train,
                "trade_log": self.trade_log_train,
                "decision_log": self.decision_log_train,
                "reward_log": self.reward_log_train,
                "bandit_diagnostics": self.bandit_diagnostics_train,
            }
        return {
            "actions": self.actions_val,
            "raw_actions": self.raw_actions_val,
            "rewards": self.rewards_val,
            "balance": self.balance_val,
            "times": self.times_val,
            "close": self.close_val,
            "trade_log": self.trade_log_val,
            "decision_log": self.decision_log_val,
            "reward_log": self.reward_log_val,
            "bandit_diagnostics": self.bandit_diagnostics_val,
        }

    def _run_symbol_phase(
        self,
        df_symbol,
        sym,
        phase_name,
        start_capital,
        position_size,
        update_bandit,
    ):
        df_symbol = df_symbol.sort_values("timestamp").reset_index(drop=True)
        stores = self._get_phase_stores(phase_name)

        cash = {sym: start_capital}
        assets = {sym: None}
        entry_price = {sym: None}
        state = {sym: State()}
        prev_action = {sym: 0}
        pending_updates = deque()

        print(f"{sym}: фаза {phase_name} началась: {df_symbol['timestamp'].min()}")

        for i in range(len(df_symbol)):
            row = df_symbol.iloc[i]
            timestamp = pd.to_datetime(row["timestamp"])
            current_price = float(row["close"])

            # 1. Resolve delayed updates due at this bar.
            while pending_updates and pending_updates[0]["due_index"] <= i:
                upd = pending_updates.popleft()

                # Action и prev_position для reward и bandit.update выбираются
                # в зависимости от bandit_update_action_source:
                #   "executed" (default): canonical safe-RL action masking semantics
                #                         — bandit обучается на действии, которое было
                #                         реально выполнено execution layer'ом.
                #   "raw" (legacy):       counterfactual update без IPS-correction.
                if self.bandit_update_action_source == "executed":
                    action_for_update = upd["executed_action"]
                    # prev_position for reward — это позиция ДО executed transition;
                    # сохранена в upd["prev_position"] (current_position на момент решения).
                    prev_for_reward = upd["prev_position"]
                else:
                    action_for_update = upd["raw_action"]
                    prev_for_reward = upd["prev_position"]

                reward, future_log_ret, reward_details = self._delayed_reward(
                    start_price=upd["start_price"],
                    end_price=current_price,
                    raw_action=action_for_update,
                    prev_position=prev_for_reward,
                    return_details=True,
                )

                if update_bandit:
                    self.bandit[sym].update(
                        chosen_action=action_for_update,
                        x=upd["context"],
                        reward=reward,
                    )

                stores["rewards"][sym][action_for_update].append(reward)

                stores["reward_log"][sym].append({
                    "symbol": sym,
                    "phase": phase_name,
                    "decision_timestamp": upd["decision_timestamp"],
                    "update_timestamp": timestamp,
                    "raw_action": upd["raw_action"],
                    "executed_action_at_decision": upd["executed_action"],
                    "action_after_threshold_at_decision": upd["action_after_threshold"],
                    "prev_position_at_decision": upd["prev_position"],
                    "constraint_type_at_decision": upd["constraint_type"],
                    "threshold_applied_at_decision": upd["threshold_applied"],
                    "edge_at_decision": upd["edge"],
                    "abs_edge_at_decision": abs(upd["edge"]),
                    "score_0_at_decision": upd["score_0"],
                    "score_1_at_decision": upd["score_1"],
                    "mean_0_at_decision": upd["mean_0"],
                    "mean_1_at_decision": upd["mean_1"],
                    "uncertainty_0_at_decision": upd["uncertainty_0"],
                    "uncertainty_1_at_decision": upd["uncertainty_1"],
                    "start_price": upd["start_price"],
                    "end_price": current_price,
                    "updated": bool(update_bandit),
                    **reward_details,
                    **{f"decision_{k}": v for k, v in upd["state_dict"].items()},
                })

            # 2. State update before decision: current bar close is observable.
            state[sym].on_bar(current_price)

            # 3. Context.
            bandit_context, state_dict = self._make_context(row, state[sym])

            # 4. Compute feasible action set (BEFORE selection).
            feasible_set, singleton_constraint = self._compute_feasible_set(state[sym])
            is_decision_point = (len(feasible_set) >= 2)
            current_position = prev_action[sym]

            # 5. Action selection — branch on bandit_update_policy and feasibility.
            if (self.bandit_update_policy == "decision_only") and (not is_decision_point):
                # Execution-continuation bar: bandit is NOT consulted, no pending update.
                # The single feasible action is taken automatically.
                # Pending rewards from PRIOR decision-point bars still resolved in step 1.
                forced_action = feasible_set[0]
                raw_action = forced_action
                action_after_threshold = forced_action
                executed_action = forced_action
                edge = 0.0
                threshold_applied = False
                constraint_type = singleton_constraint
                scores = (float("nan"), float("nan"))
                means = (float("nan"), float("nan"))
                uncertainty = (float("nan"), float("nan"))
                queue_pending = False
            else:
                # Decision point (or legacy "all_bars" mode): consult bandit normally.
                raw_action, score_info = self.bandit[sym].select_action(
                    bandit_context,
                    return_scores=True,
                )
                scores = score_info["scores"]
                means = score_info["means"]
                uncertainty = score_info["uncertainty"]

                # Confidence threshold.
                action_after_threshold, edge, threshold_applied = self._apply_confidence_threshold(
                    raw_action=raw_action,
                    score_info=score_info,
                    prev_action=prev_action[sym],
                )

                # Execution constraints (post-hoc shielding). In "decision_only" mode
                # this branch is reached only when feasible_set == (0, 1) so constraints
                # never trigger here; in "all_bars" mode constraints may override action.
                executed_action, constraint_type = self._apply_constraints(
                    action=action_after_threshold,
                    state_obj=state[sym],
                )
                queue_pending = True

            # 6. Execute transition.
            cash, assets, entry_price, state = self._execute_transition(
                sym=sym,
                timestamp=timestamp,
                current_price=current_price,
                current_position=current_position,
                next_position=executed_action,
                cash=cash,
                assets=assets,
                entry_price=entry_price,
                state=state,
                trade_log_store=stores["trade_log"],
                position_size=position_size,
            )

            # 7. Queue delayed update only for decision-point bars (or in legacy mode).
            # `bandit_consulted` reflects whether bandit's select_action was called;
            # `will_queue_update` additionally requires that reward matures within phase
            # (last HORIZON bars of phase: no future bar to receive reward, no queue).
            due_index = i + self.horizon
            bandit_consulted = bool(queue_pending)
            will_queue_update = queue_pending and (due_index < len(df_symbol))
            if will_queue_update:
                pending_updates.append({
                    "due_index": due_index,
                    "decision_timestamp": timestamp,
                    "context": bandit_context.copy(),
                    "raw_action": raw_action,
                    "action_after_threshold": action_after_threshold,
                    "executed_action": executed_action,
                    "prev_position": current_position,
                    "start_price": current_price,
                    "constraint_type": constraint_type,
                    "threshold_applied": threshold_applied,
                    "edge": float(edge),
                    "score_0": float(scores[0]),
                    "score_1": float(scores[1]),
                    "mean_0": float(means[0]),
                    "mean_1": float(means[1]),
                    "uncertainty_0": float(uncertainty[0]),
                    "uncertainty_1": float(uncertainty[1]),
                    "state_dict": dict(state_dict),
                })

            prev_action[sym] = executed_action

            portfolio_value = self._portfolio_value(
                cash=cash[sym],
                asset_qty=assets[sym],
                close_price=current_price,
            )

            stores["actions"][sym].append(executed_action)
            stores["raw_actions"][sym].append(raw_action)
            stores["balance"][sym].append(portfolio_value)
            stores["times"][sym].append(timestamp)
            stores["close"][sym].append(current_price)

            stores["decision_log"][sym].append({
                "timestamp": timestamp,
                "symbol": sym,
                "phase": phase_name,
                "raw_action": raw_action,
                "action_after_threshold": action_after_threshold,
                "executed_action": executed_action,
                "prev_position": current_position,
                "score_0": float(scores[0]),
                "score_1": float(scores[1]),
                "mean_0": float(means[0]),
                "mean_1": float(means[1]),
                "uncertainty_0": float(uncertainty[0]),
                "uncertainty_1": float(uncertainty[1]),
                "edge": float(edge),
                "abs_edge": abs(float(edge)),
                "threshold_applied": bool(threshold_applied),
                "constraint_type": constraint_type,
                "constraint_applied": constraint_type is not None,
                # Decision-point diagnostics:
                "feasible_set_size": int(len(feasible_set)),
                "is_decision_point": bool(is_decision_point),
                "bandit_consulted": bool(bandit_consulted),
                # `pending_update_queued` is True only if reward will actually arrive
                # within this phase. Last HORIZON bars: bandit_consulted may be True,
                # but pending_update_queued=False (reward would mature out of phase).
                "pending_update_queued": bool(will_queue_update),
                "portfolio_value": float(portfolio_value),
                "close": current_price,
                "cash": float(cash[sym]),
                "asset_qty": float(assets[sym]) if assets[sym] is not None else 0.0,
                **state_dict,
            })

        if hasattr(self.bandit[sym], "diagnostics"):
            diagnostics = self.bandit[sym].diagnostics()
            for row in diagnostics:
                row["symbol"] = sym
                row["phase"] = phase_name
            stores["bandit_diagnostics"][sym] = diagnostics

        print(f"{sym}: фаза {phase_name} закончилась: {df_symbol['timestamp'].max()}")

    # ------------------------------------------------------------------
    # Bandit factory / public API
    # ------------------------------------------------------------------

    def _make_bandit(self, config_for_bandit: dict):
        config = dict(config_for_bandit)
        bandit_type = config.pop("bandit_type", "discounted_lints")

        if bandit_type == "discounted_lints":
            return DiscountedLinearTS(**config)
        if bandit_type == "discounted_linucb":
            return DiscountedLinearUCB(**config)
        if bandit_type == "sw_lints":
            return SlidingWindowLinearTS(**config)
        if bandit_type == "sw_linucb":
            return SlidingWindowLinearUCB(**config)

        raise ValueError(
            f"Неизвестный bandit_type={bandit_type}. "
            "Ожидалось: discounted_lints, discounted_linucb, sw_lints, sw_linucb."
        )

    def backtest(
        self,
        dataframe_train,
        dataframe_val,
        symbols,
        start_capital=100,
        position_size=0.1,
    ):
        self._validate_dataframe(dataframe_train, "dataframe_train")
        self._validate_dataframe(dataframe_val, "dataframe_val")

        self.symbols = list(symbols)
        expected_n_features = len(self.feature_columns) + len(self.state_feature_columns)

        if self.config_for_bandit["n_features"] != expected_n_features:
            raise ValueError(
                f"config_for_bandit['n_features']={self.config_for_bandit['n_features']}, "
                f"но ожидается {expected_n_features} "
                f"({len(self.feature_columns)} market + {len(self.state_feature_columns)} state)"
            )

        for sym_idx, sym in enumerate(symbols):
            if sym not in dataframe_train["symbol"].unique():
                raise ValueError(f"В train_df отсутствует актив: {sym}")
            if sym not in dataframe_val["symbol"].unique():
                raise ValueError(f"В val_df отсутствует актив: {sym}")

            bandit_config = dict(self.config_for_bandit)
            if self.use_symbol_seed_offset:
                bandit_config["seed"] = int(bandit_config.get("seed", self.seed)) + sym_idx
            self.bandit[sym] = self._make_bandit(bandit_config)

            for store in [
                self.actions_train,
                self.raw_actions_train,
                self.balance_train,
                self.times_train,
                self.close_train,
                self.trade_log_train,
                self.decision_log_train,
                self.reward_log_train,
            ]:
                store[sym] = []
            self.rewards_train[sym] = defaultdict(list)
            self.bandit_diagnostics_train[sym] = []

            for store in [
                self.actions_val,
                self.raw_actions_val,
                self.balance_val,
                self.times_val,
                self.close_val,
                self.trade_log_val,
                self.decision_log_val,
                self.reward_log_val,
            ]:
                store[sym] = []
            self.rewards_val[sym] = defaultdict(list)
            self.bandit_diagnostics_val[sym] = []

        for sym in symbols:
            train_symbol = dataframe_train[dataframe_train["symbol"] == sym].copy()
            val_symbol = dataframe_val[dataframe_val["symbol"] == sym].copy()

            self._run_symbol_phase(
                df_symbol=train_symbol,
                sym=sym,
                phase_name="train",
                start_capital=start_capital,
                position_size=position_size,
                update_bandit=True,
            )

            self._run_symbol_phase(
                df_symbol=val_symbol,
                sym=sym,
                phase_name="val",
                start_capital=start_capital,
                position_size=position_size,
                update_bandit=self.update_on_validation,
            )

    def get_bandit_diagnostics_frame(self) -> pd.DataFrame:
        rows = []
        for phase_store in [self.bandit_diagnostics_train, self.bandit_diagnostics_val]:
            for sym_rows in phase_store.values():
                rows.extend(sym_rows)
        return pd.DataFrame(rows)

    def get_decision_log_frame(self, phase="val") -> pd.DataFrame:
        store = self.decision_log_val if phase == "val" else self.decision_log_train
        frames = [pd.DataFrame(rows) for rows in store.values() if rows]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_reward_log_frame(self, phase="val") -> pd.DataFrame:
        store = self.reward_log_val if phase == "val" else self.reward_log_train
        frames = [pd.DataFrame(rows) for rows in store.values() if rows]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
