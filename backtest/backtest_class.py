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
        - raw_action=1 means long / enter-or-hold long;
        - raw_action=0 means out / stay-flat-or-exit;
        - delayed reward is assigned after `horizon` bars to the raw bandit action;
        - execution constraints can override the raw action, but reward_log stores both
          raw and executed actions for diagnostics;
        - selected state features are appended to market features in the context.

    Delayed reward semantics:
        prev_position=0, raw_action=1: entry long      reward = log_ret - side_cost
        prev_position=0, raw_action=0: stay flat       reward = -alpha_out * log_ret
        prev_position=1, raw_action=1: hold long       reward = log_ret
        prev_position=1, raw_action=0: exit to flat    reward = -alpha_out * log_ret - side_cost

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
    ):
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
        scores = score_info["scores"]
        edge = float(scores[1] - scores[0])

        weak_edge = abs(edge) < self.confidence_threshold
        if weak_edge:
            final_action = prev_action
            applied = raw_action != prev_action
            return final_action, edge, applied

        return raw_action, edge, False

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

                reward, future_log_ret, reward_details = self._delayed_reward(
                    start_price=upd["start_price"],
                    end_price=current_price,
                    raw_action=upd["raw_action"],
                    prev_position=upd["prev_position"],
                    return_details=True,
                )

                if update_bandit:
                    self.bandit[sym].update(
                        chosen_action=upd["raw_action"],
                        x=upd["context"],
                        reward=reward,
                    )

                stores["rewards"][sym][upd["raw_action"]].append(reward)

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

            # 4. Raw action from bandit.
            raw_action, score_info = self.bandit[sym].select_action(
                bandit_context,
                return_scores=True,
            )
            scores = score_info["scores"]
            means = score_info["means"]
            uncertainty = score_info["uncertainty"]

            # 5. Confidence threshold.
            action_after_threshold, edge, threshold_applied = self._apply_confidence_threshold(
                raw_action=raw_action,
                score_info=score_info,
                prev_action=prev_action[sym],
            )

            # 6. Execution constraints.
            executed_action, constraint_type = self._apply_constraints(
                action=action_after_threshold,
                state_obj=state[sym],
            )
            current_position = prev_action[sym]

            # 7. Execute transition.
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

            # 8. Queue delayed update for raw action if reward matures within this phase.
            due_index = i + self.horizon
            if due_index < len(df_symbol):
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
