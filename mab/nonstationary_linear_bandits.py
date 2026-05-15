"""
Non-stationary linear contextual bandits for long/out crypto trading.

Implemented algorithms:
    - DiscountedLinearTS
    - DiscountedLinearUCB
    - SlidingWindowLinearTS
    - SlidingWindowLinearUCB

The classes intentionally share the same API used by the Backtesting code:
    bandit = Algo(n_features=d, actions=[0, 1], ...)
    action = bandit.select_action(x)
    action, info = bandit.select_action(x, return_scores=True)
    bandit.update(chosen_action, x, reward)
    diagnostics = bandit.diagnostics()

Design choices:
    - Per-action independent linear reward models.
    - Ridge prior is kept non-discounted for discounted algorithms:
        A <- gamma * (A - lambda_prior * I) + lambda_prior * I
      so old observations are forgotten while numerical regularization remains stable.
    - No adaptive TS covariance scaling. Exploration is controlled by noise_std.
    - Sliding-window algorithms use a global time window of the latest update events,
      not per-action windows, which is usually safer for non-stationary markets.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class _Observation:
    action: Any
    x: np.ndarray
    reward: float


class _BaseLinearBandit:
    """Common functionality for per-action linear contextual bandits."""

    def __init__(
        self,
        n_features: int,
        actions: Iterable[Any],
        lambda_prior: float = 1.0,
        reward_clip: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        if int(n_features) <= 0:
            raise ValueError("n_features must be positive")
        if float(lambda_prior) <= 0:
            raise ValueError("lambda_prior must be positive")
        if reward_clip is not None and float(reward_clip) <= 0:
            raise ValueError("reward_clip must be positive or None")

        self.d = int(n_features)
        self.actions = list(actions)
        if len(self.actions) == 0:
            raise ValueError("actions must be non-empty")

        self.lambda_prior = float(lambda_prior)
        self.reward_clip = reward_clip
        self.rng = np.random.default_rng(seed)
        self.I = np.eye(self.d, dtype=np.float64)

        self.A: Dict[Any, np.ndarray] = {}
        self.b: Dict[Any, np.ndarray] = {}
        self.A_inv: Dict[Any, np.ndarray] = {}

        self.action_counts = {a: 0 for a in self.actions}
        self.update_counts = {a: 0 for a in self.actions}
        self.reward_sums = {a: 0.0 for a in self.actions}
        self.total_updates = 0

        self._reset_matrices()

    def _reset_matrices(self) -> None:
        for a in self.actions:
            self.A[a] = self.lambda_prior * self.I.copy()
            self.b[a] = np.zeros(self.d, dtype=np.float64)
            self.A_inv[a] = (1.0 / self.lambda_prior) * self.I.copy()

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.d:
            raise ValueError(f"x has size {x.shape[0]}, expected {self.d}")
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("x contains NaN or inf")
        return x

    def _validate_action(self, action: Any) -> None:
        if action not in self.actions:
            raise ValueError(f"Unknown action: {action}")

    def _process_reward(self, reward: float) -> float:
        reward = float(reward)
        if np.isnan(reward) or np.isinf(reward):
            raise ValueError("reward contains NaN or inf")
        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        return reward

    def _theta_hat(self, action: Any) -> np.ndarray:
        return self.A_inv[action] @ self.b[action]

    def _uncertainty(self, action: Any, x: np.ndarray) -> float:
        return float(np.sqrt(max(x @ self.A_inv[action] @ x, _EPS)))

    def _refresh_inverse(self, action: Any) -> None:
        # Small jitter is a defensive numerical guard; with lambda_prior > 0 A should be SPD.
        self.A_inv[action] = np.linalg.inv(self.A[action] + 1e-12 * self.I)

    def _diagnostics_rows(self, algorithm: str) -> List[dict]:
        rows = []
        for a in self.actions:
            theta = self._theta_hat(a)
            rows.append(
                {
                    "algorithm": algorithm,
                    "action": a,
                    "selected_count": int(self.action_counts[a]),
                    "update_count": int(self.update_counts[a]),
                    "reward_sum": float(self.reward_sums[a]),
                    "reward_mean": (
                        float(self.reward_sums[a] / self.update_counts[a])
                        if self.update_counts[a] > 0
                        else np.nan
                    ),
                    "A_trace": float(np.trace(self.A[a])),
                    "A_inv_trace": float(np.trace(self.A_inv[a])),
                    "A_condition_number": float(np.linalg.cond(self.A[a])),
                    "b_norm": float(np.linalg.norm(self.b[a])),
                    "theta_norm": float(np.linalg.norm(theta)),
                    "total_updates": int(self.total_updates),
                }
            )
        return rows


class DiscountedLinearTS(_BaseLinearBandit):
    """Discounted Linear Thompson Sampling.

    Non-stationarity is handled through exponential discounting of accumulated data.
    The ridge prior is not discounted.
    """

    def __init__(
        self,
        n_features: int,
        actions: Iterable[Any],
        discount_factor: float = 0.999,
        lambda_prior: float = 1.0,
        noise_std: float = 0.01,
        reward_clip: Optional[float] = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__(n_features, actions, lambda_prior, reward_clip, seed)
        if not (0.0 < float(discount_factor) <= 1.0):
            raise ValueError("discount_factor must be in (0, 1]")
        if float(noise_std) <= 0:
            raise ValueError("noise_std must be positive")
        self.gamma = float(discount_factor)
        self.noise_std = float(noise_std)

    def _decay_all_actions(self) -> None:
        for a in self.actions:
            self.A[a] = self.gamma * (self.A[a] - self.lambda_prior * self.I) + self.lambda_prior * self.I
            self.b[a] = self.gamma * self.b[a]
            self._refresh_inverse(a)

    def select_action(self, x: np.ndarray, return_scores: bool = False):
        x = self._validate_x(x)
        scores, means, uncertainty = {}, {}, {}

        for a in self.actions:
            mu = self._theta_hat(a)
            cov = (self.noise_std ** 2) * self.A_inv[a]
            cov = 0.5 * (cov + cov.T)  # numerical symmetry for multivariate_normal
            theta_sample = self.rng.multivariate_normal(mean=mu, cov=cov)
            scores[a] = float(x @ theta_sample)
            means[a] = float(x @ mu)
            uncertainty[a] = self._uncertainty(a, x)

        action = max(scores, key=scores.get)
        self.action_counts[action] += 1

        if return_scores:
            return action, {"scores": scores, "means": means, "uncertainty": uncertainty}
        return action

    def update(self, chosen_action: Any, x: np.ndarray, reward: float) -> None:
        self._validate_action(chosen_action)
        x = self._validate_x(x)
        reward = self._process_reward(reward)

        self._decay_all_actions()
        self.A[chosen_action] += np.outer(x, x)
        self.b[chosen_action] += reward * x
        self._refresh_inverse(chosen_action)

        self.update_counts[chosen_action] += 1
        self.reward_sums[chosen_action] += reward
        self.total_updates += 1

    def diagnostics(self) -> List[dict]:
        rows = self._diagnostics_rows("discounted_lints")
        for row in rows:
            row.update({"discount_factor": self.gamma, "noise_std": self.noise_std})
        return rows


class DiscountedLinearUCB(_BaseLinearBandit):
    """Discounted Linear UCB.

    Selects the action maximizing mean reward plus an uncertainty bonus.
    """

    def __init__(
        self,
        n_features: int,
        actions: Iterable[Any],
        discount_factor: float = 0.999,
        lambda_prior: float = 1.0,
        ucb_alpha: float = 0.25,
        reward_clip: Optional[float] = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__(n_features, actions, lambda_prior, reward_clip, seed)
        if not (0.0 < float(discount_factor) <= 1.0):
            raise ValueError("discount_factor must be in (0, 1]")
        if float(ucb_alpha) < 0:
            raise ValueError("ucb_alpha must be non-negative")
        self.gamma = float(discount_factor)
        self.ucb_alpha = float(ucb_alpha)

    def _decay_all_actions(self) -> None:
        for a in self.actions:
            self.A[a] = self.gamma * (self.A[a] - self.lambda_prior * self.I) + self.lambda_prior * self.I
            self.b[a] = self.gamma * self.b[a]
            self._refresh_inverse(a)

    def select_action(self, x: np.ndarray, return_scores: bool = False):
        x = self._validate_x(x)
        scores, means, uncertainty = {}, {}, {}

        for a in self.actions:
            theta = self._theta_hat(a)
            mean = float(x @ theta)
            unc = self._uncertainty(a, x)
            scores[a] = mean + self.ucb_alpha * unc
            means[a] = mean
            uncertainty[a] = unc

        action = max(scores, key=scores.get)
        self.action_counts[action] += 1

        if return_scores:
            return action, {"scores": scores, "means": means, "uncertainty": uncertainty}
        return action

    def update(self, chosen_action: Any, x: np.ndarray, reward: float) -> None:
        self._validate_action(chosen_action)
        x = self._validate_x(x)
        reward = self._process_reward(reward)

        self._decay_all_actions()
        self.A[chosen_action] += np.outer(x, x)
        self.b[chosen_action] += reward * x
        self._refresh_inverse(chosen_action)

        self.update_counts[chosen_action] += 1
        self.reward_sums[chosen_action] += reward
        self.total_updates += 1

    def diagnostics(self) -> List[dict]:
        rows = self._diagnostics_rows("discounted_linucb")
        for row in rows:
            row.update({"discount_factor": self.gamma, "ucb_alpha": self.ucb_alpha})
        return rows


class _SlidingWindowBase(_BaseLinearBandit):
    """Base class for sliding-window linear bandits.

    Implementation note:
        Older versions rebuilt A and b from the whole buffer on every update.
        That is correct but expensive: O(W * d^2) per update. Here we use an
        incremental sliding-window update:
            - subtract the observation that leaves the window;
            - append the new observation;
            - add the new observation contribution.

        To control floating-point drift from repeated add/subtract operations,
        a full rebuild from the buffer is performed every ``rebuild_interval``
        updates. By default this interval is equal to ``window_size``.
    """

    def __init__(
        self,
        n_features: int,
        actions: Iterable[Any],
        window_size: int = 500,
        lambda_prior: float = 1.0,
        reward_clip: Optional[float] = 0.1,
        seed: int = 42,
        rebuild_interval: Optional[int] = None,
    ) -> None:
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive")
        if rebuild_interval is not None and int(rebuild_interval) <= 0:
            raise ValueError("rebuild_interval must be positive or None")

        super().__init__(n_features, actions, lambda_prior, reward_clip, seed)
        self.window_size = int(window_size)
        self.rebuild_interval = int(rebuild_interval) if rebuild_interval is not None else self.window_size
        self.buffer: Deque[_Observation] = deque(maxlen=self.window_size)

    def _rebuild_from_buffer(self) -> None:
        """Recompute A, b and A_inv exactly from the current window buffer."""
        self._reset_matrices()
        for obs in self.buffer:
            self.A[obs.action] += np.outer(obs.x, obs.x)
            self.b[obs.action] += obs.reward * obs.x
        for a in self.actions:
            self._refresh_inverse(a)

    def update(self, chosen_action: Any, x: np.ndarray, reward: float) -> None:
        self._validate_action(chosen_action)
        x = self._validate_x(x).copy()
        reward = self._process_reward(reward)

        # If the deque is already full, this observation will be evicted by append().
        # Remove its contribution before the append to keep A and b consistent with
        # the post-append buffer.
        old_obs: Optional[_Observation] = None
        if len(self.buffer) == self.window_size:
            old_obs = self.buffer[0]
            self.A[old_obs.action] -= np.outer(old_obs.x, old_obs.x)
            self.b[old_obs.action] -= old_obs.reward * old_obs.x

        # Store a copy so future mutations outside the class cannot corrupt the buffer.
        self.buffer.append(_Observation(chosen_action, x, reward))

        # Add the new contribution.
        self.A[chosen_action] += np.outer(x, x)
        self.b[chosen_action] += reward * x

        self.update_counts[chosen_action] += 1
        self.reward_sums[chosen_action] += reward
        self.total_updates += 1

        # Refresh inverses only for affected actions instead of every action.
        affected_actions = {chosen_action}
        if old_obs is not None:
            affected_actions.add(old_obs.action)

        for a in affected_actions:
            self._refresh_inverse(a)

        # Periodic exact rebuild limits floating-point drift from repeated subtraction.
        if self.rebuild_interval and self.total_updates % self.rebuild_interval == 0:
            self._rebuild_from_buffer()

    def _window_extra_diagnostics(self, rows: List[dict]) -> List[dict]:
        for row in rows:
            row.update(
                {
                    "window_size": self.window_size,
                    "buffer_size": len(self.buffer),
                    "rebuild_interval": self.rebuild_interval,
                }
            )
        return rows


class SlidingWindowLinearTS(_SlidingWindowBase):
    """Sliding-window Linear Thompson Sampling."""

    def __init__(
        self,
        n_features: int,
        actions: Iterable[Any],
        window_size: int = 500,
        lambda_prior: float = 1.0,
        noise_std: float = 0.01,
        reward_clip: Optional[float] = 0.1,
        seed: int = 42,
        rebuild_interval: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_features,
            actions,
            window_size,
            lambda_prior,
            reward_clip,
            seed,
            rebuild_interval=rebuild_interval,
        )
        if float(noise_std) <= 0:
            raise ValueError("noise_std must be positive")
        self.noise_std = float(noise_std)

    def select_action(self, x: np.ndarray, return_scores: bool = False):
        x = self._validate_x(x)
        scores, means, uncertainty = {}, {}, {}

        for a in self.actions:
            mu = self._theta_hat(a)
            cov = (self.noise_std ** 2) * self.A_inv[a]
            cov = 0.5 * (cov + cov.T)
            theta_sample = self.rng.multivariate_normal(mean=mu, cov=cov)
            scores[a] = float(x @ theta_sample)
            means[a] = float(x @ mu)
            uncertainty[a] = self._uncertainty(a, x)

        action = max(scores, key=scores.get)
        self.action_counts[action] += 1

        if return_scores:
            return action, {"scores": scores, "means": means, "uncertainty": uncertainty}
        return action

    def diagnostics(self) -> List[dict]:
        rows = self._diagnostics_rows("sliding_window_lints")
        for row in rows:
            row.update({"noise_std": self.noise_std})
        return self._window_extra_diagnostics(rows)


class SlidingWindowLinearUCB(_SlidingWindowBase):
    """Sliding-window Linear UCB."""

    def __init__(
        self,
        n_features: int,
        actions: Iterable[Any],
        window_size: int = 500,
        lambda_prior: float = 1.0,
        ucb_alpha: float = 0.25,
        reward_clip: Optional[float] = 0.1,
        seed: int = 42,
        rebuild_interval: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_features,
            actions,
            window_size,
            lambda_prior,
            reward_clip,
            seed,
            rebuild_interval=rebuild_interval,
        )
        if float(ucb_alpha) < 0:
            raise ValueError("ucb_alpha must be non-negative")
        self.ucb_alpha = float(ucb_alpha)

    def select_action(self, x: np.ndarray, return_scores: bool = False):
        x = self._validate_x(x)
        scores, means, uncertainty = {}, {}, {}

        for a in self.actions:
            theta = self._theta_hat(a)
            mean = float(x @ theta)
            unc = self._uncertainty(a, x)
            scores[a] = mean + self.ucb_alpha * unc
            means[a] = mean
            uncertainty[a] = unc

        action = max(scores, key=scores.get)
        self.action_counts[action] += 1

        if return_scores:
            return action, {"scores": scores, "means": means, "uncertainty": uncertainty}
        return action

    def diagnostics(self) -> List[dict]:
        rows = self._diagnostics_rows("sliding_window_linucb")
        for row in rows:
            row.update({"ucb_alpha": self.ucb_alpha})
        return self._window_extra_diagnostics(rows)


# Short aliases that are sometimes convenient in configs.
DiscountingLinearTS = DiscountedLinearTS
DiscountingLinearUCB = DiscountedLinearUCB
SWLinearTS = SlidingWindowLinearTS
SWLinearUCB = SlidingWindowLinearUCB
