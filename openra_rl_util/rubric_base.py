"""Minimal rubric base classes vendored from OpenEnv.

These implement the core Rubric API (forward/call pattern, trajectory
accumulation, exponential discounting, weighted sum). When openenv-core
publishes openenv.core.rubrics on PyPI, this file can be replaced with
a re-export:

    from openenv.core.rubrics import (  # noqa: F401
        Rubric, TrajectoryRubric,
        ExponentialDiscountingTrajectoryRubric, WeightedSum,
    )

Source: https://github.com/OpenEnvs/OpenEnv  (BSD license)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class Rubric(ABC):
    """Abstract base class for reward computation.

    Subclasses implement forward() to define reward logic.
    Call via rubric(action, observation).
    """

    def __init__(self):
        self.last_score: float | None = None

    def __call__(self, action: Any, observation: Any) -> float:
        result = self.forward(action, observation)
        self.last_score = result
        return result

    @abstractmethod
    def forward(self, action: Any, observation: Any) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass


class TrajectoryRubric(Rubric):
    """Rubric that accumulates trajectory and scores at episode end.

    Subclasses implement score_trajectory() and compute_step_rewards().
    """

    def __init__(self, intermediate_reward: float = 0.0):
        super().__init__()
        self.intermediate_reward = intermediate_reward
        self._trajectory: List[Tuple[Any, Any]] = []

    def forward(self, action: Any, observation: Any) -> float:
        self._trajectory.append((action, observation))
        if getattr(observation, "done", False):
            return self.score_trajectory(self._trajectory)
        return self.intermediate_reward

    @abstractmethod
    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_step_rewards(self) -> List[float]:
        raise NotImplementedError

    def reset(self) -> None:
        self._trajectory = []

    @property
    def trajectory(self) -> List[Tuple[Any, Any]]:
        return list(self._trajectory)


class ExponentialDiscountingTrajectoryRubric(TrajectoryRubric):
    """TrajectoryRubric with exponential discounting for credit assignment.

    Per-step reward: r_t = gamma^(T-1-t) * R_final
    """

    def __init__(self, gamma: float = 0.99, intermediate_reward: float = 0.0):
        super().__init__(intermediate_reward=intermediate_reward)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self.gamma = gamma

    def compute_step_rewards(self) -> List[float]:
        if not self._trajectory:
            return []
        final_score = self.score_trajectory(self._trajectory)
        T = len(self._trajectory)
        return [final_score * (self.gamma ** (T - 1 - t)) for t in range(T)]


class WeightedSum(Rubric):
    """Weighted combination of child rubrics."""

    def __init__(self, rubrics: List[Rubric], weights: List[float]):
        super().__init__()
        if len(rubrics) != len(weights):
            raise ValueError(
                f"Number of rubrics ({len(rubrics)}) must match "
                f"number of weights ({len(weights)})"
            )
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
        self._rubric_list = list(rubrics)
        self._weights = list(weights)

    def forward(self, action: Any, observation: Any) -> float:
        total = 0.0
        for rubric, weight in zip(self._rubric_list, self._weights):
            score = rubric(action, observation)
            total += score * weight
        return total

    @property
    def weights(self) -> List[float]:
        return list(self._weights)
