"""OpenRA-Bench rubrics for agent evaluation.

Follows the OpenEnv rubric pattern (see openenv.core.rubrics).
These rubrics score game episodes based on win/loss, military efficiency,
and economic performance.

Scoring Systems in the OpenRA-RL Ecosystem
==========================================

There are two intentionally separate scoring systems:

1. **Reward Vector** (reward_vector.py):
   - Per-tick, multi-dimensional training signal for RL
   - 7 skill dimensions + terminal outcome
   - Dense, delta-based signal for policy gradient
   - Used during agent training and gameplay

2. **Benchmark Score** (this file):
   - Episode-level composite score for leaderboard ranking
   - Components: win rate (50%), military efficiency (25%), economy (25%)
   - Normalized to 0-100 scale
   - Used by OpenRA-Bench evaluation harness

These are complementary: the reward vector trains agent skills,
the benchmark measures overall performance. An agent with strong
intelligence and tempo may win games through positioning rather than
raw combat — the benchmark captures the win, the vector explains why.

Usage:
    rubric = OpenRABenchRubric()
    rubric.reset()
    for action, obs in episode:
        reward = rubric(action, obs)  # 0.0 until done
    step_rewards = rubric.win_loss.compute_step_rewards()
"""

from typing import Any, Dict, List, Tuple

from openenv.core.rubrics import (
    ExponentialDiscountingTrajectoryRubric,
    TrajectoryRubric,
    WeightedSum,
)


class OpenRAWinLossRubric(ExponentialDiscountingTrajectoryRubric):
    """Score game based on win/loss/draw outcome with temporal discounting.

    Terminal rewards:
    - Win:  +1.0
    - Loss: -1.0
    - Draw:  0.0
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        result = getattr(final_obs, "result", "")
        if result == "win":
            return 1.0
        elif result == "lose":
            return -1.0
        return 0.0


class MilitaryEfficiencyRubric(TrajectoryRubric):
    """Score based on kill/death cost ratio from final observation.

    Score = kills_cost / max(kills_cost + deaths_cost, 1)
    Normalized to 0.0-1.0 range.
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        military = getattr(final_obs, "military", None)
        if military is None:
            return 0.0
        kills = getattr(military, "kills_cost", 0)
        deaths = getattr(military, "deaths_cost", 0)
        total = kills + deaths
        if total == 0:
            return 0.5  # No combat occurred
        return kills / total

    def compute_step_rewards(self) -> List[float]:
        if not self._trajectory:
            return []
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)


class EconomyRubric(TrajectoryRubric):
    """Score based on final economic state.

    Score = assets_value / (assets_value + 10000)
    Sigmoid-like normalization to 0.0-1.0 range.
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        military = getattr(final_obs, "military", None)
        if military is None:
            return 0.0
        assets = getattr(military, "assets_value", 0)
        # Sigmoid normalization: maps [0, inf) -> [0, 1)
        return assets / (assets + 10000) if assets >= 0 else 0.0

    def compute_step_rewards(self) -> List[float]:
        if not self._trajectory:
            return []
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)


class OpenRABenchRubric(WeightedSum):
    """Composite benchmark score combining win/loss, military, and economy.

    Weights: 50% win/loss, 25% military efficiency, 25% economy.
    """

    def __init__(self, gamma: float = 0.99):
        win_loss = OpenRAWinLossRubric(gamma=gamma)
        military = MilitaryEfficiencyRubric()
        economy = EconomyRubric()
        super().__init__(
            rubrics=[win_loss, military, economy],
            weights=[0.5, 0.25, 0.25],
        )
        # Keep named references for direct access
        self.win_loss = win_loss
        self.military = military
        self.economy = economy

    def reset(self) -> None:
        self.win_loss.reset()
        self.military.reset()
        self.economy.reset()


def compute_game_metrics(final_obs: Any) -> Dict[str, Any]:
    """Extract benchmark metrics from a final game observation.

    Args:
        final_obs: The terminal GameObservation (where done=True).

    Returns:
        Dict with keys: result, ticks, kills_cost, deaths_cost,
        kd_ratio, assets_value, cash, win (bool).
    """
    military = getattr(final_obs, "military", None)
    economy = getattr(final_obs, "economy", None)

    kills = getattr(military, "kills_cost", 0) if military else 0
    deaths = getattr(military, "deaths_cost", 0) if military else 0
    assets = getattr(military, "assets_value", 0) if military else 0
    cash = getattr(economy, "cash", 0) if economy else 0
    result = getattr(final_obs, "result", "")
    tick = getattr(final_obs, "tick", 0)

    return {
        "result": result,
        "win": result == "win",
        "ticks": tick,
        "kills_cost": kills,
        "deaths_cost": deaths,
        "kd_ratio": kills / max(deaths, 1),
        "assets_value": assets,
        "cash": cash,
    }


def compute_composite_score_from_games(game_results: List[Dict[str, Any]]) -> float:
    """Compute the OpenRA-Bench composite score from aggregated game results.

    This is the single source of truth for benchmark scoring. The formula
    matches OpenRABenchRubric: 50% win rate + 25% military efficiency + 25% economy.

    Per-game rubric formulas are averaged (not ratio-then-normalize) to avoid
    Jensen's inequality distortion on multi-game aggregation.

    Args:
        game_results: List of dicts from compute_game_metrics().

    Returns:
        Composite score on 0-100 scale.
    """
    total = len(game_results)
    if total == 0:
        return 0.0

    # Win rate
    win_rate = sum(1 for g in game_results if g["win"]) / total

    # Per-game military efficiency, then average (matches MilitaryEfficiencyRubric)
    mil_scores = []
    for g in game_results:
        kills, deaths = g["kills_cost"], g["deaths_cost"]
        total_cost = kills + deaths
        mil_scores.append(kills / total_cost if total_cost > 0 else 0.5)
    avg_mil = sum(mil_scores) / total

    # Per-game economy, then average (matches EconomyRubric)
    econ_scores = []
    for g in game_results:
        assets = g["assets_value"]
        econ_scores.append(assets / (assets + 10000) if assets >= 0 else 0.0)
    avg_econ = sum(econ_scores) / total

    return 100.0 * (0.5 * win_rate + 0.25 * avg_mil + 0.25 * avg_econ)
