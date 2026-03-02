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
   - Components: win rate (50%), military efficiency (20%), economy (20%), speed (10%)
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

from openra_rl_util.rubric_base import (
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
            return 0.0  # No combat = no military score
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


class GameLengthRubric(TrajectoryRubric):
    """Score based on game speed — faster decisive games score higher.

    Score = 1.0 / (1.0 + ticks / 3000)
    Sigmoid-like decay: 1.0 at tick 0, 0.5 at 3000, ~0.23 at 10000.
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        ticks = getattr(final_obs, "tick", 0)
        return 1.0 / (1.0 + ticks / 3000)

    def compute_step_rewards(self) -> List[float]:
        if not self._trajectory:
            return []
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)


class OpenRABenchRubric(WeightedSum):
    """Composite benchmark score combining win/loss, military, economy, and speed.

    Weights: 50% win/loss, 20% military efficiency, 20% economy, 10% speed.
    """

    def __init__(self, gamma: float = 0.99):
        win_loss = OpenRAWinLossRubric(gamma=gamma)
        military = MilitaryEfficiencyRubric()
        economy = EconomyRubric()
        speed = GameLengthRubric()
        super().__init__(
            rubrics=[win_loss, military, economy, speed],
            weights=[0.5, 0.20, 0.20, 0.10],
        )
        # Keep named references for direct access
        self.win_loss = win_loss
        self.military = military
        self.economy = economy
        self.speed = speed

    def reset(self) -> None:
        self.win_loss.reset()
        self.military.reset()
        self.economy.reset()
        self.speed.reset()


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


def compute_composite_score_from_games(
    game_results: List[Dict[str, Any]],
    difficulty: str = "Normal",
) -> float:
    """Compute the OpenRA-Bench composite score from aggregated game results.

    This is the single source of truth for benchmark scoring. The formula
    matches OpenRABenchRubric: 50% win + 20% military + 20% economy + 10% speed.

    Args:
        game_results: List of dicts from compute_game_metrics().
        difficulty: AI opponent difficulty for score scaling.

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
        mil_scores.append(kills / total_cost if total_cost > 0 else 0.0)
    avg_mil = sum(mil_scores) / total

    # Per-game economy, then average (matches EconomyRubric)
    econ_scores = []
    for g in game_results:
        assets = g["assets_value"]
        econ_scores.append(assets / (assets + 10000) if assets >= 0 else 0.0)
    avg_econ = sum(econ_scores) / total

    # Per-game speed, then average (matches GameLengthRubric)
    speed_scores = []
    for g in game_results:
        ticks = g.get("ticks", 0)
        speed_scores.append(1.0 / (1.0 + ticks / 3000))
    avg_speed = sum(speed_scores) / total

    return 100.0 * (0.5 * win_rate + 0.20 * avg_mil + 0.20 * avg_econ + 0.10 * avg_speed)
