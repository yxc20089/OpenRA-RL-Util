"""Tests for rubrics module."""

import pytest

from openra_rl_util.rubrics import (
    EconomyRubric,
    MilitaryEfficiencyRubric,
    OpenRABenchRubric,
    OpenRAWinLossRubric,
    compute_composite_score_from_games,
    compute_game_metrics,
)

from tests.conftest import MockObs


class TestWinLossRubric:
    """Test win/loss scoring."""

    def test_win_scores_positive(self):
        r = OpenRAWinLossRubric(gamma=0.99)
        trajectory = [(None, MockObs(result="win"))]
        assert r.score_trajectory(trajectory) == 1.0

    def test_lose_scores_negative(self):
        r = OpenRAWinLossRubric(gamma=0.99)
        trajectory = [(None, MockObs(result="lose"))]
        assert r.score_trajectory(trajectory) == -1.0

    def test_draw_scores_zero(self):
        r = OpenRAWinLossRubric(gamma=0.99)
        trajectory = [(None, MockObs(result="draw"))]
        assert r.score_trajectory(trajectory) == 0.0

    def test_empty_trajectory(self):
        r = OpenRAWinLossRubric(gamma=0.99)
        assert r.score_trajectory([]) == 0.0


class TestMilitaryEfficiency:
    """Test military efficiency scoring."""

    def test_all_kills_no_deaths(self):
        r = MilitaryEfficiencyRubric()
        obs = MockObs(kills_cost=10000, deaths_cost=0)
        trajectory = [(None, obs)]
        assert r.score_trajectory(trajectory) == 1.0

    def test_all_deaths_no_kills(self):
        r = MilitaryEfficiencyRubric()
        obs = MockObs(kills_cost=0, deaths_cost=10000)
        trajectory = [(None, obs)]
        assert r.score_trajectory(trajectory) == 0.0

    def test_even_combat(self):
        r = MilitaryEfficiencyRubric()
        obs = MockObs(kills_cost=5000, deaths_cost=5000)
        trajectory = [(None, obs)]
        assert r.score_trajectory(trajectory) == 0.5

    def test_no_combat(self):
        """No kills and no deaths = 0.5 (neutral)."""
        r = MilitaryEfficiencyRubric()
        obs = MockObs(kills_cost=0, deaths_cost=0)
        trajectory = [(None, obs)]
        assert r.score_trajectory(trajectory) == 0.5

    def test_favorable_ratio(self):
        r = MilitaryEfficiencyRubric()
        obs = MockObs(kills_cost=8000, deaths_cost=2000)
        trajectory = [(None, obs)]
        score = r.score_trajectory(trajectory)
        assert score == 0.8

    def test_empty_trajectory(self):
        r = MilitaryEfficiencyRubric()
        assert r.score_trajectory([]) == 0.0


class TestEconomyRubric:
    """Test economy scoring."""

    def test_high_assets(self):
        r = EconomyRubric()
        obs = MockObs(assets_value=20000)
        trajectory = [(None, obs)]
        score = r.score_trajectory(trajectory)
        # 20000 / (20000 + 10000) = 0.667
        assert abs(score - 0.667) < 0.01

    def test_zero_assets(self):
        r = EconomyRubric()
        obs = MockObs(assets_value=0)
        trajectory = [(None, obs)]
        assert r.score_trajectory(trajectory) == 0.0

    def test_very_high_assets_approaches_one(self):
        r = EconomyRubric()
        obs = MockObs(assets_value=1000000)
        trajectory = [(None, obs)]
        score = r.score_trajectory(trajectory)
        assert score > 0.99

    def test_moderate_assets(self):
        r = EconomyRubric()
        obs = MockObs(assets_value=10000)
        trajectory = [(None, obs)]
        # 10000 / 20000 = 0.5
        assert r.score_trajectory(trajectory) == 0.5

    def test_empty_trajectory(self):
        r = EconomyRubric()
        assert r.score_trajectory([]) == 0.0


class TestCompositeRubric:
    """Test the combined benchmark rubric."""

    def test_perfect_game(self):
        r = OpenRABenchRubric()
        obs = MockObs(result="win", kills_cost=10000, deaths_cost=0, assets_value=50000)
        # WeightedSum uses forward() per step, not score_trajectory
        score = r(None, obs)
        # 50% * 1.0 + 25% * 1.0 + 25% * (50000/60000) = 0.5 + 0.25 + 0.208 = 0.958
        assert score > 0.9

    def test_reset_clears_state(self):
        r = OpenRABenchRubric()
        r.reset()
        # Should not raise


class TestComputeGameMetrics:
    """Test metrics extraction from observations."""

    def test_win_game(self):
        obs = MockObs(result="win", tick=2000, kills_cost=8000,
                      deaths_cost=2000, assets_value=15000, cash=3000)
        m = compute_game_metrics(obs)
        assert m["win"] is True
        assert m["result"] == "win"
        assert m["ticks"] == 2000
        assert m["kills_cost"] == 8000
        assert m["deaths_cost"] == 2000
        assert m["kd_ratio"] == 4.0
        assert m["assets_value"] == 15000
        assert m["cash"] == 3000

    def test_loss_game(self):
        obs = MockObs(result="lose", tick=1500, kills_cost=1000, deaths_cost=5000)
        m = compute_game_metrics(obs)
        assert m["win"] is False
        assert m["kd_ratio"] == 0.2

    def test_zero_deaths_kd(self):
        obs = MockObs(result="win", kills_cost=5000, deaths_cost=0)
        m = compute_game_metrics(obs)
        assert m["kd_ratio"] == 5000.0  # kills / max(deaths, 1)

    def test_no_military_data(self):
        obs = type("O", (), {"military": None, "economy": None, "result": "", "tick": 0})()
        m = compute_game_metrics(obs)
        assert m["kills_cost"] == 0
        assert m["deaths_cost"] == 0


class TestCompositeScoreFromGames:
    """Test aggregate scoring function."""

    def test_empty_games(self):
        assert compute_composite_score_from_games([]) == 0.0

    def test_all_wins_perfect(self):
        games = [
            {"win": True, "kills_cost": 10000, "deaths_cost": 0, "assets_value": 50000},
            {"win": True, "kills_cost": 8000, "deaths_cost": 0, "assets_value": 40000},
        ]
        score = compute_composite_score_from_games(games)
        assert score > 90  # Near-perfect across all dimensions

    def test_all_losses(self):
        games = [
            {"win": False, "kills_cost": 0, "deaths_cost": 10000, "assets_value": 0},
            {"win": False, "kills_cost": 0, "deaths_cost": 5000, "assets_value": 0},
        ]
        score = compute_composite_score_from_games(games)
        assert score < 10  # Very low but not zero (no-combat = 0.5 mil)

    def test_mixed_results(self):
        games = [
            {"win": True, "kills_cost": 8000, "deaths_cost": 2000, "assets_value": 15000},
            {"win": False, "kills_cost": 3000, "deaths_cost": 7000, "assets_value": 5000},
        ]
        score = compute_composite_score_from_games(games)
        assert 20 < score < 80

    def test_score_range(self):
        """Score should always be 0-100."""
        games = [
            {"win": True, "kills_cost": 10000, "deaths_cost": 0, "assets_value": 999999},
        ]
        score = compute_composite_score_from_games(games)
        assert 0 <= score <= 100

    def test_no_combat_games(self):
        games = [
            {"win": True, "kills_cost": 0, "deaths_cost": 0, "assets_value": 10000},
        ]
        score = compute_composite_score_from_games(games)
        # 50% * 1.0 + 25% * 0.5 + 25% * 0.5 = 75.0
        assert abs(score - 75.0) < 0.1

    def test_single_game_matches_rubric(self):
        """Composite score from 1 game should match rubric computation."""
        games = [
            {"win": True, "kills_cost": 8000, "deaths_cost": 2000, "assets_value": 20000},
        ]
        score = compute_composite_score_from_games(games)
        # Manual: 50%*1.0 + 25%*(8000/10000) + 25%*(20000/30000) = 50 + 20 + 16.67 = 86.67
        assert abs(score - 86.67) < 0.1
