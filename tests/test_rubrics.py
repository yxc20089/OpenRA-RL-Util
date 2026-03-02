"""Tests for rubrics module."""

import pytest

from openra_rl_util.rubrics import (
    EconomyRubric,
    GameLengthRubric,
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
        """No kills and no deaths = 0.0 (no military credit for avoiding combat)."""
        r = MilitaryEfficiencyRubric()
        obs = MockObs(kills_cost=0, deaths_cost=0)
        trajectory = [(None, obs)]
        assert r.score_trajectory(trajectory) == 0.0

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


class TestGameLengthRubric:
    """Test game speed scoring."""

    def test_instant_game(self):
        r = GameLengthRubric()
        obs = MockObs(tick=0)
        trajectory = [(None, obs)]
        assert r.score_trajectory(trajectory) == 1.0

    def test_medium_game(self):
        r = GameLengthRubric()
        obs = MockObs(tick=3000)
        trajectory = [(None, obs)]
        assert r.score_trajectory(trajectory) == 0.5

    def test_long_game(self):
        r = GameLengthRubric()
        obs = MockObs(tick=10000)
        trajectory = [(None, obs)]
        score = r.score_trajectory(trajectory)
        assert abs(score - 1.0 / (1.0 + 10000 / 3000)) < 0.01

    def test_empty_trajectory(self):
        r = GameLengthRubric()
        assert r.score_trajectory([]) == 0.0


class TestCompositeRubric:
    """Test the combined benchmark rubric."""

    def test_perfect_game(self):
        r = OpenRABenchRubric()
        obs = MockObs(result="win", kills_cost=10000, deaths_cost=0, assets_value=50000)
        # WeightedSum uses forward() per step, not score_trajectory
        score = r(None, obs)
        # 50% * 1.0 + 20% * 1.0 + 20% * (50000/60000) + 10% * speed
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
            {"win": True, "kills_cost": 10000, "deaths_cost": 0, "assets_value": 50000, "ticks": 2000},
            {"win": True, "kills_cost": 8000, "deaths_cost": 0, "assets_value": 40000, "ticks": 2000},
        ]
        score = compute_composite_score_from_games(games)
        assert score > 90  # Near-perfect across all dimensions

    def test_all_losses(self):
        games = [
            {"win": False, "kills_cost": 0, "deaths_cost": 10000, "assets_value": 0, "ticks": 5000},
            {"win": False, "kills_cost": 0, "deaths_cost": 5000, "assets_value": 0, "ticks": 3000},
        ]
        score = compute_composite_score_from_games(games)
        assert score < 10  # Very low

    def test_mixed_results(self):
        games = [
            {"win": True, "kills_cost": 8000, "deaths_cost": 2000, "assets_value": 15000, "ticks": 2000},
            {"win": False, "kills_cost": 3000, "deaths_cost": 7000, "assets_value": 5000, "ticks": 2000},
        ]
        score = compute_composite_score_from_games(games)
        assert 20 < score < 80

    def test_score_range(self):
        """Score should always be 0-100."""
        games = [
            {"win": True, "kills_cost": 10000, "deaths_cost": 0, "assets_value": 999999, "ticks": 2000},
        ]
        score = compute_composite_score_from_games(games)
        assert 0 <= score <= 100

    def test_no_combat_games(self):
        games = [
            {"win": True, "kills_cost": 0, "deaths_cost": 0, "assets_value": 10000, "ticks": 2000},
        ]
        score = compute_composite_score_from_games(games)
        # 50%*1.0 + 20%*0.0 + 20%*0.5 + 10%*(1/(1+2000/3000))
        # = 50 + 0 + 10 + 10*(0.6) = 66.0
        assert abs(score - 66.0) < 0.5

    def test_single_game_matches_rubric(self):
        """Composite score from 1 game should match rubric computation."""
        games = [
            {"win": True, "kills_cost": 8000, "deaths_cost": 2000, "assets_value": 20000, "ticks": 1500},
        ]
        score = compute_composite_score_from_games(games)
        # 50%*1.0 + 20%*(8000/10000) + 20%*(20000/30000) + 10%*(1/(1+0.5))
        # = 50 + 16 + 13.33 + 6.67 = 86.0
        assert abs(score - 86.0) < 0.5
