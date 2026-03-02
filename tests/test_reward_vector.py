"""Tests for reward_vector module."""

import copy

from openra_rl_util.reward_vector import (
    DEFAULT_WEIGHTS,
    RewardVector,
    RewardVectorComputer,
    RewardVectorState,
)


class TestRewardVector:
    """Test RewardVector dataclass."""

    def test_default_is_zero(self):
        v = RewardVector()
        assert all(x == 0.0 for x in v.as_array())

    def test_as_dict(self):
        v = RewardVector(combat=0.5, economy=-0.2)
        d = v.as_dict()
        assert d["combat"] == 0.5
        assert d["economy"] == -0.2
        assert d["outcome"] == 0.0

    def test_as_array_length(self):
        v = RewardVector()
        assert len(v.as_array()) == 8

    def test_weighted_scalar_default(self):
        v = RewardVector(combat=1.0, economy=1.0, outcome=1.0)
        s = v.weighted_scalar()
        expected = 0.30 + 0.15 + 1.0  # combat + economy + outcome weights
        assert abs(s - expected) < 0.001

    def test_weighted_scalar_custom(self):
        v = RewardVector(combat=1.0)
        s = v.weighted_scalar({"combat": 2.0})
        assert s == 2.0

    def test_weighted_scalar_zero_vector(self):
        v = RewardVector()
        assert v.weighted_scalar() == 0.0


class TestRewardVectorComputer:
    """Test the full reward computation pipeline."""

    def test_reset_clears_state(self):
        c = RewardVectorComputer()
        c._state.prev_kills_cost = 999
        c._state.discovered_enemy_base = True
        c.reset()
        assert c._state.prev_kills_cost == 0
        assert c._state.discovered_enemy_base is False

    def test_empty_obs_returns_zero_vector(self, empty_obs):
        c = RewardVectorComputer()
        v = c.compute(empty_obs)
        # First tick: most signals should be near zero
        assert isinstance(v, RewardVector)

    def test_win_sets_outcome(self, empty_obs):
        c = RewardVectorComputer()
        c.compute(empty_obs)  # tick 0
        obs2 = copy.deepcopy(empty_obs)
        obs2["done"] = True
        obs2["result"] = "win"
        v = c.compute(obs2)
        assert v.outcome == 1.0

    def test_lose_sets_negative_outcome(self, empty_obs):
        c = RewardVectorComputer()
        c.compute(empty_obs)
        obs2 = copy.deepcopy(empty_obs)
        obs2["done"] = True
        obs2["result"] = "lose"
        v = c.compute(obs2)
        assert v.outcome == -1.0

    def test_draw_has_zero_outcome(self, empty_obs):
        c = RewardVectorComputer()
        c.compute(empty_obs)
        obs2 = copy.deepcopy(empty_obs)
        obs2["done"] = True
        obs2["result"] = "draw"
        v = c.compute(obs2)
        assert v.outcome == 0.0


class TestCombatDimension:
    """Test combat reward computation."""

    def test_kills_cost_increases_combat(self, basic_obs):
        c = RewardVectorComputer()
        c.compute(basic_obs)  # baseline

        obs2 = copy.deepcopy(basic_obs)
        obs2["military"]["kills_cost"] = 1000  # killed $1000 of enemies
        v = c.compute(obs2)
        assert v.combat > 0

    def test_deaths_cost_decreases_combat(self, basic_obs):
        c = RewardVectorComputer()
        c.compute(basic_obs)

        obs2 = copy.deepcopy(basic_obs)
        obs2["military"]["deaths_cost"] = 2000  # lost $2000 of own units
        v = c.compute(obs2)
        assert v.combat < 0

    def test_partial_damage_to_enemy(self, basic_obs):
        """Reducing enemy HP without killing rewards combat."""
        c = RewardVectorComputer()
        c.compute(basic_obs)

        obs2 = copy.deepcopy(basic_obs)
        # Enemy tank went from 0.8 to 0.3 HP
        obs2["visible_enemies"][1]["hp_percent"] = 0.3
        v = c.compute(obs2)
        assert v.combat > 0, "Damaging enemy tank should reward combat"

    def test_own_unit_taking_damage(self, basic_obs):
        """Own units losing HP penalizes combat."""
        c = RewardVectorComputer()
        c.compute(basic_obs)

        obs2 = copy.deepcopy(basic_obs)
        obs2["units"][3]["hp_percent"] = 0.5  # Light tank damaged
        v = c.compute(obs2)
        assert v.combat < 0, "Own unit taking damage should penalize combat"

    def test_combat_is_clamped(self, empty_obs):
        """Combat reward stays in [-1, 1]."""
        c = RewardVectorComputer()
        c.compute(empty_obs)

        obs2 = copy.deepcopy(empty_obs)
        obs2["military"]["kills_cost"] = 999999
        v = c.compute(obs2)
        assert -1.0 <= v.combat <= 1.0


class TestEconomyDimension:
    """Test economy reward computation."""

    def test_cash_increase_rewards_economy(self, basic_obs):
        c = RewardVectorComputer()
        c.compute(basic_obs)

        obs2 = copy.deepcopy(basic_obs)
        obs2["economy"]["cash"] = 5000  # +2000 cash
        v = c.compute(obs2)
        assert v.economy > 0

    def test_harvester_loss_penalizes_economy(self, basic_obs):
        c = RewardVectorComputer()
        c.compute(basic_obs)

        obs2 = copy.deepcopy(basic_obs)
        obs2["economy"]["harvester_count"] = 1  # Lost one harvester
        v = c.compute(obs2)
        assert v.economy < 0, "Losing harvester should penalize economy"

    def test_economy_is_clamped(self, empty_obs):
        c = RewardVectorComputer()
        c.compute(empty_obs)

        obs2 = copy.deepcopy(empty_obs)
        obs2["economy"]["cash"] = 999999
        v = c.compute(obs2)
        assert -1.0 <= v.economy <= 1.0


class TestInfrastructureDimension:
    """Test infrastructure reward computation."""

    def test_new_building_type_rewards(self, empty_obs):
        c = RewardVectorComputer()
        c.compute(empty_obs)

        obs2 = copy.deepcopy(empty_obs)
        obs2["buildings"] = [
            {"actor_id": 100, "type": "fact", "hp_percent": 1.0, "is_producing": False},
            {"actor_id": 101, "type": "powr", "hp_percent": 1.0, "is_producing": False},
        ]
        v = c.compute(obs2)
        assert v.infrastructure > 0, "New buildings should reward infrastructure"

    def test_production_utilization(self, empty_obs):
        c = RewardVectorComputer()
        c.compute(empty_obs)

        obs2 = copy.deepcopy(empty_obs)
        obs2["buildings"] = [
            {"actor_id": 100, "type": "fact", "hp_percent": 1.0, "is_producing": False},
            {"actor_id": 101, "type": "tent", "hp_percent": 1.0, "is_producing": True},
            {"actor_id": 102, "type": "weap", "hp_percent": 1.0, "is_producing": True},
        ]
        v = c.compute(obs2)
        # 2/2 production buildings are active = good utilization
        assert v.infrastructure > 0

    def test_power_deficit_penalizes(self, empty_obs):
        c = RewardVectorComputer()
        # Already built buildings in previous tick
        c._state.own_building_types_built = {"fact", "powr", "tent"}
        c.compute(empty_obs)

        obs2 = copy.deepcopy(empty_obs)
        obs2["economy"]["power_provided"] = 50
        obs2["economy"]["power_drained"] = 200  # deficit
        obs2["buildings"] = [
            {"actor_id": 100, "type": "fact", "hp_percent": 1.0, "is_producing": False},
        ]
        v = c.compute(obs2)
        assert v.infrastructure < 0, "Power deficit should penalize infrastructure"


class TestIntelligenceDimension:
    """Test intelligence reward computation."""

    def test_first_enemy_base_discovery(self, empty_obs):
        c = RewardVectorComputer()
        c.compute(empty_obs)

        obs2 = copy.deepcopy(empty_obs)
        obs2["visible_enemy_buildings"] = [
            {"actor_id": 300, "type": "fact", "hp_percent": 1.0},
        ]
        v = c.compute(obs2)
        assert v.intelligence > 0, "Discovering enemy base should reward intel"

    def test_new_enemy_units_spotted(self, empty_obs):
        c = RewardVectorComputer()
        c.compute(empty_obs)

        obs2 = copy.deepcopy(empty_obs)
        obs2["visible_enemies"] = [
            {"actor_id": 200, "type": "e1", "hp_percent": 1.0},
            {"actor_id": 201, "type": "3tnk", "hp_percent": 1.0},
        ]
        v = c.compute(obs2)
        assert v.intelligence > 0, "Spotting new enemies should reward intel"

    def test_no_new_info_zero_intel(self, basic_obs):
        """Seeing the same enemies twice gives no new intel."""
        c = RewardVectorComputer()
        c.compute(basic_obs)
        v = c.compute(basic_obs)  # same obs again
        assert v.intelligence == 0.0, "No new info should give zero intel"


class TestCompositionDimension:
    """Test force composition scoring."""

    def test_rockets_counter_tanks(self, empty_obs):
        c = RewardVectorComputer()
        obs = copy.deepcopy(empty_obs)
        obs["units"] = [
            {"type": "e3"}, {"type": "e3"}, {"type": "e3"},
        ]
        obs["visible_enemies"] = [
            {"type": "3tnk"}, {"type": "4tnk"},
        ]
        v = c.compute(obs)
        assert v.composition > 0, "Rockets vs tanks should give positive composition"

    def test_rifles_weak_vs_tanks(self, empty_obs):
        c = RewardVectorComputer()
        obs = copy.deepcopy(empty_obs)
        obs["units"] = [
            {"type": "e1"}, {"type": "e1"}, {"type": "e1"},
        ]
        obs["visible_enemies"] = [
            {"type": "3tnk"}, {"type": "4tnk"},
        ]
        v = c.compute(obs)
        assert v.composition < 0, "Rifles vs tanks should give negative composition"

    def test_no_enemies_zero_composition(self, empty_obs):
        c = RewardVectorComputer()
        v = c.compute(empty_obs)
        assert v.composition == 0.0


class TestTempoDimension:
    """Test tempo (time-efficiency) scoring."""

    def test_idle_combat_units_penalize(self, empty_obs):
        c = RewardVectorComputer()
        obs = copy.deepcopy(empty_obs)
        obs["units"] = [
            {"type": "e1", "is_idle": True},
            {"type": "e1", "is_idle": True},
            {"type": "1tnk", "is_idle": True},
        ]
        obs["military"]["order_count"] = 0
        c.compute(obs)  # baseline

        obs2 = copy.deepcopy(obs)
        v = c.compute(obs2)
        assert v.tempo <= 0, "All idle combat units should penalize tempo"

    def test_active_units_positive_tempo(self, empty_obs):
        c = RewardVectorComputer()
        obs = copy.deepcopy(empty_obs)
        obs["units"] = [
            {"type": "e1", "is_idle": False},
            {"type": "1tnk", "is_idle": False},
        ]
        obs["military"]["order_count"] = 0
        c.compute(obs)

        obs2 = copy.deepcopy(obs)
        obs2["military"]["order_count"] = 10  # 10 new orders
        v = c.compute(obs2)
        assert v.tempo > 0, "Active units + orders should give positive tempo"


class TestDisruptionDimension:
    """Test strategic disruption scoring."""

    def test_enemy_power_plant_destroyed(self, empty_obs):
        c = RewardVectorComputer()
        obs = copy.deepcopy(empty_obs)
        obs["visible_enemy_buildings"] = [
            {"actor_id": 300, "type": "powr", "hp_percent": 1.0},
            {"actor_id": 301, "type": "weap", "hp_percent": 1.0},
        ]
        c.compute(obs)

        obs2 = copy.deepcopy(obs)
        # Power plant disappeared (destroyed)
        obs2["visible_enemy_buildings"] = [
            {"actor_id": 301, "type": "weap", "hp_percent": 1.0},
        ]
        v = c.compute(obs2)
        assert v.disruption > 0, "Destroying enemy power should reward disruption"

    def test_enemy_production_destroyed(self, empty_obs):
        c = RewardVectorComputer()
        obs = copy.deepcopy(empty_obs)
        obs["visible_enemy_buildings"] = [
            {"actor_id": 300, "type": "weap", "hp_percent": 1.0},
        ]
        c.compute(obs)

        obs2 = copy.deepcopy(obs)
        obs2["visible_enemy_buildings"] = []  # War factory destroyed
        v = c.compute(obs2)
        assert v.disruption > 0, "Destroying enemy factory should reward disruption"

    def test_no_disruption_when_stable(self, basic_obs):
        c = RewardVectorComputer()
        c.compute(basic_obs)
        v = c.compute(basic_obs)
        assert v.disruption == 0.0, "Stable enemy should give zero disruption"


class TestMultiTick:
    """Test multi-tick reward sequences."""

    def test_two_ticks_accumulate_correctly(self, empty_obs):
        c = RewardVectorComputer()
        v1 = c.compute(empty_obs)

        obs2 = copy.deepcopy(empty_obs)
        obs2["military"]["kills_cost"] = 500
        obs2["economy"]["cash"] = 6000  # +1000
        v2 = c.compute(obs2)

        assert v2.combat > v1.combat, "Second tick with kills should have higher combat"
        assert v2.economy > 0, "Cash increase should give positive economy"

    def test_reset_between_episodes(self, empty_obs, basic_obs):
        c = RewardVectorComputer()
        c.compute(basic_obs)
        c.reset()

        # After reset, should act like first tick
        v = c.compute(empty_obs)
        assert isinstance(v, RewardVector)
        assert c._state.prev_kills_cost == 0

    def test_all_dimensions_bounded(self, basic_obs):
        """All dimensions should stay in [-1, 1]."""
        c = RewardVectorComputer()
        v = c.compute(basic_obs)
        for dim_name, val in v.as_dict().items():
            assert -1.0 <= val <= 1.0, f"{dim_name} = {val} out of bounds"


class TestEconomyHoardingPenalty:
    """Economy reward should favor investment over hoarding."""

    def test_hoarding_cash_scores_less_than_investing(self):
        """Same total growth but cash-heavy should score less than assets-heavy."""
        c1 = RewardVectorComputer()
        c2 = RewardVectorComputer()

        # Tick 0: both start same
        base_obs = {
            "military": {"kills_cost": 0, "deaths_cost": 0, "units_killed": 0,
                         "units_lost": 0, "buildings_killed": 0, "buildings_lost": 0,
                         "assets_value": 5000, "order_count": 0},
            "economy": {"cash": 5000, "ore": 0, "power_provided": 100,
                        "power_drained": 0, "harvester_count": 1},
            "units": [], "buildings": [], "visible_enemies": [],
            "visible_enemy_buildings": [], "production_queues": [],
            "done": False, "result": "", "tick": 0,
        }
        c1.compute(dict(base_obs))
        c2.compute(dict(base_obs))

        import copy
        # Hoarding agent: cash grows, assets flat
        hoarding = copy.deepcopy(base_obs)
        hoarding["economy"]["cash"] = 10000  # +5000 cash
        hoarding["military"]["assets_value"] = 5000  # unchanged
        hoarding["tick"] = 1

        # Investing agent: cash flat, assets grow
        investing = copy.deepcopy(base_obs)
        investing["economy"]["cash"] = 5000  # unchanged
        investing["military"]["assets_value"] = 10000  # +5000 assets
        investing["tick"] = 1

        v_hoard = c1.compute(hoarding)
        v_invest = c2.compute(investing)

        # Same total growth (+5000) but investor should score higher
        assert v_invest.economy > v_hoard.economy
