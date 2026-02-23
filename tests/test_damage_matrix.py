"""Tests for damage_matrix module."""

from openra_rl_util.damage_matrix import (
    BUILDING_ARMOR,
    BUILDING_COST,
    ECONOMIC_BUILDINGS,
    ECONOMIC_UNITS,
    NON_COMBAT_UNITS,
    POWER_BUILDINGS,
    PRODUCTION_BUILDINGS,
    TECH_BUILDINGS,
    UNIT_ARMOR,
    UNIT_COST,
    UNIT_EFFECTIVENESS,
    can_attack,
    compute_army_counter_score,
    get_building_armor,
    get_building_cost,
    get_effectiveness,
    get_unit_armor,
    get_unit_cost,
    get_unit_vs_unit,
    is_economic_target,
    is_power_target,
    is_production_target,
    is_tech_target,
)


class TestUnitData:
    """Test unit data completeness and consistency."""

    def test_all_units_have_armor(self):
        """Every unit in effectiveness table has armor defined."""
        for utype in UNIT_EFFECTIVENESS:
            assert utype in UNIT_ARMOR, f"{utype} missing from UNIT_ARMOR"

    def test_all_units_have_cost(self):
        """Every unit in effectiveness table has cost defined."""
        for utype in UNIT_EFFECTIVENESS:
            assert utype in UNIT_COST, f"{utype} missing from UNIT_COST"

    def test_infantry_armor_is_none(self):
        """All infantry have 'none' armor type."""
        infantry = ["e1", "e2", "e3", "e4", "e6", "e7", "medi", "mech",
                     "spy", "thf", "shok", "dog"]
        for u in infantry:
            assert UNIT_ARMOR[u] == "none", f"{u} should have 'none' armor"

    def test_tanks_armor_is_heavy(self):
        """All tanks have 'heavy' armor."""
        tanks = ["1tnk", "2tnk", "3tnk", "4tnk"]
        for t in tanks:
            assert UNIT_ARMOR[t] == "heavy", f"{t} should have 'heavy' armor"

    def test_aircraft_armor_is_light(self):
        """All aircraft have 'light' armor."""
        aircraft = ["heli", "hind", "mh60", "tran", "yak", "mig"]
        for a in aircraft:
            assert UNIT_ARMOR[a] == "light", f"{a} should have 'light' armor"


class TestEffectiveness:
    """Test weapon effectiveness lookups."""

    def test_rifle_vs_infantry_is_strong(self):
        assert get_effectiveness("e1", "none") == 1.5

    def test_rifle_vs_heavy_armor_is_weak(self):
        assert get_effectiveness("e1", "heavy") == 0.1

    def test_rocket_vs_heavy_is_effective(self):
        assert get_effectiveness("e3", "heavy") == 1.0

    def test_rocket_vs_infantry_is_weak(self):
        assert get_effectiveness("e3", "none") == 0.1

    def test_grenadier_vs_wood_buildings(self):
        assert get_effectiveness("e2", "wood") == 1.0

    def test_grenadier_vs_concrete_buildings(self):
        assert get_effectiveness("e2", "concrete") == 1.0

    def test_tesla_vs_infantry_is_devastating(self):
        assert get_effectiveness("shok", "none") == 10.0

    def test_light_tank_vs_light_is_effective(self):
        assert get_effectiveness("1tnk", "light") == 1.16

    def test_medium_tank_vs_heavy_is_effective(self):
        assert get_effectiveness("2tnk", "heavy") == 1.15

    def test_non_combat_unit_returns_zero(self):
        assert get_effectiveness("harv", "none") == 0.0
        assert get_effectiveness("mcv", "heavy") == 0.0
        assert get_effectiveness("e6", "none") == 0.0

    def test_unknown_unit_returns_zero(self):
        assert get_effectiveness("nonexistent", "none") == 0.0

    def test_unknown_armor_defaults_to_one(self):
        # For units with weapon data, unknown armor = full damage (1.0 default)
        assert get_effectiveness("e1", "unknown_armor") == 1.0


class TestUnitVsUnit:
    """Test unit vs unit effectiveness."""

    def test_rifle_vs_rifle(self):
        """Rifle vs rifle = 1.5 (infantry armor = none, rifle bonus vs none)."""
        assert get_unit_vs_unit("e1", "e1") == 1.5

    def test_rifle_vs_tank(self):
        """Rifle vs heavy tank = very weak."""
        assert get_unit_vs_unit("e1", "3tnk") == 0.1

    def test_rocket_vs_tank(self):
        """Rocket soldier vs tank = effective."""
        assert get_unit_vs_unit("e3", "3tnk") == 1.0

    def test_tank_vs_infantry(self):
        """Medium tank vs infantry = weak (cannon vs infantry)."""
        assert get_unit_vs_unit("2tnk", "e1") == 0.3


class TestCanAttack:
    """Test combat capability detection."""

    def test_combat_units_can_attack(self):
        for u in ["e1", "e2", "e3", "1tnk", "2tnk", "heli", "mig"]:
            assert can_attack(u), f"{u} should be able to attack"

    def test_non_combat_units_cannot_attack(self):
        for u in ["harv", "mcv", "e6", "medi", "mech", "thf", "tran", "lst"]:
            assert not can_attack(u), f"{u} should not be able to attack"

    def test_non_combat_set_is_correct(self):
        expected_non_combat = {"e6", "medi", "mech", "thf", "harv", "mcv",
                               "mnly", "qtnk", "dtrk", "mgg", "mrj", "truk",
                               "tran", "lst"}
        assert NON_COMBAT_UNITS == expected_non_combat


class TestBuildingClassification:
    """Test building role classification."""

    def test_economic_buildings(self):
        assert "proc" in ECONOMIC_BUILDINGS
        assert "silo" in ECONOMIC_BUILDINGS

    def test_production_buildings(self):
        for b in ["barr", "tent", "weap", "hpad", "afld"]:
            assert b in PRODUCTION_BUILDINGS

    def test_tech_buildings(self):
        for b in ["dome", "atek", "stek", "fix"]:
            assert b in TECH_BUILDINGS

    def test_power_buildings(self):
        assert "powr" in POWER_BUILDINGS
        assert "apwr" in POWER_BUILDINGS

    def test_is_economic_target(self):
        assert is_economic_target("harv")
        assert is_economic_target("proc")
        assert not is_economic_target("e1")

    def test_is_production_target(self):
        assert is_production_target("weap")
        assert not is_production_target("powr")

    def test_is_tech_target(self):
        assert is_tech_target("dome")
        assert not is_tech_target("barr")

    def test_is_power_target(self):
        assert is_power_target("powr")
        assert not is_power_target("weap")


class TestArmyCounterScore:
    """Test army composition scoring."""

    def test_empty_armies(self):
        counter, vuln = compute_army_counter_score([], [])
        assert counter == 0.5
        assert vuln == 0.5

    def test_rockets_vs_tanks(self):
        """Rockets should counter tanks well."""
        own = [{"type": "e3"}, {"type": "e3"}, {"type": "e3"}]
        enemy = [{"type": "3tnk"}, {"type": "4tnk"}]
        counter, vuln = compute_army_counter_score(own, enemy)
        assert counter > 0.5, "Rockets should be effective vs tanks"

    def test_rifles_vs_tanks(self):
        """Rifles should be vulnerable to tanks."""
        own = [{"type": "e1"}, {"type": "e1"}, {"type": "e1"}]
        enemy = [{"type": "3tnk"}, {"type": "4tnk"}]
        counter, vuln = compute_army_counter_score(own, enemy)
        assert vuln > 0.5, "Rifles should be vulnerable to tanks"

    def test_mixed_army_balance(self):
        """Mixed army should have moderate scores."""
        own = [{"type": "e1"}, {"type": "e3"}, {"type": "1tnk"}]
        enemy = [{"type": "e1"}, {"type": "3tnk"}]
        counter, vuln = compute_army_counter_score(own, enemy)
        # Mixed army should have some effectiveness and some vulnerability
        assert 0.0 <= counter <= 1.0
        assert 0.0 <= vuln <= 1.0

    def test_non_combat_units_excluded(self):
        """Non-combat units shouldn't count in composition."""
        own = [{"type": "harv"}, {"type": "mcv"}]
        enemy = [{"type": "3tnk"}]
        counter, vuln = compute_army_counter_score(own, enemy)
        # No combat units = can't counter anything
        assert counter == 0.0
        assert vuln == 1.0


class TestCosts:
    """Test cost lookups."""

    def test_unit_costs(self):
        assert get_unit_cost("e1") == 100
        assert get_unit_cost("4tnk") == 2000
        assert get_unit_cost("harv") == 1100

    def test_building_costs(self):
        assert get_building_cost("fact") == 2000
        assert get_building_cost("powr") == 300
        assert get_building_cost("tsla") == 1200

    def test_unknown_cost_is_zero(self):
        assert get_unit_cost("nonexistent") == 0
        assert get_building_cost("nonexistent") == 0

    def test_building_armor_types(self):
        assert get_building_armor("powr") == "wood"
        assert get_building_armor("tsla") == "heavy"
        assert get_building_armor("pbox") == "heavy"
