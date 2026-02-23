"""Shared test fixtures for OpenRA-RL-Util tests."""

import pytest


@pytest.fixture
def empty_obs():
    """Minimal observation with all required fields."""
    return {
        "military": {
            "units_killed": 0, "units_lost": 0,
            "buildings_killed": 0, "buildings_lost": 0,
            "army_value": 0, "active_unit_count": 0,
            "kills_cost": 0, "deaths_cost": 0,
            "assets_value": 0, "experience": 0, "order_count": 0,
        },
        "economy": {
            "cash": 5000, "ore": 0,
            "power_provided": 100, "power_drained": 0,
            "resource_capacity": 2000, "harvester_count": 1,
        },
        "units": [],
        "buildings": [],
        "visible_enemies": [],
        "visible_enemy_buildings": [],
        "production_queues": [],
        "done": False,
        "result": "",
        "tick": 0,
    }


@pytest.fixture
def basic_obs():
    """Observation with a small base and army."""
    return {
        "military": {
            "units_killed": 0, "units_lost": 0,
            "buildings_killed": 0, "buildings_lost": 0,
            "army_value": 1500, "active_unit_count": 5,
            "kills_cost": 0, "deaths_cost": 0,
            "assets_value": 5000, "experience": 0, "order_count": 10,
        },
        "economy": {
            "cash": 3000, "ore": 500,
            "power_provided": 200, "power_drained": 100,
            "resource_capacity": 2000, "harvester_count": 2,
        },
        "units": [
            {"actor_id": 1, "type": "e1", "hp_percent": 1.0, "is_idle": False},
            {"actor_id": 2, "type": "e1", "hp_percent": 1.0, "is_idle": False},
            {"actor_id": 3, "type": "e3", "hp_percent": 1.0, "is_idle": True},
            {"actor_id": 4, "type": "1tnk", "hp_percent": 1.0, "is_idle": False},
            {"actor_id": 5, "type": "harv", "hp_percent": 1.0, "is_idle": False},
        ],
        "buildings": [
            {"actor_id": 100, "type": "fact", "hp_percent": 1.0, "is_producing": False},
            {"actor_id": 101, "type": "powr", "hp_percent": 1.0, "is_producing": False},
            {"actor_id": 102, "type": "tent", "hp_percent": 1.0, "is_producing": True},
            {"actor_id": 103, "type": "proc", "hp_percent": 1.0, "is_producing": False},
        ],
        "visible_enemies": [
            {"actor_id": 200, "type": "e1", "hp_percent": 1.0},
            {"actor_id": 201, "type": "3tnk", "hp_percent": 0.8},
        ],
        "visible_enemy_buildings": [
            {"actor_id": 300, "type": "fact", "hp_percent": 1.0},
            {"actor_id": 301, "type": "powr", "hp_percent": 1.0},
        ],
        "production_queues": [],
        "done": False,
        "result": "",
        "tick": 100,
    }


class MockObs:
    """Mock observation for rubric testing."""

    def __init__(self, result="", tick=0, kills_cost=0, deaths_cost=0,
                 assets_value=0, cash=0):
        self.result = result
        self.tick = tick
        self.done = result != ""
        self.military = type("M", (), {
            "kills_cost": kills_cost,
            "deaths_cost": deaths_cost,
            "assets_value": assets_value,
        })()
        self.economy = type("E", (), {"cash": cash})()
