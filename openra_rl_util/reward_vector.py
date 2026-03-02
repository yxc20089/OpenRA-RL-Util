"""Multi-dimensional reward vector for OpenRA-RL.

Instead of a single scalar reward, provides a 7+1 dimensional vector where
each dimension represents a distinct strategic skill:

  1. combat:         Cost-weighted damage exchange (winning fights efficiently)
  2. economy:        Economic growth and warfare (income, assets, harvesters)
  3. infrastructure: Base building, tech progression, production flow
  4. intelligence:   Scouting, fog removal, threat detection
  5. composition:    Force mix quality vs enemy army
  6. tempo:          Time-efficiency of actions (no idle units)
  7. disruption:     Strategic sabotage (power, production, tech regression)
  8. outcome:        Terminal win/loss signal (+1/-1, only on game end)

The reward vector (per-tick training signal) is intentionally separate from
the benchmark composite score (episode-level leaderboard ranking in rubrics.py).
The vector trains agent skills; the benchmark measures overall performance.

Usage:
    computer = RewardVectorComputer()
    computer.reset()

    for tick in game:
        obs = get_observation()
        vector = computer.compute(obs)
        # vector.combat, vector.economy, etc.
        scalar = vector.weighted_scalar(weights)  # for single-value-head RL
"""

from dataclasses import dataclass, field
from typing import Optional

from openra_rl_util.damage_matrix import (
    BUILDING_COST,
    ECONOMIC_BUILDINGS,
    ECONOMIC_UNITS,
    POWER_BUILDINGS,
    PRODUCTION_BUILDINGS,
    TECH_BUILDINGS,
    UNIT_COST,
    can_attack,
    compute_army_counter_score,
    get_building_armor,
    get_effectiveness,
    get_unit_armor,
    get_unit_cost,
)

# ── Default weights for collapsing vector to scalar ──────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "combat": 0.30,
    "economy": 0.15,
    "infrastructure": 0.10,
    "intelligence": 0.10,
    "composition": 0.10,
    "tempo": 0.10,
    "disruption": 0.15,
    "outcome": 1.00,
}

# ── Normalizers ──────────────────────────────────────────────────────────────

COMBAT_NORMALIZER = 5000.0      # ~cost of a medium tank engagement
ECONOMY_NORMALIZER = 10000.0    # ~cost of a refinery
HARVESTER_BONUS = 0.3           # Extra reward per enemy harvester killed
REFINERY_BONUS = 0.5            # Extra reward per enemy refinery killed
POWER_PLANT_BONUS = 0.2         # Extra reward per enemy power plant killed
PRODUCTION_BONUS = 0.2          # Extra reward per enemy production bldg killed
TECH_BONUS = 0.3                # Extra reward per enemy tech bldg killed
NEW_BUILDING_TYPE_REWARD = 0.2  # Reward per new building type completed
ENEMY_BASE_DISCOVERY_BONUS = 0.5
ENEMY_PRODUCTION_DISCOVERY_BONUS = 0.2
ENEMY_UNIT_SIGHTING_BONUS = 0.05


@dataclass
class RewardVector:
    """7+1 dimensional reward signal — one per agent skill."""

    combat: float = 0.0
    economy: float = 0.0
    infrastructure: float = 0.0
    intelligence: float = 0.0
    composition: float = 0.0
    tempo: float = 0.0
    disruption: float = 0.0
    outcome: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "combat": self.combat,
            "economy": self.economy,
            "infrastructure": self.infrastructure,
            "intelligence": self.intelligence,
            "composition": self.composition,
            "tempo": self.tempo,
            "disruption": self.disruption,
            "outcome": self.outcome,
        }

    def as_array(self) -> list[float]:
        return [
            self.combat, self.economy, self.infrastructure,
            self.intelligence, self.composition, self.tempo,
            self.disruption, self.outcome,
        ]

    def weighted_scalar(self, weights: Optional[dict[str, float]] = None) -> float:
        """Collapse to scalar for algorithms that need a single reward value."""
        w = weights or DEFAULT_WEIGHTS
        total = 0.0
        for dim, val in self.as_dict().items():
            total += val * w.get(dim, 0.0)
        return total


@dataclass
class RewardVectorState:
    """Tracks per-tick state for delta-based reward computation."""

    # Military deltas
    prev_kills_cost: int = 0
    prev_deaths_cost: int = 0
    prev_units_killed: int = 0
    prev_buildings_killed: int = 0
    prev_units_lost: int = 0
    prev_buildings_lost: int = 0

    # Economy deltas
    prev_cash: int = 0
    prev_ore: int = 0
    prev_assets_value: int = 0
    prev_harvester_count: int = 0

    # HP tracking for partial damage
    prev_own_unit_hp: dict[int, float] = field(default_factory=dict)
    prev_own_building_hp: dict[int, float] = field(default_factory=dict)
    prev_enemy_unit_hp: dict[int, float] = field(default_factory=dict)
    prev_enemy_building_hp: dict[int, float] = field(default_factory=dict)

    # Intelligence tracking
    prev_visible_cell_count: int = 0
    discovered_enemy_building_types: set[str] = field(default_factory=set)
    discovered_enemy_base: bool = False
    prev_visible_enemy_ids: set[int] = field(default_factory=set)

    # Infrastructure tracking
    own_building_types_built: set[str] = field(default_factory=set)
    prev_order_count: int = 0

    # Enemy state tracking (for disruption)
    prev_enemy_building_count: int = 0
    prev_enemy_power_buildings: int = 0
    prev_enemy_production_buildings: int = 0
    prev_enemy_tech_buildings: int = 0


class RewardVectorComputer:
    """Computes the multi-dimensional reward vector from game observations.

    Each call to compute() takes a full observation dict and returns a
    RewardVector with all 7+1 dimensions populated.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self._state = RewardVectorState()

    def reset(self) -> None:
        """Reset all tracking state for a new episode."""
        self._state = RewardVectorState()

    def compute(self, obs: dict) -> RewardVector:
        """Compute reward vector from an observation dictionary.

        Args:
            obs: Observation dict with keys: military, economy, units,
                 buildings, visible_enemies, visible_enemy_buildings,
                 production_queues, spatial_map_meta, done, result, tick.

        Returns:
            RewardVector with all dimensions populated.
        """
        military = obs.get("military", {})
        economy = obs.get("economy", {})
        units = obs.get("units", [])
        buildings = obs.get("buildings", [])
        visible_enemies = obs.get("visible_enemies", [])
        visible_enemy_buildings = obs.get("visible_enemy_buildings", [])
        production_queues = obs.get("production_queues", [])
        done = obs.get("done", False)
        result = obs.get("result", "")

        vector = RewardVector()

        vector.combat = self._compute_combat(military, units, buildings,
                                             visible_enemies, visible_enemy_buildings)
        vector.economy = self._compute_economy(economy, military,
                                               visible_enemies, visible_enemy_buildings)
        vector.infrastructure = self._compute_infrastructure(buildings, production_queues, economy)
        vector.intelligence = self._compute_intelligence(visible_enemies, visible_enemy_buildings)
        vector.composition = self._compute_composition(units, visible_enemies)
        vector.tempo = self._compute_tempo(units, military, production_queues)
        vector.disruption = self._compute_disruption(visible_enemy_buildings)

        if done:
            if result == "win":
                vector.outcome = 1.0
            elif result == "lose":
                vector.outcome = -1.0

        self._update_state(military, economy, units, buildings,
                           visible_enemies, visible_enemy_buildings)

        return vector

    # ── Dimension computations ───────────────────────────────────────────

    def _compute_combat(
        self,
        military: dict,
        own_units: list,
        own_buildings: list,
        enemies: list,
        enemy_buildings: list,
    ) -> float:
        """Cost-weighted damage exchange with partial damage tracking."""
        reward = 0.0

        # 1. Cost-weighted kills (from aggregate stats)
        kills_cost = military.get("kills_cost", 0)
        deaths_cost = military.get("deaths_cost", 0)
        kills_delta = kills_cost - self._state.prev_kills_cost
        deaths_delta = deaths_cost - self._state.prev_deaths_cost
        reward += (kills_delta - deaths_delta) / COMBAT_NORMALIZER

        # 2. Partial damage dealt to enemies (HP reduction without kill)
        for enemy in enemies:
            eid = enemy.get("actor_id", 0)
            hp = enemy.get("hp_percent", 1.0)
            etype = enemy.get("type", "")
            prev_hp = self._state.prev_enemy_unit_hp.get(eid)

            if prev_hp is not None and hp < prev_hp:
                cost = get_unit_cost(etype)
                damage_value = cost * (prev_hp - hp)
                reward += damage_value / COMBAT_NORMALIZER

        for eb in enemy_buildings:
            eid = eb.get("actor_id", 0)
            hp = eb.get("hp_percent", 1.0)
            etype = eb.get("type", "")
            prev_hp = self._state.prev_enemy_building_hp.get(eid)

            if prev_hp is not None and hp < prev_hp:
                cost = BUILDING_COST.get(etype.lower(), 500)
                damage_value = cost * (prev_hp - hp)
                reward += damage_value / COMBAT_NORMALIZER

        # 3. Partial damage taken (own units losing HP)
        for unit in own_units:
            uid = unit.get("actor_id", 0)
            hp = unit.get("hp_percent", 1.0)
            utype = unit.get("type", "")
            prev_hp = self._state.prev_own_unit_hp.get(uid)

            if prev_hp is not None and hp < prev_hp:
                cost = get_unit_cost(utype)
                damage_value = cost * (prev_hp - hp)
                reward -= damage_value / COMBAT_NORMALIZER

        for bldg in own_buildings:
            bid = bldg.get("actor_id", 0)
            hp = bldg.get("hp_percent", 1.0)
            btype = bldg.get("type", "")
            prev_hp = self._state.prev_own_building_hp.get(bid)

            if prev_hp is not None and hp < prev_hp:
                cost = BUILDING_COST.get(btype.lower(), 500)
                damage_value = cost * (prev_hp - hp)
                reward -= damage_value / COMBAT_NORMALIZER

        return max(-1.0, min(1.0, reward))

    def _compute_economy(
        self,
        economy: dict,
        military: dict,
        enemies: list,
        enemy_buildings: list,
    ) -> float:
        """Economic growth and warfare."""
        reward = 0.0

        # Own economic growth
        cash = economy.get("cash", 0)
        ore = economy.get("ore", 0)
        assets = military.get("assets_value", 0)

        prev_total = self._state.prev_cash + self._state.prev_ore + self._state.prev_assets_value
        curr_total = cash + ore + assets
        econ_delta = curr_total - prev_total

        # Investment-adjusted: reward spending on assets more than hoarding cash
        investment_ratio = assets / max(cash + assets, 1)
        reward += (econ_delta / ECONOMY_NORMALIZER) * (0.5 + 0.5 * investment_ratio)

        # Enemy harvester kills (detect by counting visible enemy harvesters)
        enemy_harv_count = sum(
            1 for e in enemies if e.get("type", "").lower() == "harv"
        )
        # We can detect kills via kills_cost delta + unit type
        # But simpler: check for killed enemy economic units via buildings_killed delta
        buildings_killed = military.get("buildings_killed", 0)
        prev_bk = self._state.prev_buildings_killed

        # Check enemy building types killed (approximate via visible enemy buildings)
        for eb in enemy_buildings:
            eid = eb.get("actor_id", 0)
            hp = eb.get("hp_percent", 1.0)
            etype = eb.get("type", "").lower()
            prev_hp = self._state.prev_enemy_building_hp.get(eid)

            # Building just destroyed (was visible, now at 0 or gone)
            if prev_hp is not None and prev_hp > 0 and hp <= 0:
                if etype in ECONOMIC_BUILDINGS:
                    reward += REFINERY_BONUS

        # Own harvester losses
        harv_count = economy.get("harvester_count", 0)
        if harv_count < self._state.prev_harvester_count:
            lost = self._state.prev_harvester_count - harv_count
            reward -= HARVESTER_BONUS * lost

        return max(-1.0, min(1.0, reward))

    def _compute_infrastructure(
        self,
        buildings: list,
        production_queues: list,
        economy: dict,
    ) -> float:
        """Base building, tech progression, production utilization."""
        # 1. New building types completed
        tech_reward = 0.0
        current_types = set()
        for b in buildings:
            btype = b.get("type", "").lower()
            if btype:
                current_types.add(btype)

        new_types = current_types - self._state.own_building_types_built
        tech_reward = len(new_types) * NEW_BUILDING_TYPE_REWARD

        # 2. Production utilization
        producing_count = 0
        total_production_buildings = 0
        for b in buildings:
            btype = b.get("type", "").lower()
            if btype in {"barr", "tent", "weap", "hpad", "afld", "spen", "syrd", "kenn"}:
                total_production_buildings += 1
                if b.get("is_producing", False):
                    producing_count += 1

        production_util = (
            producing_count / total_production_buildings
            if total_production_buildings > 0
            else 0.0
        )

        # 3. Power health
        power_provided = economy.get("power_provided", 0)
        power_drained = economy.get("power_drained", 0)
        surplus = power_provided - power_drained
        power_health = max(-1.0, min(1.0, surplus / 100.0))

        # Combine (average of three sub-signals)
        infra = (tech_reward + production_util + power_health) / 3.0

        return max(-1.0, min(1.0, infra))

    def _compute_intelligence(
        self,
        enemies: list,
        enemy_buildings: list,
    ) -> float:
        """Scouting, fog removal, threat detection."""
        reward = 0.0

        # 1. New enemy units spotted
        current_enemy_ids = set()
        for e in enemies:
            eid = e.get("actor_id", 0)
            if eid:
                current_enemy_ids.add(eid)

        new_sightings = current_enemy_ids - self._state.prev_visible_enemy_ids
        reward += len(new_sightings) * ENEMY_UNIT_SIGHTING_BONUS

        # 2. Enemy building type discovery (one-time per type)
        for eb in enemy_buildings:
            btype = eb.get("type", "").lower()
            if btype and btype not in self._state.discovered_enemy_building_types:
                self._state.discovered_enemy_building_types.add(btype)
                # First enemy production building = significant intel
                if btype in PRODUCTION_BUILDINGS:
                    reward += ENEMY_PRODUCTION_DISCOVERY_BONUS
                # Any building discovery = base location
                if not self._state.discovered_enemy_base:
                    self._state.discovered_enemy_base = True
                    reward += ENEMY_BASE_DISCOVERY_BONUS

        return max(-1.0, min(1.0, reward))

    def _compute_composition(
        self,
        own_units: list,
        enemies: list,
    ) -> float:
        """Force mix quality vs enemy army."""
        if not own_units or not enemies:
            return 0.0

        counter_score, vulnerability_score = compute_army_counter_score(
            own_units, enemies
        )

        # Net composition advantage: good counters minus vulnerabilities
        return max(-1.0, min(1.0, counter_score - vulnerability_score))

    def _compute_tempo(
        self,
        units: list,
        military: dict,
        production_queues: list,
    ) -> float:
        """Time-efficiency: idle unit penalty, action rate."""
        if not units:
            return 0.0

        # 1. Idle combat unit penalty
        combat_units = [u for u in units if can_attack(u.get("type", ""))]
        if combat_units:
            idle_combat = sum(1 for u in combat_units if u.get("is_idle", True))
            idle_ratio = idle_combat / len(combat_units)
            idle_penalty = idle_ratio * 0.1
        else:
            idle_penalty = 0.0

        # 2. Order rate (are we issuing commands?)
        order_count = military.get("order_count", 0)
        order_delta = order_count - self._state.prev_order_count
        order_rate = min(1.0, order_delta / 5.0)  # 5 orders/tick = max

        # Combine: rewarded for activity, penalized for idle combat units
        tempo = (order_rate * 0.05) - idle_penalty

        return max(-1.0, min(1.0, tempo))

    def _compute_disruption(
        self,
        enemy_buildings: list,
    ) -> float:
        """Strategic sabotage — breaking enemy capabilities."""
        reward = 0.0

        # Count enemy building types
        power_count = 0
        production_count = 0
        tech_count = 0
        total_count = 0

        for eb in enemy_buildings:
            btype = eb.get("type", "").lower()
            if not btype:
                continue
            total_count += 1
            if btype in POWER_BUILDINGS:
                power_count += 1
            if btype in PRODUCTION_BUILDINGS:
                production_count += 1
            if btype in TECH_BUILDINGS:
                tech_count += 1

        # Detect destroyed enemy buildings (count decreased)
        if self._state.prev_enemy_building_count > 0:
            # Power disruption
            if power_count < self._state.prev_enemy_power_buildings:
                lost = self._state.prev_enemy_power_buildings - power_count
                reward += lost * POWER_PLANT_BONUS

            # Production disruption
            if production_count < self._state.prev_enemy_production_buildings:
                lost = self._state.prev_enemy_production_buildings - production_count
                reward += lost * PRODUCTION_BONUS

            # Tech regression
            if tech_count < self._state.prev_enemy_tech_buildings:
                lost = self._state.prev_enemy_tech_buildings - tech_count
                reward += lost * TECH_BONUS

        return max(-1.0, min(1.0, reward))

    # ── State update ─────────────────────────────────────────────────────

    def _update_state(
        self,
        military: dict,
        economy: dict,
        own_units: list,
        own_buildings: list,
        enemies: list,
        enemy_buildings: list,
    ) -> None:
        """Update tracking state after computing rewards."""
        s = self._state

        # Military
        s.prev_kills_cost = military.get("kills_cost", 0)
        s.prev_deaths_cost = military.get("deaths_cost", 0)
        s.prev_units_killed = military.get("units_killed", 0)
        s.prev_buildings_killed = military.get("buildings_killed", 0)
        s.prev_units_lost = military.get("units_lost", 0)
        s.prev_buildings_lost = military.get("buildings_lost", 0)
        s.prev_order_count = military.get("order_count", 0)

        # Economy
        s.prev_cash = economy.get("cash", 0)
        s.prev_ore = economy.get("ore", 0)
        s.prev_assets_value = military.get("assets_value", 0)
        s.prev_harvester_count = economy.get("harvester_count", 0)

        # HP tracking
        s.prev_own_unit_hp = {
            u.get("actor_id", 0): u.get("hp_percent", 1.0)
            for u in own_units if u.get("actor_id")
        }
        s.prev_own_building_hp = {
            b.get("actor_id", 0): b.get("hp_percent", 1.0)
            for b in own_buildings if b.get("actor_id")
        }
        s.prev_enemy_unit_hp = {
            e.get("actor_id", 0): e.get("hp_percent", 1.0)
            for e in enemies if e.get("actor_id")
        }
        s.prev_enemy_building_hp = {
            eb.get("actor_id", 0): eb.get("hp_percent", 1.0)
            for eb in enemy_buildings if eb.get("actor_id")
        }

        # Intelligence
        s.prev_visible_enemy_ids = {
            e.get("actor_id", 0) for e in enemies if e.get("actor_id")
        }

        # Infrastructure
        for b in own_buildings:
            btype = b.get("type", "").lower()
            if btype:
                s.own_building_types_built.add(btype)

        # Enemy building tracking (for disruption)
        s.prev_enemy_building_count = len(enemy_buildings)
        s.prev_enemy_power_buildings = sum(
            1 for eb in enemy_buildings
            if eb.get("type", "").lower() in POWER_BUILDINGS
        )
        s.prev_enemy_production_buildings = sum(
            1 for eb in enemy_buildings
            if eb.get("type", "").lower() in PRODUCTION_BUILDINGS
        )
        s.prev_enemy_tech_buildings = sum(
            1 for eb in enemy_buildings
            if eb.get("type", "").lower() in TECH_BUILDINGS
        )
