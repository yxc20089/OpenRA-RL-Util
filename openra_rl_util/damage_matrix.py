"""Unit effectiveness data derived from OpenRA Red Alert weapon definitions.

Maps each unit type to its weapon's effectiveness (Versus percentages) against
each armor type. Data extracted from OpenRA/mods/ra/weapons/*.yaml.

Armor types in RA:
  - none:     Infantry (unarmored)
  - light:    Light vehicles, helicopters, aircraft, submarines
  - heavy:    Tanks, heavy vehicles, destroyers, transports
  - wood:     Wooden structures (barracks, power plants, etc.)
  - concrete: Concrete structures (pillbox, tesla coil, etc.)

Versus percentage: 100 = normal damage, >100 = bonus, <100 = penalty.
Values are expressed as multipliers (1.0 = 100%).
"""

from typing import Optional

# ── Armor type for each unit ─────────────────────────────────────────────────

UNIT_ARMOR: dict[str, str] = {
    # Infantry (all "none" armor)
    "e1": "none", "e2": "none", "e3": "none", "e4": "none",
    "e6": "none", "e7": "none", "medi": "none", "mech": "none",
    "spy": "none", "thf": "none", "shok": "none", "dog": "none",
    # Vehicles
    "1tnk": "heavy", "2tnk": "heavy", "3tnk": "heavy", "4tnk": "heavy",
    "v2rl": "light", "jeep": "light", "apc": "heavy", "arty": "light",
    "harv": "heavy", "mcv": "heavy", "ftrk": "light", "mnly": "heavy",
    "ttnk": "heavy", "ctnk": "light", "stnk": "light", "qtnk": "heavy",
    "dtrk": "light", "mgg": "light", "mrj": "light", "truk": "light",
    # Aircraft
    "heli": "light", "hind": "light", "mh60": "light", "tran": "light",
    "yak": "light", "mig": "light",
    # Ships
    "ss": "light", "dd": "heavy", "ca": "heavy", "pt": "heavy",
    "lst": "heavy", "msub": "light",
}

# Armor for buildings (structures use "wood" or "concrete" in RA)
BUILDING_ARMOR: dict[str, str] = {
    "fact": "wood", "powr": "wood", "apwr": "wood",
    "barr": "wood", "tent": "wood", "proc": "wood",
    "weap": "wood", "dome": "wood", "fix": "wood",
    "atek": "wood", "stek": "wood", "hpad": "wood", "afld": "wood",
    "spen": "wood", "syrd": "wood", "silo": "wood", "kenn": "wood",
    # Defenses are concrete
    "pbox": "concrete", "hbox": "concrete", "gun": "concrete",
    "ftur": "concrete", "tsla": "concrete", "agun": "concrete",
    "sam": "concrete", "gap": "wood",
    # Superweapons
    "iron": "wood", "pdox": "wood", "mslo": "wood",
}

# ── Unit costs (duplicated from game_data for standalone use) ────────────────

UNIT_COST: dict[str, int] = {
    # Infantry
    "e1": 100, "e2": 160, "e3": 300, "e4": 300, "e6": 500,
    "e7": 1200, "medi": 200, "mech": 500, "spy": 500, "thf": 500,
    "shok": 350, "dog": 200,
    # Vehicles
    "1tnk": 700, "2tnk": 800, "3tnk": 950, "4tnk": 1700,
    "v2rl": 700, "jeep": 600, "apc": 800, "arty": 600,
    "harv": 1400, "mcv": 2500, "ftrk": 500, "mnly": 800,
    "ttnk": 1500, "ctnk": 1200, "stnk": 900, "qtnk": 2300,
    "dtrk": 1500, "mgg": 600, "mrj": 600, "truk": 800,
    # Aircraft
    "heli": 1200, "hind": 1200, "mh60": 1200, "tran": 900,
    "yak": 800, "mig": 2000,
    # Ships
    "ss": 950, "dd": 1000, "ca": 2000, "pt": 500, "lst": 700, "msub": 2000,
}

BUILDING_COST: dict[str, int] = {
    "fact": 2500, "powr": 300, "apwr": 500, "barr": 500, "tent": 500,
    "proc": 2000, "weap": 2000, "dome": 1000, "fix": 1200,
    "atek": 1500, "stek": 1500, "hpad": 1500, "afld": 1000,
    "spen": 650, "syrd": 650, "silo": 150, "kenn": 200,
    "pbox": 400, "hbox": 600, "gun": 600, "ftur": 600, "tsla": 1500,
    "agun": 600, "sam": 750, "gap": 500,
    "iron": 2800, "pdox": 2800, "mslo": 2500,
}

# ── Weapon effectiveness (Versus %) ─────────────────────────────────────────
#
# Each entry maps armor_type → multiplier (1.0 = 100% normal damage).
# Derived from OpenRA/mods/ra/weapons/*.yaml Versus: sections.
# Default (missing key) = 1.0 (full damage).
#
# Non-combat units have empty dict (cannot attack).

_DEFAULT_VS = {"none": 1.0, "light": 1.0, "heavy": 1.0, "wood": 1.0, "concrete": 1.0}

UNIT_EFFECTIVENESS: dict[str, dict[str, float]] = {
    # ── Infantry ──────────────────────────────────────────────────────────

    # e1 (Rifle): M1Carbine — LightMG with Wood: 30% override
    "e1": {"none": 1.5, "light": 0.4, "heavy": 0.1, "wood": 0.3, "concrete": 0.1},

    # e2 (Grenadier): Grenade — great vs buildings, weak vs armor
    "e2": {"none": 0.6, "light": 0.25, "heavy": 0.25, "wood": 1.0, "concrete": 1.0},

    # e3 (Rocket Soldier): Dragon — anti-armor missile
    "e3": {"none": 0.1, "light": 0.34, "heavy": 1.0, "wood": 0.74, "concrete": 0.5},

    # e4 (Flamethrower): Flamer — anti-infantry/structure, short range
    "e4": {"none": 0.9, "light": 0.5, "heavy": 0.25, "wood": 0.5, "concrete": 0.25},

    # e6 (Engineer): no attack
    "e6": {},

    # e7 (Tanya): Colt45 — insta-kill infantry, C4 buildings
    # Modeled as very effective vs infantry + structures, weak vs armor
    "e7": {"none": 10.0, "light": 0.1, "heavy": 0.1, "wood": 5.0, "concrete": 5.0},

    # medi (Medic): no attack
    "medi": {},

    # mech (Mechanic): no attack
    "mech": {},

    # spy: Pistol — minimal damage
    "spy": {"none": 0.1, "light": 0.01, "heavy": 0.01, "wood": 0.01, "concrete": 0.01},

    # thf (Thief): no attack
    "thf": {},

    # shok (Shock Trooper): PortaTesla — devastating vs infantry
    "shok": {"none": 10.0, "light": 1.0, "heavy": 0.6, "wood": 0.73, "concrete": 1.0},

    # dog: melee anti-infantry (insta-kill infantry only)
    "dog": {"none": 5.0, "light": 0.0, "heavy": 0.0, "wood": 0.0, "concrete": 0.0},

    # ── Vehicles ──────────────────────────────────────────────────────────

    # 1tnk (Light Tank): 25mm cannon — good vs light vehicles
    "1tnk": {"none": 0.32, "light": 1.16, "heavy": 0.48, "wood": 0.52, "concrete": 0.32},

    # 2tnk (Medium Tank): 90mm cannon — effective vs heavy armor
    "2tnk": {"none": 0.3, "light": 0.75, "heavy": 1.15, "wood": 0.75, "concrete": 0.5},

    # 3tnk (Heavy Tank): 105mm dual cannon — similar to medium but stronger
    "3tnk": {"none": 0.3, "light": 0.75, "heavy": 1.15, "wood": 0.75, "concrete": 0.5},

    # 4tnk (Mammoth): 120mm cannon + MammothTusk missiles — versatile
    # Cannon handles armor, missiles handle infantry/air
    "4tnk": {"none": 0.65, "light": 0.68, "heavy": 0.7, "wood": 0.75, "concrete": 0.5},

    # v2rl (V2 Rocket): SCUD — long range, good vs structures
    "v2rl": {"none": 0.9, "light": 0.7, "heavy": 0.4, "wood": 0.75, "concrete": 1.0},

    # jeep (Ranger): HeavyMG — anti-infantry, fast scout
    "jeep": {"none": 1.2, "light": 0.72, "heavy": 0.28, "wood": 0.6, "concrete": 0.28},

    # apc: HeavyMG — same weapons as jeep
    "apc": {"none": 1.2, "light": 0.72, "heavy": 0.28, "wood": 0.6, "concrete": 0.28},

    # arty (Artillery): 155mm — long range area, good vs infantry
    "arty": {"none": 0.6, "light": 0.6, "heavy": 0.25, "wood": 0.4, "concrete": 0.5},

    # harv (Ore Truck): no attack
    "harv": {},

    # mcv: no attack
    "mcv": {},

    # ftrk (Flak Truck): FLAK-23 — anti-air primary, weak ground
    "ftrk": {"none": 0.4, "light": 0.6, "heavy": 0.1, "wood": 0.1, "concrete": 0.2},

    # mnly (Minelayer): no direct attack (lays mines)
    "mnly": {},

    # ttnk (Tesla Tank): TTankZap — tesla weapon, strong vs all
    "ttnk": {"none": 10.0, "light": 1.0, "heavy": 1.0, "wood": 0.6, "concrete": 1.0},

    # ctnk (Chrono Tank): APTusk missiles — anti-ground
    "ctnk": {"none": 0.1, "light": 0.34, "heavy": 1.0, "wood": 0.74, "concrete": 0.5},

    # stnk (Phase Transport): APTusk.stnk — weaker missiles
    "stnk": {"none": 0.1, "light": 0.34, "heavy": 1.0, "wood": 0.74, "concrete": 0.5},

    # qtnk (MAD Tank): deploys seismic, no standard attack
    "qtnk": {},

    # dtrk (Demo Truck): suicide explosion, no standard attack
    "dtrk": {},

    # mgg (Mobile Gap Gen): no attack
    "mgg": {},

    # mrj (Radar Jammer): no attack
    "mrj": {},

    # truk (Supply Truck): no attack
    "truk": {},

    # ── Aircraft ──────────────────────────────────────────────────────────

    # heli (Longbow): HellfireAG missiles — anti-armor helicopter
    "heli": {"none": 0.3, "light": 0.9, "heavy": 1.0, "wood": 0.9, "concrete": 1.0},

    # hind: ChainGun — anti-infantry helicopter
    "hind": {"none": 1.44, "light": 0.72, "heavy": 0.28, "wood": 0.6, "concrete": 0.28},

    # mh60 (Black Hawk): HeavyMG
    "mh60": {"none": 1.2, "light": 0.72, "heavy": 0.28, "wood": 0.6, "concrete": 0.28},

    # tran (Chinook): no attack
    "tran": {},

    # yak: ChainGun.Yak — strafing anti-ground
    "yak": {"none": 1.0, "light": 0.6, "heavy": 0.25, "wood": 0.5, "concrete": 0.25},

    # mig: Maverick missiles — heavy anti-armor/structure
    "mig": {"none": 0.3, "light": 0.9, "heavy": 1.15, "wood": 0.9, "concrete": 1.0},

    # ── Ships ─────────────────────────────────────────────────────────────

    # ss (Submarine): TorpTube — anti-ship torpedoes
    "ss": {"none": 0.0, "light": 0.75, "heavy": 1.0, "wood": 0.75, "concrete": 5.0},

    # dd (Destroyer): 2Inch cannon + DepthCharge
    "dd": {"none": 0.28, "light": 0.72, "heavy": 1.0, "wood": 0.72, "concrete": 0.48},

    # ca (Cruiser): 8Inch long-range bombardment
    "ca": {"none": 0.6, "light": 0.6, "heavy": 0.25, "wood": 0.35, "concrete": 1.0},

    # pt (Gunboat): 2Inch cannon
    "pt": {"none": 0.28, "light": 0.72, "heavy": 1.0, "wood": 0.72, "concrete": 0.48},

    # lst (Transport): no attack
    "lst": {},

    # msub (Missile Sub): SubMissile — long-range cruise missiles
    "msub": {"none": 0.8, "light": 0.48, "heavy": 0.3, "wood": 0.5, "concrete": 1.0},
}

# Defense structures (for disruption dimension)
DEFENSE_EFFECTIVENESS: dict[str, dict[str, float]] = {
    "pbox": {"none": 1.5, "light": 0.3, "heavy": 0.1, "wood": 0.1, "concrete": 0.1},
    "hbox": {"none": 1.5, "light": 0.3, "heavy": 0.1, "wood": 0.1, "concrete": 0.1},
    "gun": {"none": 0.2, "light": 0.75, "heavy": 1.15, "wood": 0.5, "concrete": 0.5},
    "ftur": {"none": 0.9, "light": 0.5, "heavy": 0.25, "wood": 0.5, "concrete": 0.25},
    "tsla": {"none": 10.0, "light": 1.0, "heavy": 1.0, "wood": 0.6, "concrete": 1.0},
    "agun": {"none": 0.0, "light": 1.0, "heavy": 0.0, "wood": 0.0, "concrete": 0.0},
    "sam": {"none": 0.0, "light": 1.0, "heavy": 0.0, "wood": 0.0, "concrete": 0.0},
}

# ── Special unit roles ───────────────────────────────────────────────────────

# Units that are economic targets (killing them disrupts economy)
ECONOMIC_UNITS = {"harv", "truk"}

# Buildings that are economic infrastructure
ECONOMIC_BUILDINGS = {"proc", "silo"}

# Buildings that are production facilities
PRODUCTION_BUILDINGS = {"barr", "tent", "weap", "hpad", "afld", "spen", "syrd", "kenn"}

# Buildings that are tech enablers (killing regresses tech tree)
TECH_BUILDINGS = {"dome", "atek", "stek", "fix"}

# Buildings that provide power
POWER_BUILDINGS = {"powr", "apwr"}

# Non-combat units (cannot attack)
NON_COMBAT_UNITS = {
    utype for utype, vs in UNIT_EFFECTIVENESS.items() if not vs
}


# ── Query functions ──────────────────────────────────────────────────────────


def get_effectiveness(attacker_type: str, target_armor: str) -> float:
    """Get damage multiplier for an attacker against a target armor type.

    Args:
        attacker_type: Unit type (e.g., "e3", "1tnk").
        target_armor: Armor type (e.g., "none", "heavy").

    Returns:
        Damage multiplier (1.0 = normal, >1 = effective, <1 = weak).
        Returns 0.0 if unit cannot attack.
    """
    vs = UNIT_EFFECTIVENESS.get(attacker_type.lower(), {})
    if not vs:
        return 0.0
    return vs.get(target_armor.lower(), 1.0)


def get_unit_vs_unit(attacker: str, target: str) -> float:
    """Get effectiveness of attacker against a specific target unit.

    Args:
        attacker: Attacker unit type.
        target: Target unit type.

    Returns:
        Damage multiplier based on target's armor type.
    """
    target_armor = UNIT_ARMOR.get(target.lower(), "none")
    return get_effectiveness(attacker, target_armor)


def get_unit_armor(unit_type: str) -> str:
    """Get armor type for a unit. Returns 'none' if unknown."""
    return UNIT_ARMOR.get(unit_type.lower(), "none")


def get_building_armor(building_type: str) -> str:
    """Get armor type for a building. Returns 'wood' if unknown."""
    return BUILDING_ARMOR.get(building_type.lower(), "wood")


def get_unit_cost(unit_type: str) -> int:
    """Get cost of a unit. Returns 0 if unknown."""
    return UNIT_COST.get(unit_type.lower(), 0)


def get_building_cost(building_type: str) -> int:
    """Get cost of a building. Returns 0 if unknown."""
    return BUILDING_COST.get(building_type.lower(), 0)


def can_attack(unit_type: str) -> bool:
    """Check if a unit type can attack."""
    return bool(UNIT_EFFECTIVENESS.get(unit_type.lower(), {}))


def is_economic_target(unit_or_building: str) -> bool:
    """Check if destroying this target causes economic disruption."""
    t = unit_or_building.lower()
    return t in ECONOMIC_UNITS or t in ECONOMIC_BUILDINGS


def is_production_target(building_type: str) -> bool:
    """Check if destroying this building disrupts production."""
    return building_type.lower() in PRODUCTION_BUILDINGS


def is_tech_target(building_type: str) -> bool:
    """Check if destroying this building causes tech regression."""
    return building_type.lower() in TECH_BUILDINGS


def is_power_target(building_type: str) -> bool:
    """Check if destroying this building disrupts power supply."""
    return building_type.lower() in POWER_BUILDINGS


def compute_army_counter_score(
    own_units: list[dict],
    enemy_units: list[dict],
) -> tuple[float, float]:
    """Compute how well an army counters the enemy army.

    Args:
        own_units: List of dicts with at least 'type' key.
        enemy_units: List of dicts with at least 'type' key.

    Returns:
        (counter_score, vulnerability_score) both in [0, 1].
        counter_score: fraction of own combat units effective vs enemy.
        vulnerability_score: fraction of own combat units vulnerable to enemy.
    """
    if not own_units or not enemy_units:
        return 0.5, 0.5

    # Get enemy armor distribution
    enemy_armors: dict[str, int] = {}
    for u in enemy_units:
        armor = get_unit_armor(u.get("type", ""))
        enemy_armors[armor] = enemy_armors.get(armor, 0) + 1

    total_enemy = sum(enemy_armors.values())
    if total_enemy == 0:
        return 0.5, 0.5

    own_combat = [u for u in own_units if can_attack(u.get("type", ""))]
    if not own_combat:
        return 0.0, 1.0

    effective_count = 0
    vulnerable_count = 0

    for u in own_combat:
        utype = u.get("type", "")
        # Weighted average effectiveness against enemy composition
        avg_eff = 0.0
        for armor, count in enemy_armors.items():
            avg_eff += get_effectiveness(utype, armor) * (count / total_enemy)

        if avg_eff >= 1.0:
            effective_count += 1
        elif avg_eff < 0.5:
            vulnerable_count += 1

    n = len(own_combat)
    return effective_count / n, vulnerable_count / n
