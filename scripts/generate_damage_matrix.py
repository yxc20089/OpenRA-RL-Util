#!/usr/bin/env python3
"""Generate damage_matrix.py from OpenRA MiniYAML weapon/unit definitions.

Reads the actual YAML files from the OpenRA submodule and produces
the Python data module. This ensures the damage matrix always matches
the game's real values.

Usage:
    python scripts/generate_damage_matrix.py [--openra-path PATH]

Default OpenRA path: ../OpenRA-RL/OpenRA
"""

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Armor types we track
ARMOR_TYPES = ["none", "light", "heavy", "wood", "concrete"]

# ── MiniYAML Parser ────────────────────────────────────────────────────────


def parse_miniyaml(text: str) -> dict[str, Any]:
    """Parse OpenRA MiniYAML into nested dicts.

    MiniYAML uses tab indentation. Each node is key: value with children
    indented one tab deeper.
    """
    root: dict[str, Any] = OrderedDict()
    stack: list[tuple[int, dict]] = [(-1, root)]

    for line in text.splitlines():
        stripped = line.lstrip("\t")
        if not stripped or stripped.startswith("#"):
            continue

        depth = len(line) - len(stripped)

        # Pop stack to find parent
        while stack and stack[-1][0] >= depth:
            stack.pop()

        parent = stack[-1][1]

        # Parse key: value
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
        else:
            key = stripped.strip()
            value = ""

        # Create child dict
        child: dict[str, Any] = OrderedDict()
        child["__value__"] = value

        # Handle removal nodes (prefixed with -)
        if key.startswith("-"):
            parent[key] = child
        else:
            parent[key] = child

        stack.append((depth, child))

    return root


def resolve_inherits(definitions: dict[str, dict], name: str,
                     resolved_cache: dict[str, dict] | None = None,
                     resolving: set | None = None) -> dict:
    """Deep-merge a definition with its parent(s)."""
    if resolved_cache is None:
        resolved_cache = {}
    if resolving is None:
        resolving = set()

    if name in resolved_cache:
        return resolved_cache[name]

    if name not in definitions:
        return {}

    if name in resolving:
        return definitions.get(name, {})  # Break circular

    resolving.add(name)
    node = definitions[name]
    result = OrderedDict()

    # Collect all Inherits directives
    parents = []
    for key, val in node.items():
        if key.startswith("Inherits"):
            parent_name = val.get("__value__", "")
            if parent_name:
                parents.append(parent_name)

    # Merge parents first (in order)
    for parent_name in parents:
        parent_resolved = resolve_inherits(definitions, parent_name,
                                           resolved_cache, resolving)
        deep_merge(result, parent_resolved)

    # Merge own values on top
    deep_merge(result, node)

    # Handle removal nodes
    removals = [k for k in result if k.startswith("-")]
    for removal_key in removals:
        target = removal_key[1:]  # Strip the -
        result.pop(target, None)
        del result[removal_key]

    resolving.discard(name)
    resolved_cache[name] = result
    return result


def deep_copy_dict(d: dict) -> dict:
    """Deep copy a nested dict structure (OrderedDict-preserving)."""
    result = OrderedDict()
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = deep_copy_dict(v)
        else:
            result[k] = v
    return result


def deep_merge(target: dict, source: dict) -> None:
    """Deep merge source into target.

    IMPORTANT: Dict values from source are deep-copied to avoid aliasing
    the same objects across cached resolution results.
    """
    for key, val in source.items():
        if key.startswith("Inherits"):
            continue  # Skip Inherits directives
        if key in target and isinstance(target[key], dict) and isinstance(val, dict):
            deep_merge(target[key], val)
        else:
            # Deep copy dicts to prevent mutation of cached parent entries
            target[key] = deep_copy_dict(val) if isinstance(val, dict) else val


def get_value(node: dict, *keys: str) -> str:
    """Navigate nested dicts by key path, return __value__."""
    current = node
    for key in keys:
        if not isinstance(current, dict):
            return ""
        # Try exact match first
        if key in current:
            current = current[key]
        else:
            # Try case-insensitive
            found = False
            for k in current:
                if k.lower() == key.lower():
                    current = current[k]
                    found = True
                    break
            if not found:
                return ""
    if isinstance(current, dict):
        return current.get("__value__", "")
    return str(current)


def get_child(node: dict, *keys: str) -> dict:
    """Navigate nested dicts by key path, return child dict."""
    current = node
    for key in keys:
        if not isinstance(current, dict):
            return {}
        if key in current and isinstance(current[key], dict):
            current = current[key]
        else:
            # Case-insensitive fallback
            found = False
            for k in current:
                if k.lower() == key.lower() and isinstance(current[k], dict):
                    current = current[k]
                    found = True
                    break
            if not found:
                return {}
    return current


# ── Extraction Functions ───────────────────────────────────────────────────


def extract_versus(weapon_def: dict) -> dict[str, float]:
    """Extract Versus percentages from a resolved weapon definition.

    Searches all Warhead nodes for SpreadDamage with Versus blocks.
    Returns the primary damage warhead's Versus as multipliers.
    """
    versus = {}

    # Look for primary damage warhead (usually @1Dam or just the first SpreadDamage)
    for key, val in weapon_def.items():
        if not isinstance(val, dict):
            continue
        if not key.lower().startswith("warhead"):
            continue
        warhead_type = val.get("__value__", "")
        if "SpreadDamage" not in warhead_type and "TargetDamage" not in warhead_type:
            continue

        vs_node = get_child(val, "Versus")
        if not vs_node:
            continue

        # Extract each armor type
        for armor in ARMOR_TYPES:
            for k, v in vs_node.items():
                if k == "__value__":
                    continue
                if k.lower() == armor:
                    pct = v.get("__value__", "") if isinstance(v, dict) else str(v)
                    try:
                        versus[armor] = int(pct) / 100.0
                    except (ValueError, TypeError):
                        pass

        # Use the first/primary damage warhead
        if versus:
            break

    return versus


def extract_weapons(openra_path: Path) -> dict[str, dict]:
    """Parse all weapon YAML files and resolve inheritance."""
    weapon_dir = openra_path / "mods" / "ra" / "weapons"
    all_defs: dict[str, dict] = OrderedDict()

    for yaml_file in sorted(weapon_dir.glob("*.yaml")):
        text = yaml_file.read_text(encoding="utf-8")
        parsed = parse_miniyaml(text)
        for name, node in parsed.items():
            if isinstance(node, dict):
                all_defs[name] = node

    # Resolve inheritance
    resolved_cache: dict[str, dict] = {}
    resolved: dict[str, dict] = OrderedDict()
    for name in all_defs:
        resolved[name] = resolve_inherits(all_defs, name, resolved_cache)

    return resolved


def extract_units(openra_path: Path) -> dict[str, dict]:
    """Parse unit/vehicle/aircraft/ship YAML files and resolve inheritance."""
    rules_dir = openra_path / "mods" / "ra" / "rules"
    all_defs: dict[str, dict] = OrderedDict()

    # Also parse defaults for base types
    for yaml_file in ["defaults.yaml", "infantry.yaml", "vehicles.yaml",
                       "aircraft.yaml", "ships.yaml", "structures.yaml"]:
        fpath = rules_dir / yaml_file
        if fpath.exists():
            text = fpath.read_text(encoding="utf-8")
            parsed = parse_miniyaml(text)
            for name, node in parsed.items():
                if isinstance(node, dict):
                    all_defs[name] = node

    # Resolve inheritance
    resolved_cache: dict[str, dict] = {}
    resolved: dict[str, dict] = OrderedDict()
    for name in all_defs:
        resolved[name] = resolve_inherits(all_defs, name, resolved_cache)

    return resolved


def get_primary_weapon(unit_def: dict) -> str:
    """Get the primary weapon name from a unit definition."""
    # Check Armament@PRIMARY first
    arm_primary = get_child(unit_def, "Armament@PRIMARY")
    if arm_primary:
        return get_value(arm_primary, "Weapon")

    # Then plain Armament
    arm = get_child(unit_def, "Armament")
    if arm:
        return get_value(arm, "Weapon")

    # Check for Armament@AG (anti-ground, e.g., flak truck)
    arm_ag = get_child(unit_def, "Armament@AG")
    if arm_ag:
        return get_value(arm_ag, "Weapon")

    # Check Armament@SECONDARY for ground weapons
    arm_sec = get_child(unit_def, "Armament@SECONDARY")
    if arm_sec:
        weapon = get_value(arm_sec, "Weapon")
        if weapon:
            return weapon

    return ""


def get_secondary_weapon(unit_def: dict) -> str:
    """Get secondary weapon name (for dual-weapon units like 4TNK)."""
    arm = get_child(unit_def, "Armament@SECONDARY")
    if arm:
        return get_value(arm, "Weapon")
    return ""


def get_cost(unit_def: dict) -> int:
    """Extract unit cost."""
    cost_str = get_value(unit_def, "Valued", "Cost")
    try:
        return int(cost_str)
    except (ValueError, TypeError):
        return 0


def get_armor_type(unit_def: dict) -> str:
    """Extract armor type."""
    armor = get_value(unit_def, "Armor", "Type")
    return armor.lower() if armor else "none"


# ── Unit Lists ─────────────────────────────────────────────────────────────

# Units to include in the damage matrix (lowercase IDs)
INFANTRY = ["e1", "e2", "e3", "e4", "e6", "e7", "medi", "mech", "spy", "thf", "shok", "dog"]
VEHICLES = ["1tnk", "2tnk", "3tnk", "4tnk", "v2rl", "jeep", "apc", "arty",
            "harv", "mcv", "ftrk", "mnly", "ttnk", "ctnk", "stnk", "qtnk",
            "dtrk", "mgg", "mrj", "truk"]
AIRCRAFT = ["heli", "hind", "mh60", "tran", "yak", "mig"]
SHIPS = ["ss", "dd", "ca", "pt", "lst", "msub"]

BUILDINGS = ["fact", "powr", "apwr", "barr", "tent", "proc", "weap", "dome",
             "fix", "atek", "stek", "hpad", "afld", "spen", "syrd", "silo",
             "kenn", "pbox", "hbox", "gun", "ftur", "tsla", "agun", "sam",
             "gap", "iron", "pdox", "mslo"]

DEFENSES = ["pbox", "hbox", "gun", "ftur", "tsla", "agun", "sam"]

# Units that need special weapon modeling (targeting restrictions make
# raw Versus inaccurate for the agent's decision-making)
SPECIAL_MODELING = {
    # e7 (Tanya): Colt45 targets Infantry only + C4 demolishes buildings
    # Model as: devastating vs infantry, devastating vs buildings, useless vs armor
    "e7": {"none": 10.0, "light": 0.1, "heavy": 0.1, "wood": 5.0, "concrete": 5.0},
    # spy: SilencedPPK targets Infantry only, minimal damage
    "spy": {"none": 0.1, "light": 0.01, "heavy": 0.01, "wood": 0.01, "concrete": 0.01},
    # dog: DogJaw insta-kills infantry, can't attack anything else
    "dog": {"none": 5.0, "light": 0.0, "heavy": 0.0, "wood": 0.0, "concrete": 0.0},
    # ss: TorpTube only targets Water/Underwater units
    "ss": {"none": 0.0, "light": 0.75, "heavy": 1.0, "wood": 0.75, "concrete": 5.0},
}

# Units with dual weapons — average both for overall effectiveness
DUAL_WEAPON_UNITS = {"4tnk"}  # 120mm (vs armor) + MammothTusk (vs infantry/air)

# Units where @PRIMARY is AA-only — use @SECONDARY (ground weapon) for the matrix
# since ground combat determines most matchup effectiveness
GROUND_WEAPON_OVERRIDE = {
    "e3": "Dragon",       # PRIMARY=RedEye (AA), ground weapon=Dragon
    "heli": "HellfireAG", # PRIMARY=HellfireAA, ground weapon=HellfireAG
}

# Defense buildings with garrisoned weapons (no Armament in YAML)
DEFENSE_WEAPON_OVERRIDE = {
    "pbox": "M60mg",   # Allied pillbox — garrisoned infantry with M60mg
    "hbox": "M60mg",   # Camo pillbox — same garrisoned weapon
}


def fill_versus(vs: dict[str, float]) -> dict[str, float]:
    """Fill in all 5 armor types, defaulting missing ones to 1.0."""
    return {armor: vs.get(armor, 1.0) for armor in ARMOR_TYPES}


def build_versus_for_unit(unit_id: str, unit_def: dict,
                          weapons: dict[str, dict]) -> dict[str, float]:
    """Build the Versus effectiveness dict for a unit."""
    uid = unit_id.lower()

    # Special modeling overrides
    if uid in SPECIAL_MODELING:
        return SPECIAL_MODELING[uid]

    # Ground weapon override for units with AA-only primary
    if uid in GROUND_WEAPON_OVERRIDE:
        weapon_name = GROUND_WEAPON_OVERRIDE[uid]
    else:
        weapon_name = get_primary_weapon(unit_def)

    if not weapon_name:
        return {}  # Non-combat unit

    weapon_def = weapons.get(weapon_name, {})
    primary_vs = extract_versus(weapon_def)

    if not primary_vs:
        return {}  # Non-combat or no Versus data

    # Dual-weapon averaging
    if uid in DUAL_WEAPON_UNITS:
        sec_name = get_secondary_weapon(unit_def)
        if sec_name and sec_name in weapons:
            sec_vs = extract_versus(weapons[sec_name])
            if sec_vs:
                combined = {}
                all_armors = set(primary_vs.keys()) | set(sec_vs.keys())
                for armor in all_armors:
                    p = primary_vs.get(armor, 1.0)
                    s = sec_vs.get(armor, 1.0)
                    combined[armor] = round((p + s) / 2, 2)
                return fill_versus(combined)

    return fill_versus(primary_vs)


def build_versus_for_defense(building_id: str, unit_def: dict,
                              weapons: dict[str, dict]) -> dict[str, float]:
    """Build the Versus effectiveness dict for a defense structure."""
    bid = building_id.lower()

    # AA-only defenses: model as 0 vs non-air
    if bid in ("agun", "sam"):
        return {"none": 0.0, "light": 1.0, "heavy": 0.0, "wood": 0.0, "concrete": 0.0}

    # Defense weapon overrides for garrisoned buildings
    if bid in DEFENSE_WEAPON_OVERRIDE:
        weapon_name = DEFENSE_WEAPON_OVERRIDE[bid]
    else:
        weapon_name = get_primary_weapon(unit_def)
        if not weapon_name:
            garm = get_child(unit_def, "Armament@GARRISONED")
            if garm:
                weapon_name = get_value(garm, "Weapon")

    if not weapon_name:
        return {}

    weapon_def = weapons.get(weapon_name, {})
    vs = extract_versus(weapon_def)
    return fill_versus(vs) if vs else {}


# ── Code Generation ────────────────────────────────────────────────────────


def format_dict(d: dict[str, Any], indent: int = 1) -> str:
    """Format a dict as Python source with aligned values."""
    if not d:
        return "{}"
    items = []
    for k, v in d.items():
        if isinstance(v, float):
            items.append(f'"{k}": {v}')
        elif isinstance(v, int):
            items.append(f'"{k}": {v}')
        elif isinstance(v, str):
            items.append(f'"{k}": "{v}"')
    return "{" + ", ".join(items) + "}"


def generate_module(unit_armor: dict, building_armor: dict,
                    unit_cost: dict, building_cost: dict,
                    unit_effectiveness: dict, defense_effectiveness: dict) -> str:
    """Generate the damage_matrix.py source code."""
    lines = []
    lines.append('"""Unit effectiveness data derived from OpenRA Red Alert weapon definitions.')
    lines.append("")
    lines.append("Auto-generated by scripts/generate_damage_matrix.py from the OpenRA")
    lines.append("MiniYAML weapon and unit definitions. Do not edit manually — re-run")
    lines.append("the generator script instead.")
    lines.append("")
    lines.append("Armor types in RA:")
    lines.append("  - none:     Infantry (unarmored)")
    lines.append("  - light:    Light vehicles, helicopters, aircraft, submarines")
    lines.append("  - heavy:    Tanks, heavy vehicles, destroyers, transports")
    lines.append("  - wood:     Wooden structures (barracks, power plants, etc.)")
    lines.append("  - concrete: Concrete structures (walls)")
    lines.append("")
    lines.append("Versus percentage: 100 = normal damage, >100 = bonus, <100 = penalty.")
    lines.append("Values are expressed as multipliers (1.0 = 100%).")
    lines.append('"""')
    lines.append("")
    lines.append("from typing import Optional")
    lines.append("")

    # UNIT_ARMOR
    lines.append("# " + "─" * 76)
    lines.append("# Armor type for each unit (from Armor: Type: in rules YAML)")
    lines.append("")
    lines.append("UNIT_ARMOR: dict[str, str] = {")
    lines.append("    # Infantry (all \"none\" armor)")
    inf_items = [(k, v) for k, v in unit_armor.items() if k in INFANTRY]
    lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in inf_items) + ",")
    lines.append("    # Vehicles")
    veh_items = [(k, v) for k, v in unit_armor.items() if k in VEHICLES]
    for i in range(0, len(veh_items), 5):
        chunk = veh_items[i:i+5]
        lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in chunk) + ",")
    lines.append("    # Aircraft")
    air_items = [(k, v) for k, v in unit_armor.items() if k in AIRCRAFT]
    lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in air_items) + ",")
    lines.append("    # Ships")
    ship_items = [(k, v) for k, v in unit_armor.items() if k in SHIPS]
    lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in ship_items) + ",")
    lines.append("}")
    lines.append("")

    # BUILDING_ARMOR
    lines.append("# Armor for buildings")
    lines.append("BUILDING_ARMOR: dict[str, str] = {")
    regular = [(k, v) for k, v in building_armor.items() if k not in DEFENSES + ["gap", "iron", "pdox", "mslo"]]
    lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in regular) + ",")
    lines.append("    # Defenses")
    defense_items = [(k, v) for k, v in building_armor.items() if k in DEFENSES]
    lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in defense_items) + ",")
    misc = [(k, v) for k, v in building_armor.items() if k in ["gap", "iron", "pdox", "mslo"]]
    if misc:
        lines.append("    # Other")
        lines.append("    " + ", ".join(f'"{k}": "{v}"' for k, v in misc) + ",")
    lines.append("}")
    lines.append("")

    # UNIT_COST
    lines.append("# " + "─" * 76)
    lines.append("# Unit costs (from Valued: Cost: in rules YAML)")
    lines.append("")
    lines.append("UNIT_COST: dict[str, int] = {")
    lines.append("    # Infantry")
    lines.append("    " + ", ".join(f'"{k}": {unit_cost[k]}' for k in INFANTRY) + ",")
    lines.append("    # Vehicles")
    for i in range(0, len(VEHICLES), 5):
        chunk = VEHICLES[i:i+5]
        lines.append("    " + ", ".join(f'"{k}": {unit_cost[k]}' for k in chunk if k in unit_cost) + ",")
    lines.append("    # Aircraft")
    lines.append("    " + ", ".join(f'"{k}": {unit_cost[k]}' for k in AIRCRAFT) + ",")
    lines.append("    # Ships")
    lines.append("    " + ", ".join(f'"{k}": {unit_cost[k]}' for k in SHIPS) + ",")
    lines.append("}")
    lines.append("")

    # BUILDING_COST
    lines.append("BUILDING_COST: dict[str, int] = {")
    bcost_items = list(building_cost.items())
    for i in range(0, len(bcost_items), 5):
        chunk = bcost_items[i:i+5]
        lines.append("    " + ", ".join(f'"{k}": {v}' for k, v in chunk) + ",")
    lines.append("}")
    lines.append("")

    # UNIT_EFFECTIVENESS
    lines.append("# " + "─" * 76)
    lines.append("# Weapon effectiveness (Versus %)")
    lines.append("#")
    lines.append("# Each entry maps armor_type → multiplier (1.0 = 100% normal damage).")
    lines.append("# Derived from OpenRA/mods/ra/weapons/*.yaml Versus: sections.")
    lines.append("# Non-combat units have empty dict (cannot attack).")
    lines.append("")
    lines.append("UNIT_EFFECTIVENESS: dict[str, dict[str, float]] = {")

    def _format_vs(name: str, vs: dict, comment: str = "") -> str:
        cmt = f"  # {comment}" if comment else ""
        if not vs:
            return f'    "{name}": {{}},{cmt}'
        vs_str = ", ".join(f'"{k}": {v}' for k, v in vs.items())
        return f'    "{name}": {{{vs_str}}},{cmt}'

    for section_name, section_units in [("Infantry", INFANTRY), ("Vehicles", VEHICLES),
                                         ("Aircraft", AIRCRAFT), ("Ships", SHIPS)]:
        lines.append(f"    # ── {section_name} " + "─" * (60 - len(section_name)))
        for uid in section_units:
            vs = unit_effectiveness.get(uid, {})
            lines.append(_format_vs(uid, vs))
        lines.append("")

    lines.append("}")
    lines.append("")

    # DEFENSE_EFFECTIVENESS
    lines.append("# Defense structures (for disruption dimension)")
    lines.append("DEFENSE_EFFECTIVENESS: dict[str, dict[str, float]] = {")
    for did in DEFENSES:
        vs = defense_effectiveness.get(did, {})
        lines.append(_format_vs(did, vs))
    lines.append("}")
    lines.append("")

    # Special unit roles
    lines.append("# " + "─" * 76)
    lines.append("# Special unit roles")
    lines.append("")
    lines.append('ECONOMIC_UNITS = {"harv", "truk"}')
    lines.append('ECONOMIC_BUILDINGS = {"proc", "silo"}')
    lines.append('PRODUCTION_BUILDINGS = {"barr", "tent", "weap", "hpad", "afld", "spen", "syrd", "kenn"}')
    lines.append('TECH_BUILDINGS = {"dome", "atek", "stek", "fix"}')
    lines.append('POWER_BUILDINGS = {"powr", "apwr"}')
    lines.append("")
    lines.append("NON_COMBAT_UNITS = {")
    lines.append("    utype for utype, vs in UNIT_EFFECTIVENESS.items() if not vs")
    lines.append("}")
    lines.append("")

    # Query functions (unchanged)
    lines.append("")
    lines.append("# " + "─" * 76)
    lines.append("# Query functions")
    lines.append("")
    lines.append("")
    lines.append('def get_effectiveness(attacker_type: str, target_armor: str) -> float:')
    lines.append('    """Get damage multiplier for an attacker against a target armor type.')
    lines.append("")
    lines.append("    Args:")
    lines.append('        attacker_type: Unit type (e.g., "e3", "1tnk").')
    lines.append('        target_armor: Armor type (e.g., "none", "heavy").')
    lines.append("")
    lines.append("    Returns:")
    lines.append("        Damage multiplier (1.0 = normal, >1 = effective, <1 = weak).")
    lines.append("        Returns 0.0 if unit cannot attack.")
    lines.append('    """')
    lines.append('    vs = UNIT_EFFECTIVENESS.get(attacker_type.lower(), {})')
    lines.append("    if not vs:")
    lines.append("        return 0.0")
    lines.append('    return vs.get(target_armor.lower(), 1.0)')
    lines.append("")
    lines.append("")
    lines.append('def get_unit_vs_unit(attacker: str, target: str) -> float:')
    lines.append('    """Get effectiveness of attacker against a specific target unit."""')
    lines.append('    target_armor = UNIT_ARMOR.get(target.lower(), "none")')
    lines.append("    return get_effectiveness(attacker, target_armor)")
    lines.append("")
    lines.append("")
    lines.append('def get_unit_armor(unit_type: str) -> str:')
    lines.append('    """Get armor type for a unit. Returns \'none\' if unknown."""')
    lines.append('    return UNIT_ARMOR.get(unit_type.lower(), "none")')
    lines.append("")
    lines.append("")
    lines.append('def get_building_armor(building_type: str) -> str:')
    lines.append('    """Get armor type for a building. Returns \'wood\' if unknown."""')
    lines.append('    return BUILDING_ARMOR.get(building_type.lower(), "wood")')
    lines.append("")
    lines.append("")
    lines.append('def get_unit_cost(unit_type: str) -> int:')
    lines.append('    """Get cost of a unit. Returns 0 if unknown."""')
    lines.append("    return UNIT_COST.get(unit_type.lower(), 0)")
    lines.append("")
    lines.append("")
    lines.append('def get_building_cost(building_type: str) -> int:')
    lines.append('    """Get cost of a building. Returns 0 if unknown."""')
    lines.append("    return BUILDING_COST.get(building_type.lower(), 0)")
    lines.append("")
    lines.append("")
    lines.append('def can_attack(unit_type: str) -> bool:')
    lines.append('    """Check if a unit type can attack."""')
    lines.append("    return bool(UNIT_EFFECTIVENESS.get(unit_type.lower(), {}))")
    lines.append("")
    lines.append("")
    lines.append('def is_economic_target(unit_or_building: str) -> bool:')
    lines.append('    """Check if destroying this target causes economic disruption."""')
    lines.append("    t = unit_or_building.lower()")
    lines.append("    return t in ECONOMIC_UNITS or t in ECONOMIC_BUILDINGS")
    lines.append("")
    lines.append("")
    lines.append('def is_production_target(building_type: str) -> bool:')
    lines.append('    """Check if destroying this building disrupts production."""')
    lines.append("    return building_type.lower() in PRODUCTION_BUILDINGS")
    lines.append("")
    lines.append("")
    lines.append('def is_tech_target(building_type: str) -> bool:')
    lines.append('    """Check if destroying this building causes tech regression."""')
    lines.append("    return building_type.lower() in TECH_BUILDINGS")
    lines.append("")
    lines.append("")
    lines.append('def is_power_target(building_type: str) -> bool:')
    lines.append('    """Check if destroying this building disrupts power supply."""')
    lines.append("    return building_type.lower() in POWER_BUILDINGS")
    lines.append("")
    lines.append("")
    lines.append("def compute_army_counter_score(")
    lines.append("    own_units: list[dict],")
    lines.append("    enemy_units: list[dict],")
    lines.append(") -> tuple[float, float]:")
    lines.append('    """Compute how well an army counters the enemy army.')
    lines.append("")
    lines.append("    Args:")
    lines.append("        own_units: List of dicts with at least 'type' key.")
    lines.append("        enemy_units: List of dicts with at least 'type' key.")
    lines.append("")
    lines.append("    Returns:")
    lines.append("        (counter_score, vulnerability_score) both in [0, 1].")
    lines.append("        counter_score: fraction of own combat units effective vs enemy.")
    lines.append("        vulnerability_score: fraction of own combat units vulnerable to enemy.")
    lines.append('    """')
    lines.append("    if not own_units or not enemy_units:")
    lines.append("        return 0.5, 0.5")
    lines.append("")
    lines.append("    enemy_armors: dict[str, int] = {}")
    lines.append("    for u in enemy_units:")
    lines.append('        armor = get_unit_armor(u.get("type", ""))')
    lines.append("        enemy_armors[armor] = enemy_armors.get(armor, 0) + 1")
    lines.append("")
    lines.append("    total_enemy = sum(enemy_armors.values())")
    lines.append("    if total_enemy == 0:")
    lines.append("        return 0.5, 0.5")
    lines.append("")
    lines.append('    own_combat = [u for u in own_units if can_attack(u.get("type", ""))]')
    lines.append("    if not own_combat:")
    lines.append("        return 0.0, 1.0")
    lines.append("")
    lines.append("    effective_count = 0")
    lines.append("    vulnerable_count = 0")
    lines.append("")
    lines.append("    for u in own_combat:")
    lines.append('        utype = u.get("type", "")')
    lines.append("        avg_eff = 0.0")
    lines.append("        for armor, count in enemy_armors.items():")
    lines.append("            avg_eff += get_effectiveness(utype, armor) * (count / total_enemy)")
    lines.append("")
    lines.append("        if avg_eff >= 1.0:")
    lines.append("            effective_count += 1")
    lines.append("        elif avg_eff < 0.5:")
    lines.append("            vulnerable_count += 1")
    lines.append("")
    lines.append("    n = len(own_combat)")
    lines.append("    return effective_count / n, vulnerable_count / n")
    lines.append("")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate damage_matrix.py from OpenRA YAML")
    parser.add_argument("--openra-path", type=Path,
                        default=Path(__file__).parent.parent.parent / "OpenRA-RL" / "OpenRA",
                        help="Path to OpenRA repo root")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent.parent / "openra_rl_util" / "damage_matrix.py",
                        help="Output file path")
    parser.add_argument("--verify", action="store_true",
                        help="Verify mode: compare generated vs existing, don't write")
    args = parser.parse_args()

    if not (args.openra_path / "mods" / "ra" / "weapons").exists():
        print(f"ERROR: OpenRA path not found: {args.openra_path}")
        print("Use --openra-path to specify the OpenRA repository root")
        sys.exit(1)

    print(f"Reading weapons from {args.openra_path / 'mods' / 'ra' / 'weapons'}")
    weapons = extract_weapons(args.openra_path)
    print(f"  Found {len(weapons)} weapon definitions")

    print(f"Reading units from {args.openra_path / 'mods' / 'ra' / 'rules'}")
    units = extract_units(args.openra_path)
    print(f"  Found {len(units)} unit/building definitions")

    # Build data tables
    unit_armor: dict[str, str] = OrderedDict()
    unit_cost: dict[str, int] = OrderedDict()
    unit_effectiveness: dict[str, dict[str, float]] = OrderedDict()

    all_units = INFANTRY + VEHICLES + AIRCRAFT + SHIPS
    for uid in all_units:
        udef = units.get(uid.upper(), {})
        if not udef:
            print(f"  WARNING: unit {uid.upper()} not found in YAML")
            continue
        unit_armor[uid] = get_armor_type(udef)
        unit_cost[uid] = get_cost(udef)
        unit_effectiveness[uid] = build_versus_for_unit(uid, udef, weapons)

        # Debug output
        weapon = GROUND_WEAPON_OVERRIDE.get(uid, get_primary_weapon(udef))
        vs = unit_effectiveness[uid]
        if vs:
            vs_str = " ".join(f"{k}:{v}" for k, v in vs.items())
            print(f"  {uid:6s} → {weapon:20s} armor={unit_armor[uid]:6s} cost={unit_cost[uid]:5d}  {vs_str}")
        else:
            print(f"  {uid:6s} → {'(no attack)':20s} armor={unit_armor[uid]:6s} cost={unit_cost[uid]:5d}")

    building_armor: dict[str, str] = OrderedDict()
    building_cost: dict[str, int] = OrderedDict()
    defense_effectiveness: dict[str, dict[str, float]] = OrderedDict()

    for bid in BUILDINGS:
        bdef = units.get(bid.upper(), {})
        if not bdef:
            print(f"  WARNING: building {bid.upper()} not found in YAML")
            continue
        building_armor[bid] = get_armor_type(bdef)
        building_cost[bid] = get_cost(bdef)

    for did in DEFENSES:
        ddef = units.get(did.upper(), {})
        if ddef:
            defense_effectiveness[did] = build_versus_for_defense(did, ddef, weapons)
            weapon = DEFENSE_WEAPON_OVERRIDE.get(did, get_primary_weapon(ddef))
            if not weapon:
                garm = get_child(ddef, "Armament@GARRISONED")
                weapon = get_value(garm, "Weapon") if garm else "(garrisoned)"
            vs = defense_effectiveness[did]
            vs_str = " ".join(f"{k}:{v}" for k, v in vs.items()) if vs else "(none)"
            print(f"  {did:6s} → {weapon:20s} armor={building_armor.get(did, '?'):6s} cost={building_cost.get(did, 0):5d}  {vs_str}")

    # Generate output
    source = generate_module(unit_armor, building_armor, unit_cost, building_cost,
                             unit_effectiveness, defense_effectiveness)

    if args.verify:
        existing = args.output.read_text() if args.output.exists() else ""
        if existing.strip() == source.strip():
            print("\n✅ Generated data matches existing damage_matrix.py")
        else:
            print("\n❌ Generated data DIFFERS from existing damage_matrix.py")
            sys.exit(1)
    else:
        args.output.write_text(source)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
