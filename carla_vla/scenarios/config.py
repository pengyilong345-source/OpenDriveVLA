"""Scenario configuration schema + loaders.

Every scenario config is a YAML file with the keys the spec demands
(scenario_id, category, subscenario, map, route, weather, etc.).

YAML is the on-disk format; Python dicts are the runtime format. A
`Scenario` dataclass gives a typed view. The runner consumes `Scenario`
objects only.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- helpers --------------------------------------------------------

def _yaml_available() -> bool:
    try:
        import yaml  # noqa: F401
        return True
    except Exception:
        return False


def load(path: str | Path) -> Dict[str, Any]:
    """Load a YAML or JSON scenario config from disk. JSON is the fallback."""
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"} and _yaml_available():
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def save(cfg: Dict[str, Any], path: str | Path) -> None:
    """Persist a config dict to disk in YAML when available, else JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".yaml", ".yml"} and _yaml_available():
        import yaml
        p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    else:
        p.write_text(json.dumps(cfg, indent=2))


# ---------- typed view -----------------------------------------------------

@dataclass
class TriggerConfig:
    """When to fire a scenario event.

    `kind`: 'distance_to_ego' | 'ttc_below' | 'time_elapsed' | 'manual'
    `threshold_m` / `threshold_s` / `ttc_s` are used by the matching kind.
    `actor_role` names the actor that the trigger watches (optional).
    """
    kind: str
    threshold_m: Optional[float] = None
    threshold_s: Optional[float] = None
    ttc_s: Optional[float] = None
    actor_role: Optional[str] = None
    fire_once: bool = True
    note: str = ""


@dataclass
class ActorConfig:
    """A scene actor (NPC vehicle, walker, bus, bicycle...)."""
    role: str                     # 'ego' | 'lead' | 'ped_crossing' | ...
    actor_type: str               # carla.ActorTypeId; 'vehicle.tesla.model3', etc.
    spawn_mode: str = "role"      # 'role' | 'relative' | 'waypoint_at' | 'manual_xy'
    relative_to: Optional[str] = None
    offset_xy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_speed_mps: Optional[float] = None
    traffic_light_actor: bool = False
    autopiloted: bool = True
    initial_pose_xy: Optional[Tuple[float, float, float]] = None
    initial_yaw_deg: Optional[float] = None
    initial_speed_mps: Optional[float] = None
    role_args: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class Scenario:
    scenario_id: str
    category: str
    subscenario: str
    carla_map: str
    route: Dict[str, Any]               # {"spawn_point_index": int, "path_length_m": float}
    weather: Dict[str, Any]             # {cloudiness, precipitation, sun_altitude_angle, ...}
    time_of_day_sun_alt_deg: float
    ego_initial_speed_mps: float
    ego_target_speed_mps: Optional[float]
    background_traffic_count: int
    pedestrian_count: int
    bicycle_count: int = 0
    bus_count: int = 0
    triggers: List[TriggerConfig] = field(default_factory=list)
    actors: List[ActorConfig] = field(default_factory=list)
    raw_instruction: str = ""           # G2 raw NL instruction
    route_command_label: str = "FORWARD"# official mini adapter label
    behavior_constraint: str = "none"   # 'yield' | 'overtake' | 'bus_stop_pass' | 'slow' | 'none'
    target_speed_mps_override: Optional[float] = None
    target_lane_delta: int = 0
    hazard_type: str = "none"           # 'pedestrian_crossing' | 'slow_vehicle' | 'cut_in' | 'cones' | 'none'
    episode_timeout_s: float = 60.0
    success_conditions: List[str] = field(default_factory=list)
    failure_conditions: List[str] = field(default_factory=list)
    physically_avoidable: bool = True
    random_seed: int = 0
    camera_resolution: Tuple[int, int] = (1600, 900)
    camera_fov_deg: float = 70.0
    history_seconds: float = 2.5
    note: str = ""
    # ---- Stage D0 acceptance-protocol overrides ----
    # These fields are additively backward-compatible. Existing scenarios
    # load fine without setting them; they take the protocol defaults.
    acceptance_overrides: Dict[str, Any] = field(default_factory=dict)

    # -- dict compatibility --
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- construction ---------------------------------------------------

def from_dict(d: Dict[str, Any]) -> Scenario:
    trigs = [TriggerConfig(**t) for t in d.get("triggers", [])]
    actors = []
    for a in d.get("actors", []):
        a = dict(a)
        # tuple fields
        if "offset_xy" in a and not isinstance(a["offset_xy"], tuple):
            a["offset_xy"] = tuple(a["offset_xy"])
        if "initial_pose_xy" in a and a["initial_pose_xy"] is not None \
                and not isinstance(a["initial_pose_xy"], tuple):
            a["initial_pose_xy"] = tuple(a["initial_pose_xy"])
        actors.append(ActorConfig(**a))
    cam = d.get("camera_resolution", [1600, 900])
    return Scenario(
        scenario_id=d["scenario_id"],
        category=d["category"],
        subscenario=d["subscenario"],
        carla_map=d["carla_map"],
        route=d.get("route", {}),
        weather=d.get("weather", {}),
        time_of_day_sun_alt_deg=float(d.get("time_of_day_sun_alt_deg", 60.0)),
        ego_initial_speed_mps=float(d.get("ego_initial_speed_mps", 0.0)),
        ego_target_speed_mps=d.get("ego_target_speed_mps"),
        background_traffic_count=int(d.get("background_traffic_count", 0)),
        pedestrian_count=int(d.get("pedestrian_count", 0)),
        bicycle_count=int(d.get("bicycle_count", 0)),
        bus_count=int(d.get("bus_count", 0)),
        triggers=trigs,
        actors=actors,
        raw_instruction=d.get("raw_instruction", ""),
        route_command_label=d.get("route_command_label", "FORWARD"),
        behavior_constraint=d.get("behavior_constraint", "none"),
        target_speed_mps_override=d.get("target_speed_mps_override"),
        target_lane_delta=int(d.get("target_lane_delta", 0)),
        hazard_type=d.get("hazard_type", "none"),
        episode_timeout_s=float(d.get("episode_timeout_s", 60.0)),
        success_conditions=list(d.get("success_conditions", [])),
        failure_conditions=list(d.get("failure_conditions", [])),
        physically_avoidable=bool(d.get("physically_avoidable", True)),
        random_seed=int(d.get("random_seed", 0)),
        camera_resolution=tuple(cam) if not isinstance(cam, tuple) else cam,
        camera_fov_deg=float(d.get("camera_fov_deg", 70.0)),
        history_seconds=float(d.get("history_seconds", 2.5)),
        note=d.get("note", ""),
        acceptance_overrides=dict(d.get("acceptance_overrides") or {}),
    )


def list_configs(root: str | Path) -> List[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*.yaml")) + sorted(p for p in root.rglob("*.json"))


def all_scenarios(root: str | Path) -> List[Scenario]:
    out: List[Scenario] = []
    for p in list_configs(root):
        try:
            out.append(from_dict(load(p)))
        except Exception as e:
            print(f"[scenarios] WARN: failed to load {p}: {e}")
    return out
