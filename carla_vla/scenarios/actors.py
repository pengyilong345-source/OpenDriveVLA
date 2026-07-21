"""CARLA actor spawning helpers used by the scenario runner.

The runner calls these to spawn ego, NPC vehicles, walkers, buses, and
bicycles. Every actor gets a stable `role` string so the triggers and
metrics can identify them across ticks.
"""
from __future__ import annotations
import math
import random
import carla

from .config import Scenario, ActorConfig


# ---- role-based default blueprints ------------------------------------------

VEHICLE_BP_FILTERS = {
    "car":  "vehicle.tesla.model3",
    "truck": "vehicle.carlamotors.firetruck",
    "bus":  "vehicle.mitsubishi.fusorosa",
    "bike": "vehicle.bh.crossbike",
}

WALKER_BP_FILTER = "walker.pedestrian.*"


def _bp(world, pattern: str) -> carla.ActorBlueprint:
    bl = world.get_blueprint_library()
    choices = list(bl.filter(pattern))
    if not choices:
        raise RuntimeError(f"no blueprint matches '{pattern}'")
    return random.choice(choices)


def _safe_spawn(world, blueprint, transform) -> carla.Actor | None:
    return world.try_spawn_actor(blueprint, transform)


def spawn_ego(world, carla_map, scenario: Scenario) -> carla.Actor:
    """Spawn the ego at the configured route spawn point."""
    sps = carla_map.get_spawn_points()
    idx = int(scenario.route.get("spawn_point_index", 0))
    sp = sps[max(0, min(idx, len(sps) - 1))]
    ego_bp = _bp(world, "vehicle.tesla.model3")
    ego = _safe_spawn(world, ego_bp, sp)
    if ego is None:
        raise RuntimeError("failed to spawn ego at scenario route")
    if scenario.ego_initial_speed_mps > 0:
        v = sp.get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(
            v.x * scenario.ego_initial_speed_mps,
            v.y * scenario.ego_initial_speed_mps,
            v.z * scenario.ego_initial_speed_mps))
    return ego


def _spawn_at_offset_actor(world, blueprint, anchor: carla.Transform,
                            forward_m: float, right_m: float,
                            yaw_offset_deg: float = 0.0) -> carla.Actor | None:
    yaw = math.radians(anchor.rotation.yaw)
    dx, dy = math.cos(yaw), math.sin(yaw)
    lx, ly = -math.sin(yaw), math.cos(yaw)
    loc = anchor.location + carla.Location(
        x=anchor.location.x + dx * forward_m + lx * right_m,
        y=anchor.location.y + dy * forward_m + ly * right_m,
        z=anchor.location.z)
    rot = carla.Rotation(yaw=anchor.rotation.yaw + yaw_offset_deg)
    return _safe_spawn(world, blueprint, carla.Transform(loc, rot))


def spawn_role_actor(world, carla_map, ego: carla.Actor,
                      cfg: ActorConfig, scenario: Scenario) -> carla.Actor | None:
    """Spawn one actor defined by an ActorConfig."""
    anchor = ego.get_transform()
    # Conceptual markers (cones, abstract hazards) have actor_type=='none' and
    # no CARLA blueprint to spawn. They are recorded in the actor list of the
    # Scenario but never produce a CARLA Actor. The runner treats them as
    # events without physics.
    if cfg.actor_type in (None, "none", ""):
        return None
    if cfg.actor_type == "walker":
        bp = _bp(world, WALKER_BP_FILTER)
    else:
        pattern = VEHICLE_BP_FILTERS.get(cfg.actor_type, cfg.actor_type)
        bp = _bp(world, pattern)

    # spawn placement
    if cfg.spawn_mode == "relative" and cfg.relative_to == "ego":
        off = cfg.offset_xy
        a = _spawn_at_offset_actor(world, bp, anchor, off[0], off[1], off[2] or 0.0)
    elif cfg.spawn_mode == "waypoint_at" and cfg.relative_to == "ego":
        fwd = cfg.offset_xy[0]
        wp = carla_map.get_waypoint(anchor.location, project_to_road=True)
        cur = wp
        for _ in range(int(fwd)):
            nxt = cur.next(1.0)
            if not nxt: break
            cur = nxt[0]
        a = _safe_spawn(world, bp, carla.Transform(cur.transform.location,
                                                    cur.transform.rotation))
    else:  # 'role' / 'manual_xy' fall back to current ego location
        a = _safe_spawn(world, bp, anchor)

    if a is None:
        return None
    if cfg.initial_speed_mps is not None and cfg.initial_speed_mps > 0:
        v = a.get_transform().get_forward_vector()
        a.set_target_velocity(carla.Vector3D(
            v.x * cfg.initial_speed_mps, v.y * cfg.initial_speed_mps, v.z))
    if cfg.target_speed_mps is not None:
        # store on attributes for controllers to pick up
        a.attributes["scenario_target_speed_mps"] = str(cfg.target_speed_mps)
    return a


# ---- convenience for background traffic -------------------------------------

def spawn_background_traffic(world, carla_map, ego, n: int, rng) -> list[carla.Actor]:
    """Spawn `n` background cars in free spawn points (excluding ego radius)."""
    out = []
    sps = carla_map.get_spawn_points(); rng.shuffle(sps)
    ego_loc = ego.get_location()
    for sp in sps:
        if len(out) >= n: break
        if sp.location.distance(ego_loc) < 15.0: continue
        bp = _bp(world, VEHICLE_BP_FILTERS["car"])
        a = _safe_spawn(world, bp, sp)
        if a is not None:
            out.append(a)
    return out
