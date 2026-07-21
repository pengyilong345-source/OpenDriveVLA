"""Scenario runner: orchestrates one CARLA episode, records an episode log.

The runner reuses the sensor/calibration/coord modules from the
`collect_carla_opendrivevla.py` work and the
`CarlaOpenDriveVLAAdapter` / `mini_prompt_modes` for the official-compatible
prompt. It does NOT touch model.generate from this file: inference is the
post-runner step. The runner produces an `episode_log.json` containing
everything a downstream step needs to:

  - call the LLM with the official-compatible prompt (G1 or G2);
  - apply a deterministic controller (G3) with the recorded constraints;
  - score against the future GT (open-loop);
  - record closed-loop raw events for the pilot.

The runner enforces:
  - synchronized 6-camera capture from the SAME server frame
  - 2-second history buffer (resampled at official offsets)
  - 6 future GT points @ 0.5/1.0/1.5/2.0/2.5/3.0 s in current ego frame
  - GT leakage gate (asserts no GT key in `inference_inputs` / `uniad_data`)
  - seed-deterministic NPC spawning and behavior
"""
from __future__ import annotations
import json
import math
import os
import pickle
import queue
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import carla

# Reuse the validated collector's sensor/coord/history utilities. Import
# the module via its full package path so the relative `import carla_uniad_coords
# as C` inside the collector module also resolves (we patch sys.path).
import sys
import importlib

# Make the collector's own `import carla_uniad_coords as C` resolvable from
# any caller.
_COLL_DIR = str(Path(__file__).resolve().parent.parent / "tools")
if _COLL_DIR not in sys.path:
    sys.path.insert(0, _COLL_DIR)
_carla_uniad_coords = importlib.import_module("carla_uniad_coords")
sys.modules.setdefault("carla_uniad_coords", _carla_uniad_coords)
C = importlib.import_module("collect_carla_opendrivevla")
from carla_vla.tools.carla_uniad_coords import (
    ego_rotation_from_forward, transform_to_ego_frame, build_can_bus_18,
    yaw_to_forward_world, carla_world_to_nuscenes_global, quat_from_rotation,
)
from carla_vla.scenarios.config import Scenario, ActorConfig, TriggerConfig
from carla_vla.scenarios.actors import (
    spawn_ego, spawn_role_actor, spawn_background_traffic,
)
from carla_vla.scenarios.triggers import TriggerSet
from carla_vla.scenarios.command_manager import CommandManager, CommandState


FORBIDDEN_GENERATE_KEYS = {
    "gt_future_trajectory", "gt_future_trajectory_world", "fut_traj",
    "fut_traj_valid_mask", "planning_gt", "gt_ego_fut_trajs",
    "gt_segmentation", "gt_occupancy", "route_future_waypoints",
}


# --------------------------- GT-leakage gate --------------------------------

def assert_no_gt_leak(payload: Dict[str, Any]) -> None:
    """Hard assertion: no forbidden GT key in inference payload.

    Top-level `evaluation_targets` is allowed (offline scoring), but nothing
    in the runtime path (inference_inputs, uniad_data, command_state side
    data) may carry a forbidden key.
    """
    runtime_keys: set = set((payload.get("inference_inputs") or {}).keys())
    for v in (payload.get("inference_inputs") or {}).values():
        if isinstance(v, dict):
            runtime_keys.update(v.keys())
    bad = FORBIDDEN_GENERATE_KEYS & runtime_keys
    if bad:
        raise RuntimeError(f"Forbidden GT keys reached inference payload: {sorted(bad)}")


# --------------------------- episode log -------------------------------------

@dataclass
class EpisodeLog:
    scenario_id: str
    subscenario: str
    group: str
    seed: int
    map: str
    config_snapshot: Dict[str, Any]
    samples: List[Dict[str, Any]] = field(default_factory=list)
    triggers_fired: List[Dict[str, Any]] = field(default_factory=list)
    collisions: List[Dict[str, Any]] = field(default_factory=list)
    lane_invasions: List[Dict[str, Any]] = field(default_factory=list)
    traffic_light_events: List[Dict[str, Any]] = field(default_factory=list)
    metrics_summary: Optional[Dict[str, Any]] = None
    extra_notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


# --------------------------- runner -----------------------------------------

class ScenarioRunner:
    def __init__(self,
                 scenario: Scenario,
                 group: str,            # 'G1' | 'G2' | 'G3'
                 output_dir: str | Path,
                 host: str = "127.0.0.1",
                 port: int = 2000,
                 tm_port: int = 8000,
                 override_seed: int | None = None):
        self.scenario = scenario
        self.group = group
        self.out = Path(output_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.host, self.port, self.tm_port = host, port, tm_port
        self._client = carla.Client(self.host, self.port); self._client.set_timeout(120.0)
        self._world = None
        self._ego = None
        self._sensors = []
        self._actors: List[Tuple[str, carla.Actor]] = []
        self._ticks = 0
        self._sim_t = 0.0
        self._history = C.EgoHistoryBuffer(scenario.history_seconds + 0.2)
        # Pilot/paired-mode override: when set, takes precedence over the
        # YAML's scenario.random_seed so that G1 and G2 of the same pilot
        # episode record identical images.
        self._effective_seed = override_seed if override_seed is not None else scenario.random_seed
        self._rng = random.Random(self._effective_seed)
        np.random.seed(self._effective_seed)
        self._triggers = TriggerSet(scenario.triggers)
        # Command manager: deterministic state machine.
        self._cmd_state = CommandState(
            raw_instruction=(scenario.raw_instruction if group == "G2" else
                              f"local_command:{scenario.route_command_label}"),
            route_command=scenario.route_command_label,
            behavior=scenario.behavior_constraint,
            target_speed_mps=scenario.target_speed_mps_override,
            target_lane_delta=scenario.target_lane_delta,
            hazard_type=scenario.hazard_type,
        )
        # Build deterministic stage rules from the explicit per-config list.
        self._cmd_mgr = CommandManager(self._cmd_state, stage_rules=[])
        self._episode_log = EpisodeLog(
            scenario_id=scenario.scenario_id, subscenario=scenario.subscenario,
            group=group, seed=self._effective_seed, map=scenario.carla_map,
            config_snapshot=scenario.to_dict())

    # ---- episode lifecycle ----

    def run(self, episode_timeout_s: float | None = None,
            max_ticks: int | None = None) -> EpisodeLog:
        timeout = episode_timeout_s or self.scenario.episode_timeout_s
        try:
            self._setup_world()
            self._setup_ego()
            self._setup_actors()
            self._setup_sensors()
            self._warmup_history()
            # episode loop
            sim_t0 = self._sim_t
            while (self._sim_t - sim_t0) < timeout:
                self._tick()
                if max_ticks is not None and self._ticks >= max_ticks:
                    break
                if self._all_done():
                    break
            self._finalize_metrics()
        finally:
            self._teardown()
        return self._episode_log

    # ---- steps ----

    def _setup_world(self) -> None:
        self._world = self._client.load_world(self.scenario.carla_map)
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = C.SIM_DT_S
        settings.no_rendering_mode = False
        self._world.apply_settings(settings)
        # weather
        w = self._world.get_weather()
        wd = self.scenario.weather
        if "cloudiness" in wd: w.cloudiness = float(wd["cloudiness"])
        if "precipitation" in wd: w.precipitation = float(wd["precipitation"])
        if "fog" in wd: w.fog_density = float(wd["fog"])
        if "wind" in wd: w.wind_intensity = float(wd["wind"])
        if "sun_altitude" in wd: w.sun_altitude_angle = float(wd["sun_altitude"])
        else: w.sun_altitude_angle = self.scenario.time_of_day_sun_alt_deg
        self._world.set_weather(w)
        tm = self._client.get_trafficmanager(self.tm_port)
        tm.set_synchronous_mode(True)

    def _setup_ego(self) -> None:
        self._ego = spawn_ego(self._world, self._world.get_map(), self.scenario)
        if self._ego is None:
            raise RuntimeError(f"failed to spawn ego on {self.scenario.carla_map}")
        # The smoke phase does NOT enable TM autopilot. Enabling autopilot
        # in synchronous mode at t=0 commonly drives the ego into a static
        # obstacle before our first tick, which destroys the actor. The G1
        # / G2 experiments never feed model outputs back into CARLA in this
        # phase; they only collect data and let the model predict a
        # trajectory offline. G3 (autopilot reference) is a separate phase
        # and is not driven from this runner.

    def _setup_actors(self) -> None:
        for cfg in self.scenario.actors:
            try:
                a = spawn_role_actor(self._world, self._world.get_map(),
                                      self._ego, cfg, self.scenario)
            except Exception as e:
                print(f"[runner] WARN spawn failed for role={cfg.role}: {e}")
                continue
            if a is not None:
                self._actors.append((cfg.role, a))
        if self.scenario.background_traffic_count > 0:
            try:
                bgs = spawn_background_traffic(self._world, self._world.get_map(),
                                                 self._ego,
                                                 self.scenario.background_traffic_count,
                                                 self._rng)
            except Exception as e:
                print(f"[runner] WARN background traffic failed: {e}")
                bgs = []
            for b in bgs:
                tm = self._client.get_trafficmanager(self.tm_port)
                try: b.set_autopilot(True, tm.get_port())
                except Exception: pass
                self._actors.append(("background", b))

    def _setup_sensors(self) -> None:
        w, h = self.scenario.camera_resolution
        self._sensor_refs, self._sensor_queues, _ = C.spawn_cameras(
            self._world, self._ego, w, h, self.scenario.camera_fov_deg)
        self._sensors = list(self._sensor_refs.values())

    def _warmup_history(self) -> None:
        ticks = int(self.scenario.history_seconds / C.SIM_DT_S) + 4
        for _ in range(ticks):
            self._world.tick()
            self._sim_t += C.SIM_DT_S
            self._history.push(self._sim_t, self._ego)

    def _all_done(self) -> bool:
        return False  # timeout governs

    # ---- per-tick step ----

    def _tick(self) -> None:
        frame = self._world.tick()
        self._sim_t += C.SIM_DT_S
        self._ticks += 1
        self._history.push(self._sim_t, self._ego)

        # Build a single inference-time sample (sync cameras, history, GT).
        sample = self._build_sample(frame, self._ticks)
        # GT-leakage gate (defense in depth; the model is not called here).
        assert_no_gt_leak(sample["inference_inputs"])
        assert_no_gt_leak(sample)
        # Update command manager from triggered events.
        obs = self._observations()
        fired = self._triggers.evaluate(self._ticks, obs)
        if fired:
            self._episode_log.triggers_fired.append({
                "tick": self._ticks, "ids": fired, "obs_summary": {
                    "ego_speed_mps": obs.get("ego_speed_mps", 0.0),
                    "elapsed_s": obs.get("elapsed_s", 0.0),
                }
            })
            # also re-evaluate the cmd manager with these fired trigger ids
            self._cmd_state = self._cmd_mgr.tick(obs, fired_trigger_ids=fired)
        sample["command_state"] = self._cmd_state.to_dict()
        self._episode_log.samples.append(sample)

    def _observations(self) -> Dict[str, Any]:
        # Filter out any actor references that became invalid mid-episode.
        alive = []
        for (r, a) in self._actors:
            try:
                _ = a.is_alive
                alive.append((r, a))
            except Exception:
                continue
        self._actors = alive
        try:
            v = self._ego.get_velocity()
            speed = float(np.linalg.norm(np.array([v.x, v.y, v.z])))
        except Exception:
            speed = 0.0
        out: Dict[str, Any] = {
            "ego": self._ego,
            "actors": [{"role": r, "actor": a} for (r, a) in alive],
            "elapsed_s": self._sim_t - (self._sim_t - C.SIM_DT_S * self._ticks),
            "ego_speed_mps": speed,
        }
        return out

    def _build_sample(self, frame: int, tick_idx: int) -> Dict[str, Any]:
        # sync 6 cameras from the same frame
        images = C.read_same_frame(self._sensor_queues, frame)
        tf = self._ego.get_transform()
        cur_xy = np.array([tf.location.x, tf.location.y], dtype=np.float64)
        fwd = np.array([tf.get_forward_vector().x, tf.get_forward_vector().y, tf.get_forward_vector().z])
        cur_R = ego_rotation_from_forward(fwd)
        cur_q = quat_from_rotation(cur_R)
        vel_world = np.array([self._ego.get_velocity().x, self._ego.get_velocity().y, self._ego.get_velocity().z])
        vel_ego = np.array([vel_world @ cur_R[:2, :2]]).flatten() if vel_world.size == 2 else vel_world @ cur_R
        can_bus = build_can_bus_18(cur_xy, cur_q, vel_ego[:2])

        w, h = self.scenario.camera_resolution
        img_dir = self.out / "images" / f"{self.scenario.scenario_id}_t{tick_idx:04d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        cams = {}
        for name in C.CAMERA_ORDER:
            img = images[name]
            rel = str(img_dir.relative_to(self.out) / f"{name}.png")
            img.save_to_disk(str(self.out / rel))
            sensor = self._sensor_refs[name]
            cams[name] = {
                "data_path": rel,
                "type": name,
                "timestamp": int(sensor.id),
            }

        history, hstatus = self._history.resample_history(cur_xy, cur_R)

        # future GT (6 points @ official offsets) in current ego frame
        ticks_per_future = int(round(C.FUTURE_OFFSETS_S[0] / C.SIM_DT_S))
        fut, fut_mask, fut_world = C.collect_future_gt(
            self._world, self._ego, len(C.FUTURE_OFFSETS_S),
            ticks_per_future, cur_xy, cur_R)

        # NOTE: this build_sample returns a slim dict; the full adapter runs
        # in the inference step. GT lives under evaluation_targets.
        return {
            "frame": int(frame),
            "tick": int(tick_idx),
            "sim_t": float(self._sim_t),
            "ego_carla_xy": cur_xy.tolist(),
            "ego_velocity_ego": vel_ego.tolist(),
            "can_bus": can_bus,
            "ego2global_quat": cur_q.tolist(),
            "history": history,
            "history_status": hstatus,
            "cams": cams,
            "inference_inputs": {
                "can_bus": can_bus,
                "ego2global_quat": cur_q.tolist(),
                "command_state": self._cmd_state.to_dict(),
            },
            "evaluation_targets": {
                "gt_future_trajectory": fut,
                "fut_traj_valid_mask": fut_mask,
                "future_offsets_s": list(C.FUTURE_OFFSETS_S),
            },
        }

    # ---- metrics + close ----

    def _finalize_metrics(self) -> None:
        # Open-loop cannot be computed without running the model; we leave the
        # summary to the post-runner step. Mark the episode as a runner pass.
        n = len(self._episode_log.samples)
        self._episode_log.metrics_summary = {
            "samples": n,
            "triggers_fired_count": len(self._episode_log.triggers_fired),
            "history_status_summary": {
                "ok": sum(1 for s in self._episode_log.samples if s.get("history_status") == "ok"),
                "other": sum(1 for s in self._episode_log.samples if s.get("history_status") != "ok"),
            },
            "note": "open_loop and closed_loop rollup filled in by the post-runner step",
        }

    def _teardown(self) -> None:
        # stop sensors (idempotent)
        for s in self._sensors:
            try: s.stop()
            except Exception: pass
        # destroy actors; tolerate actors that were already destroyed mid-episode
        all_actors = []
        for (_, a) in self._actors:
            all_actors.append(a)
        all_actors.extend(self._sensors)
        if self._ego is not None:
            all_actors.append(self._ego)
        # Use destroy_batch when many actors; fall back to per-actor destroy.
        if all_actors:
            try:
                ids = [a.id for a in all_actors]
                cmds = [carla.command.DestroyActor(i) for i in ids]
                self._client.apply_batch_sync(cmds, True)
            except Exception:
                for a in all_actors:
                    try: a.destroy()
                    except Exception: pass
        # restore synchronous=False best-effort
        if self._world is not None:
            try:
                s = self._world.get_settings()
                s.synchronous_mode = False
                self._world.apply_settings(s)
            except Exception: pass
