"""Single-scenario probe: prints progress so we can see exactly where the runner stalls."""
import sys, time, traceback
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/workspace/OpenDriveVLA')

print("[probe] loading config", flush=True)
t0 = time.time()
from carla_vla.scenarios.config import load, from_dict
cfg = from_dict(load('/root/autodl-tmp/workspace/OpenDriveVLA/carla_vla/scenarios/configs/scenario1_basic/s1_1_lane_keeping.yaml'))
print(f"[probe] config {cfg.scenario_id} loaded {time.time()-t0:.2f}s", flush=True)

print("[probe] importing runner", flush=True)
t0 = time.time()
from carla_vla.scenarios.scenario_runner import ScenarioRunner
print(f"[probe] runner import {time.time()-t0:.2f}s", flush=True)

print("[probe] constructing runner", flush=True)
t0 = time.time()
out = Path('/root/autodl-tmp/workspace/OpenDriveVLA/output/carla_generalization/_probe')
out.mkdir(parents=True, exist_ok=True)
r = ScenarioRunner(scenario=cfg, group='G1', output_dir=out)
print(f"[probe] runner constructed {time.time()-t0:.2f}s", flush=True)

print("[probe] _setup_world", flush=True)
t1 = time.time(); r._setup_world(); print(f"[probe] _setup_world {time.time()-t1:.2f}s", flush=True)
print("[probe] _setup_ego", flush=True)
t1 = time.time(); r._setup_ego(); print(f"[probe] _setup_ego {time.time()-t1:.2f}s", flush=True)
print("[probe] _setup_actors", flush=True)
t1 = time.time(); r._setup_actors(); print(f"[probe] _setup_actors {time.time()-t1:.2f}s", flush=True)
print("[probe] _setup_sensors", flush=True)
t1 = time.time(); r._setup_sensors(); print(f"[probe] _setup_sensors {time.time()-t1:.2f}s", flush=True)
print("[probe] _warmup_history", flush=True)
t1 = time.time(); r._warmup_history(); print(f"[probe] _warmup_history {time.time()-t1:.2f}s", flush=True)

print("[probe] episode loop: max 6 ticks", flush=True)
for i in range(6):
    t1 = time.time()
    r._tick()
    print(f"[probe]   _tick {i+1} took {time.time()-t1:.2f}s", flush=True)

print("[probe] _teardown", flush=True)
t1 = time.time(); r._teardown(); print(f"[probe] _teardown {time.time()-t1:.2f}s", flush=True)
print("[probe] DONE", flush=True)