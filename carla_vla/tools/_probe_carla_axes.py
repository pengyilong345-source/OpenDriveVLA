"""Live probe: confirm CARLA axis conventions (forward/lateral/up, velocity,
angular_velocity, control fields) so the collector's right-handed conversion is
grounded in measurement, not assumption.

Run in the carla37 env against a running CARLA server.
"""
import sys, math, json
import carla

HOST, PORT = "127.0.0.1", 2000

def main():
    client = carla.Client(HOST, PORT); client.set_timeout(60.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    spawn = world.get_map().get_spawn_points()[0]
    ego = world.try_spawn_actor(bp, spawn)
    if ego is None:
        print(json.dumps({"error": "spawn failed"})); return
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    ego.set_autopilot(True, tm.get_port())

    # Step forward a bit and let it accelerate.
    for _ in range(80):
        world.tick()

    tf = ego.get_transform()
    fwd = tf.get_forward_vector()
    loc = tf.location
    rot = tf.rotation
    M = tf.get_matrix()
    Minv = tf.get_inverse_matrix()
    vel = ego.get_velocity()
    acc = ego.get_acceleration()
    ang = ego.get_angular_velocity()
    try:
        ctrl = ego.get_control()
        ctrl_out = {"throttle": ctrl.throttle, "steer": ctrl.steer, "brake": ctrl.brake}
    except RuntimeError:
        ctrl_out = {"throttle": None, "steer": None, "brake": None}

    def v2d(v): return [v.x, v.y, v.z]
    out = {
        "location": v2d(loc),
        "rotation_yaw_pitch_roll": [rot.yaw, rot.pitch, rot.roll],
        "forward_vector": v2d(fwd),
        "velocity": v2d(vel),
        "speed_magnitude": math.sqrt(vel.x**2+vel.y**2+vel.z**2),
        "acceleration": v2d(acc),
        "angular_velocity_deg_s": v2d(ang),
        "control": ctrl_out,
        "matrix_4x4": [[M[r][c] for c in range(4)] for r in range(4)],
        "inverse_4x4": [[Minv[r][c] for c in range(4)] for r in range(4)],
    }
    print(json.dumps(out, indent=2))

    try:
        ego.destroy()
    except RuntimeError:
        pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
