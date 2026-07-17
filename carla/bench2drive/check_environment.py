"""Check the local Bench2Drive Python environment and CARLA connection."""

import os
import sys
from pathlib import Path


def main() -> None:
    configured_root = os.environ.get("BENCH2DRIVE_ROOT")
    if not configured_root:
        raise RuntimeError("Set BENCH2DRIVE_ROOT to the external Bench2Drive checkout")
    bench2drive_root = Path(configured_root).expanduser().resolve()
    if not bench2drive_root.is_dir():
        raise RuntimeError(f"BENCH2DRIVE_ROOT does not exist: {bench2drive_root}")
    sys.path[:0] = [
        str(bench2drive_root),
        str(bench2drive_root / "scenario_runner"),
        str(bench2drive_root / "leaderboard"),
    ]

    import carla
    import cv2
    import leaderboard
    import networkx
    import numpy
    import py_trees
    import shapely
    import srunner
    import xmlschema

    del leaderboard, networkx, py_trees, shapely, srunner, xmlschema

    host = os.environ.get("CARLA_HOST", "localhost")
    port = int(os.environ.get("CARLA_PORT", "2000"))
    client = carla.Client(host, port)
    client.set_timeout(20.0)

    print("imports: OK")
    print(f"python: {sys.version.split()[0]}")
    print(f"carla client: {client.get_client_version()}")
    print(f"carla server: {client.get_server_version()}")
    print(f"map: {client.get_world().get_map().name}")
    print(f"numpy: {numpy.__version__}")
    print(f"opencv: {cv2.__version__}")


if __name__ == "__main__":
    main()
