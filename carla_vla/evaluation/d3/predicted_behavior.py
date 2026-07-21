"""Classify predicted trajectory into one of the frozen predicted categories.

Pure geometry; does not depend on model output text. Uses the parsed
6-point ego-frame trajectory emitted by the model server.

Coordinate convention: x = lateral (negative=left, positive=right),
y = longitudinal forward.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List

from .contracts import (
    EXACT_ALL_ZERO_MAX_DISP_M, NEAR_ZERO_MAX_DISP_M,
    FORWARD_DISPLACEMENT_MIN_M, ACCEL_THRESHOLD,
    LANE_CHANGE_LATERAL_MIN_M, CURVATURE_TURN_MIN,
    STOP_PATH_LENGTH_MAX_M, PREDICTED_TRAJECTORY_VOCABULARY,
)


def _displacements(traj: List[List[float]]) -> Dict[str, float]:
    if not traj:
        return {"max_abs_x": 0.0, "max_abs_y": 0.0, "total_path": 0.0,
                "last_x": 0.0, "last_y": 0.0, "first_x": 0.0, "first_y": 0.0}
    xs = [float(p[0]) for p in traj]
    ys = [float(p[1]) for p in traj]
    total = 0.0
    for i in range(1, len(xs)):
        total += math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
    return {"max_abs_x": max(abs(x) for x in xs), "max_abs_y": max(abs(y) for y in ys),
              "total_path": total, "first_x": xs[0], "first_y": ys[0],
              "last_x": xs[-1], "last_y": ys[-1]}


def classify_predicted_trajectory(parsed_traj: List[List[float]]) -> str:
    """Return one of PREDICTED_TRAJECTORY_VOCABULARY."""
    if parsed_traj is None or len(parsed_traj) < 2:
        return "PREDICT_INVALID"
    d = _displacements(parsed_traj)
    max_disp = max(d["max_abs_x"], d["max_abs_y"])
    if max_disp <= EXACT_ALL_ZERO_MAX_DISP_M:
        return "PREDICT_STOP"
    if max_disp < NEAR_ZERO_MAX_DISP_M:
        return "PREDICT_DECELERATE"

    lx, ly = d["last_x"], d["last_y"]
    if abs(lx) < CURVATURE_TURN_MIN and abs(ly) < CURVATURE_TURN_MIN:
        return "PREDICT_UNKNOWN"

    # Dominant direction
    abs_x, abs_y = abs(lx), abs(ly)

    # Lateral dominance with longitudinal also meaningful => turn
    if abs_x > LANE_CHANGE_LATERAL_MIN_M and abs_y > FORWARD_DISPLACEMENT_MIN_M:
        if lx < 0:
            return "PREDICT_LEFT_TURN"
        return "PREDICT_RIGHT_TURN"

    # Lateral dominance with weak longitudinal => lane change
    if abs_x > LANE_CHANGE_LATERAL_MIN_M:
        if lx < 0:
            return "PREDICT_LANE_CHANGE_LEFT"
        return "PREDICT_LANE_CHANGE_RIGHT"

    # Forward dominant
    if abs_y >= FORWARD_DISPLACEMENT_MIN_M:
        if d["total_path"] > FORWARD_DISPLACEMENT_MIN_M * 1.5:
            return "PREDICT_ACCELERATE"
        return "PREDICT_FORWARD"

    return "PREDICT_UNKNOWN"