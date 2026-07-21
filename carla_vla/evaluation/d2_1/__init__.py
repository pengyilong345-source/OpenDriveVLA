"""D2.1 evaluator package — extends the frozen D2 evaluator with input
adapters for the D2.1 schema. Does NOT modify any frozen D0/D2 threshold
or success formula.
"""
from .evaluator import (
    evaluate_episode, evaluate_episode_success,
    evaluate_collision, evaluate_red_light, evaluate_stop_line,
    evaluate_solid_line, evaluate_wrong_way, evaluate_prolonged_wrong_lane,
    evaluate_instruction_stages, evaluate_stop_resume,
    evaluate_task_completion, evaluate_route_completion,
    load_episode_frames, _wilson_ci,
    TRAFFIC_LIGHT_SCENARIOS, LANE_KEEPING_SCENARIOS,
)

__all__ = [
    "evaluate_episode", "evaluate_episode_success",
    "evaluate_collision", "evaluate_red_light", "evaluate_stop_line",
    "evaluate_solid_line", "evaluate_wrong_way", "evaluate_prolonged_wrong_lane",
    "evaluate_instruction_stages", "evaluate_stop_resume",
    "evaluate_task_completion", "evaluate_route_completion",
    "load_episode_frames", "_wilson_ci",
    "TRAFFIC_LIGHT_SCENARIOS", "LANE_KEEPING_SCENARIOS",
]