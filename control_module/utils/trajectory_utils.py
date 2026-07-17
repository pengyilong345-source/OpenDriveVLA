import math
import json
from typing import List, Dict, Tuple, Optional

def interpolate_trajectory(points: List[Dict], step: float = 0.1) -> List[Dict]:
    """
    对轨迹点进行线性插值，生成密集点。
    如果点中包含 'dt'，则使用 dt 作为时间；否则默认从 0.5 开始，步长 0.5。
    """
    if not points or len(points) < 2:
        return []

    # 检查是否有 dt 字段
    has_dt = 'dt' in points[0]
    dense_points = []
    dense_points.append({"x": 0.0, "y": 0.0})  # 起点

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        if has_dt:
            dt_start = start['dt']
            dt_end = end['dt']
        else:
            # 假设每个点间隔 0.5 秒
            dt_start = (i + 1) * 0.5
            dt_end = (i + 2) * 0.5

        dt_diff = dt_end - dt_start
        if dt_diff <= 0:
            continue
        num_steps = max(1, int(dt_diff / step))

        for j in range(1, num_steps + 1):
            ratio = j / num_steps
            interp_x = start["x"] + (end["x"] - start["x"]) * ratio
            interp_y = start["y"] + (end["y"] - start["y"]) * ratio
            dense_points.append({"x": round(interp_x, 3), "y": round(interp_y, 3)})

    return dense_points


def get_lookahead_point(
    trajectory: List[Dict], 
    current_speed: float, 
    lookahead_time: float = 1.2,
    min_lookahead_distance: float = 1.0
) -> Optional[Tuple[float, float]]:
    """
    根据当前车速，计算前瞻点 (Lookahead Point)。
    
    逻辑：
        1. 计算前瞻距离 = max(车速 * 前瞻时间, 最小前瞻距离)
        2. 在插值后的轨迹中，从起点开始搜索，找到第一个弧长距离超过前瞻距离的点。
    
    Args:
        trajectory: 插值后的密集轨迹点列表，包含 x, y。
        current_speed: 当前车速 (m/s)。
        lookahead_time: 前瞻时间 (秒)，通常设为 1.0 ~ 1.5 秒。
        min_lookahead_distance: 最低前瞻距离，防止车速为0时车辆不动。
    
    Returns:
        (x, y) 前瞻点的坐标，如果轨迹为空则返回 None。
    """
    if not trajectory:
        return None
    
    # 1. 计算前瞻距离
    lookahead_dist = max(current_speed * lookahead_time, min_lookahead_distance)
    
    # 2. 累积弧长，寻找目标点
    accumulated_dist = 0.0
    prev_x, prev_y = 0.0, 0.0  # 从原点开始
    
    for point in trajectory:
        x, y = point["x"], point["y"]
        # 计算与上一个点的欧几里得距离
        dist = math.hypot(x - prev_x, y - prev_y)
        accumulated_dist += dist
        
        if accumulated_dist >= lookahead_dist:
            return (x, y)
        
        prev_x, prev_y = x, y
    
    # 如果轨迹总长度小于前瞻距离，则直接返回最后一个点
    return (trajectory[-1]["x"], trajectory[-1]["y"])


# --- 离线测试入口 (用于验证你的 mock 数据) ---
if __name__ == "__main__":
    # 加载你刚才生成的 mock_trajectories.json
    with open("tests/test_data/mock_trajectories.json", "r") as f:
        mock_data = json.load(f)
    
    raw_points = mock_data["future_trajectory_ego_frame"]
    current_speed = mock_data["ego_state"]["speed"]
    
    print(f"原始点数: {len(raw_points)}")
    
    # 测试插值
    dense_traj = interpolate_trajectory(raw_points, step=0.1)
    print(f"插值后点数: {len(dense_traj)}")
    
    # 测试前瞻点计算
    lookahead = get_lookahead_point(dense_traj, current_speed, lookahead_time=1.2)
    if lookahead:
        print(f"当前车速: {current_speed} m/s")
        print(f"前瞻点坐标: x={lookahead[0]:.3f}, y={lookahead[1]:.3f}")
    else:
        print("前瞻点计算失败，请检查轨迹数据。")