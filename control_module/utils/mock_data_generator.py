import json
import math

def generate_straight_trajectory():
    """生成一个简单的直线匀速轨迹（6个点，覆盖3秒）"""
    points = []
    for i in range(1, 7):  # 0.5s 到 3.0s
        dt = i * 0.5
        points.append({
            "dt": round(dt, 2),
            "x": round(3.0 * dt, 2),   # 假设 3m/s 速度直行，X向前走
            "y": 0.0
        })
    return points

def generate_curve_trajectory():
    """生成一个左转弯轨迹（用于测试转向）"""
    points = []
    for i in range(1, 7):
        dt = i * 0.5
        points.append({
            "dt": round(dt, 2),
            "x": round(3.0 * dt, 2),
            "y": round(0.1 * (dt ** 2), 2)   # 抛物线，模拟左转
        })
    return points

# 生成并保存为 mock_trajectories.json
if __name__ == "__main__":
    # data = {
    #     "ego_state": {"speed": 3.0},  # 假设当前速度
    #     "future_trajectory_ego_frame": generate_curve_trajectory()
    # }
    data = { "ego_state": {"speed": 3.0}, 
            "future_trajectory_ego_frame": generate_straight_trajectory() }
    with open("tests/test_data/mock_trajectories.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Mock data generated successfully!")