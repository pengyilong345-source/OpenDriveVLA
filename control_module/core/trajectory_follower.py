# """
# 纯跟踪控制器 (Pure Pursuit)：负责横向控制（转向）
# """

# import math
# from typing import Tuple, Optional, List, Dict
# import sys
# import os
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# from control_module.utils.trajectory_utils import get_lookahead_point, interpolate_trajectory


# class PurePursuitController:
#     def __init__(self, wheelbase: float = 2.5, lookahead_time: float = 0.8):
#         """
#         初始化纯跟踪控制器。
        
#         Args:
#             wheelbase: 车辆轴距 (米)，CARLA 默认约 2.8m
#             lookahead_time: 前瞻时间 (秒)，通常 1.0 ~ 1.5 秒
#         """
#         self.wheelbase = wheelbase
#         self.lookahead_time = lookahead_time
#         self.min_lookahead_distance = 0.5  # 最低前瞻距离，防低速抖动
    
#     def compute_steer(
#         self, 
#         trajectory: List[Dict], 
#         current_speed: float,
#         dt: float = 0.1
#     ) -> Tuple[float, Optional[Tuple[float, float]]]:
#         """
#         计算转向角 (steer)。
        
#         核心思路：
#             1. 先对稀疏轨迹进行插值，得到密集路径
#             2. 根据车速计算前瞻点
#             3. 用纯跟踪公式计算前轮转角
        
#         Args:
#             trajectory: 稀疏轨迹点 (6个点，来自 JSON)
#             current_speed: 当前车速 (m/s)
#             dt: 控制周期 (秒)，用于插值步长
        
#         Returns:
#             steer: 转向角，范围 [-1.0, 1.0]
#             lookahead_point: (x, y) 前瞻点坐标，用于调试可视化
#         """
#         if not trajectory:
#             return 0.0, None
        
#         # 1. 插值得到密集轨迹
#         dense_trajectory = interpolate_trajectory(trajectory, step=dt)
#         if not dense_trajectory:
#             return 0.0, None
        
#         # 2. 计算前瞻点
#         lookahead = get_lookahead_point(
#             dense_trajectory, 
#             current_speed, 
#             self.lookahead_time,
#             self.min_lookahead_distance
#         )
        
#         if lookahead is None:
#             return 0.0, None
        
#         lookahead_x, lookahead_y = lookahead
        
#         # 3. 纯跟踪公式：计算转向角
#         # 前瞻距离 L
#         L = math.hypot(lookahead_x, lookahead_y)
#         if L < 0.001:  # 防止除零
#             return 0.0, lookahead
        
#         # 曲率 k = 2 * y / L^2
#         curvature = (2.0 * lookahead_y) / (L * L)
        
#         # 转向角 = arctan(k * wheelbase)
#         steer_rad = math.atan2(curvature * self.wheelbase, 1.0)
        
#         # 4. 归一化到 [-1.0, 1.0]，并限幅
#         # 假设最大转向角为 0.7 rad（约 40 度）
#         max_steer_rad = 1.0  # 可根据车辆模型调整
#         steer_normalized = steer_rad / max_steer_rad
#         steer = max(-1.0, min(1.0, steer_normalized))
#         # steer = -steer  # 方向调整：左转为正，右转为负
#         return steer, lookahead


# # --- 离线测试入口 ---
# if __name__ == "__main__":
#     import json
#     import sys
#     import os
    
#     # 添加项目根目录到路径（确保能找到 utils 模块）
#     sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
#     # 加载 mock 数据
#     mock_path = "tests/test_data/mock_trajectories.json"
#     if not os.path.exists(mock_path):
#         # 尝试绝对路径
#         mock_path = os.path.join(os.path.dirname(__file__), "..", "tests", "test_data", "mock_trajectories.json")
    
#     with open(mock_path, "r") as f:
#         mock_data = json.load(f)
    
#     raw_trajectory = mock_data["future_trajectory_ego_frame"]
#     current_speed = mock_data["ego_state"]["speed"]
    
#     # 初始化控制器
#     follower = PurePursuitController(wheelbase=2.8, lookahead_time=1.2)
    
#     # 计算转向角
#     steer, lookahead = follower.compute_steer(raw_trajectory, current_speed)
    
#     print("=== 纯跟踪控制器测试 ===")
#     print(f"当前车速: {current_speed} m/s")
#     print(f"前瞻点: x={lookahead[0]:.3f}, y={lookahead[1]:.3f}")
#     print(f"输出转向角: {steer:.4f} (范围 -1.0 ~ 1.0)")
#     print("-" * 50)
    
#     # 测试低速场景
#     print("\n--- 低速测试 (1.0 m/s) ---")
#     steer_low, _ = follower.compute_steer(raw_trajectory, 1.0)
#     print(f"低速转向角: {steer_low:.4f}")
    
#     # 测试高速场景
#     print("\n--- 高速测试 (10.0 m/s) ---")
#     steer_high, _ = follower.compute_steer(raw_trajectory, 10.0)
#     print(f"高速转向角: {steer_high:.4f}")

"""
纯跟踪控制器 (Pure Pursuit)：负责横向控制（转向）
"""

import math
from typing import Tuple, Optional, List, Dict
from control_module.utils.trajectory_utils import get_lookahead_point, interpolate_trajectory


class PurePursuitController:
    def __init__(self, wheelbase: float = 2.5, lookahead_time: float = 0.8):
        """
        初始化纯跟踪控制器。
        
        Args:
            wheelbase: 车辆轴距 (米)，减小可使转向更灵敏 (默认 2.5)
            lookahead_time: 前瞻时间 (秒)，减小可使转向更激进 (默认 0.8)
        """
        self.wheelbase = wheelbase
        self.lookahead_time = lookahead_time
        self.min_lookahead_distance = 0.5
        # 转向增益，可微调，暂时设为1.0
        self.steer_gain = 1.0
    
    def compute_steer(
        self, 
        trajectory: List[Dict], 
        current_speed: float,
        dt: float = 0.1
    ) -> Tuple[float, Optional[Tuple[float, float]]]:
        if not trajectory:
            return 0.0, None
        
        dense_trajectory = interpolate_trajectory(trajectory, step=dt)
        if not dense_trajectory:
            return 0.0, None
        
        lookahead = get_lookahead_point(
            dense_trajectory, 
            current_speed, 
            self.lookahead_time,
            self.min_lookahead_distance
        )
        
        if lookahead is None:
            return 0.0, None
        
        lookahead_x, lookahead_y = lookahead
        
        L = math.hypot(lookahead_x, lookahead_y)
        if L < 0.001:
            return 0.0, lookahead
        
        curvature = (2.0 * lookahead_y) / (L * L)
        steer_rad = math.atan2(curvature * self.wheelbase, 1.0)
        
        max_steer_rad = 0.7
        steer_normalized = steer_rad / max_steer_rad
        steer = max(-1.0, min(1.0, steer_normalized))
        
        # 应用转向增益 (目前为1.0，可根据需要微调)
        steer = steer * self.steer_gain
        steer = max(-1.0, min(1.0, steer))
        
        return steer, lookahead