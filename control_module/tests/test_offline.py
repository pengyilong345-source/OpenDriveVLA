"""
离线闭环测试：整合所有模块，模拟一个完整的控制循环。

流程：
    1. 加载 mock_trajectories.json
    2. 从 t=0 开始，模拟车辆运动
    3. 每步调用控制器，输出 (steer, throttle, brake)
    4. 应用控制量，更新车辆状态
    5. 记录并显示结果
"""

import sys
import os
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import math
import time
from typing import Dict, List

from control_module.core.trajectory_follower import PurePursuitController
from control_module.core.pid_controller import PIDController
from control_module.safety.safety_monitor import SafetyMonitor


class VehicleSimulator:
    """
    简单的车辆动力学模拟器（用于离线测试）。
    仅作为玩具模型，真实场景会使用 CARLA 引擎。
    """
    def __init__(self, initial_speed: float = 0.0, dt: float = 0.1):
        self.speed = initial_speed  # m/s
        self.dt = dt
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        
        # 车辆参数（简化为前轮驱动）
        self.max_accel = 3.0  # m/s^2
        self.max_brake = 4.0  # m/s^2
        self.max_steer_angle = 0.7  # rad
    
    def step(self, throttle: float, brake: float, steer: float) -> None:
        """
        更新车辆状态。
        
        简化模型：
            - 加速度 = throttle * max_accel - brake * max_brake
            - 速度 = max(0, speed + accel * dt)
            - 位置更新根据速度和航向角
        """
        # 纵向动力学
        accel = throttle * self.max_accel - brake * self.max_brake
        self.speed = max(0.0, self.speed + accel * self.dt)
        
        # 横向动力学（简化为阿克曼模型）
        if self.speed > 0.1:
            steer_rad = steer * self.max_steer_angle
            self.yaw += (self.speed / 2.8) * math.tan(steer_rad) * self.dt  # wheelbase=2.8
        
        # 位置更新
        self.x += self.speed * math.cos(self.yaw) * self.dt
        self.y += self.speed * math.sin(self.yaw) * self.dt


def run_offline_test():
    """运行完整的离线闭环测试"""
    
    print("=" * 60)
    print("DriveVLA 下游控制模块 - 离线闭环测试")
    print("=" * 60)
    
    # 1. 加载 mock 数据
    mock_path = os.path.join(os.path.dirname(__file__), "test_data", "mock_trajectories.json")
    if not os.path.exists(mock_path):
        print(f"错误: 找不到 mock 数据文件: {mock_path}")
        print("请先运行 control_module/utils/mock_data_generator.py 生成数据")
        return
    
    with open(mock_path, "r") as f:
        mock_data = json.load(f)
    
    trajectory = mock_data["future_trajectory_ego_frame"]
    target_speed = mock_data["ego_state"]["speed"]
    
    print(f"目标轨迹: {len(trajectory)} 个稀疏点")
    print(f"目标速度: {target_speed} m/s")
    print("-" * 60)
    
    # 2. 初始化控制器
    pid = PIDController(kp=0.6, ki=0.12, kd=0.08)
    pure_pursuit = PurePursuitController(wheelbase=2.8, lookahead_time=1.2)
    safety = SafetyMonitor()
    
    # 3. 初始化模拟车辆
    vehicle = VehicleSimulator(initial_speed=0.0, dt=0.1)
    
    # 4. 模拟循环
    max_steps = 80  # 模拟 8 秒 (80 * 0.1s)
    log_data = []
    
    print("开始模拟...")
    print("时间(s) | 车速(m/s) | 油门 | 刹车 | 转向 | 横向误差(m)")
    print("-" * 60)
    
    for step in range(max_steps):
        t = step * 0.1
        
        # 4.1 计算控制量
        # 横向：从轨迹中提取转向角
        steer, lookahead = pure_pursuit.compute_steer(trajectory, vehicle.speed)
        
        # 纵向：PID控制速度
        throttle, brake = pid.compute_throttle_brake(target_speed, vehicle.speed)
        
        # 4.2 安全过滤
        steer, throttle, brake = safety.filter_control(steer, throttle, brake, vehicle.speed)
        
        # 4.3 应用控制量，更新车辆
        vehicle.step(throttle, brake, steer)
        
        # 4.4 计算横向误差（相对于当前轨迹的最近点）
        # 简化：直接用当前轨迹第一个点的y值作为横向误差
        lateral_error = abs(vehicle.y)  # 假设目标轨迹在 y=0 附近
        
        # 4.5 记录日志
        log_data.append({
            "time": t,
            "speed": vehicle.speed,
            "throttle": throttle,
            "brake": brake,
            "steer": steer,
            "lateral_error": lateral_error,
            "x": vehicle.x,
            "y": vehicle.y
        })
        
        # 4.6 打印实时状态（每5步打印一次）
        if step % 5 == 0 or step == max_steps - 1:
            print(f"{t:6.1f}  | {vehicle.speed:8.2f} | {throttle:5.3f} | {brake:5.3f} | {steer:6.3f} | {lateral_error:12.4f}")
        
        # 如果车辆已经接近目标点（走了超过8米），可以提前结束
        if vehicle.x > 10.0:
            print(f"车辆已行驶 {vehicle.x:.2f} 米，提前结束模拟")
            break
    
    # 5. 输出统计信息
    print("-" * 60)
    print("模拟完成！")
    print(f"总步数: {len(log_data)}")
    print(f"最终车速: {vehicle.speed:.2f} m/s")
    print(f"总行驶距离: {vehicle.x:.2f} m")
    print(f"最大横向误差: {max(d['lateral_error'] for d in log_data):.4f} m")
    
    # 6. 保存日志到文件
    import csv
    log_file = os.path.join(os.path.dirname(__file__), "..", "logs", "offline_test.csv")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_data[0].keys())
        writer.writeheader()
        writer.writerows(log_data)
    
    print(f"日志已保存到: {log_file}")
    print("=" * 60)


if __name__ == "__main__":
    run_offline_test()