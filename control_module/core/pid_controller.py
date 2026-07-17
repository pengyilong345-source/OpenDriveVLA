# """
# PID 控制器：负责纵向速度控制（油门/刹车）
# """

# class PIDController:
#     def __init__(self, kp: float = 0.5, ki: float = 0.1, kd: float = 0.05):
#         """
#         初始化 PID 控制器。
        
#         Args:
#             kp: 比例增益 (Proportional)
#             ki: 积分增益 (Integral)
#             kd: 微分增益 (Derivative)
#         """
#         self.kp = kp
#         self.ki = ki
#         self.kd = kd
        
#         self._integral = 0.0
#         self._prev_error = 0.0
#         self._dt = 0.1  # 默认控制周期 0.1s
        
#         # 积分限幅，防止积分饱和
#         self.integral_min = -2.0
#         self.integral_max = 2.0
        
#         # 输出限幅
#         self.output_min = -1.0  # -1.0 对应最大刹车
#         self.output_max = 1.0   # 1.0 对应最大油门
    
#     def reset(self):
#         """重置积分和微分状态（用于切换目标时）"""
#         self._integral = 0.0
#         self._prev_error = 0.0
    
#     def compute(self, target_speed: float, current_speed: float, dt: float = 0.1) -> float:
#         """
#         计算控制量（正值=油门，负值=刹车）
        
#         Args:
#             target_speed: 目标速度 (m/s)
#             current_speed: 当前速度 (m/s)
#             dt: 控制周期 (秒)
        
#         Returns:
#             float: 控制量，范围 [-1.0, 1.0]
#                   - 正值表示油门 (0.0 ~ 1.0)
#                   - 负值表示刹车 (-1.0 ~ 0.0)
#         """
#         self._dt = dt
        
#         # 1. 计算误差
#         error = target_speed - current_speed
        
#         # 2. 积分项（带限幅）
#         self._integral += error * self._dt
#         self._integral = max(self.integral_min, min(self.integral_max, self._integral))
        
#         # 3. 微分项
#         derivative = (error - self._prev_error) / self._dt if self._dt > 0 else 0.0
        
#         # 4. PID 输出
#         output = self.kp * error + self.ki * self._integral + self.kd * derivative
        
#         # 5. 保存误差供下次使用
#         self._prev_error = error
        
#         # 6. 输出限幅
#         return max(self.output_min, min(self.output_max, output))
    
#     def compute_throttle_brake(self, target_speed: float, current_speed: float, dt: float = 0.1) -> tuple:
#         """
#         将 PID 输出拆分为 throttle 和 brake 指令。
        
#         Returns:
#             (throttle, brake): 两个浮点数，范围 [0.0, 1.0]
#         """
#         control = self.compute(target_speed, current_speed, dt)
        
#         if control >= 0.0:
#             # 需要加速
#             throttle = min(1.0, control)
#             brake = 0.0
#         else:
#             # 需要减速
#             throttle = 0.0
#             brake = min(1.0, abs(control))
        
#         return throttle, brake


# # --- 离线测试入口 ---
# if __name__ == "__main__":
#     # 模拟一个简单的速度控制场景
#     pid = PIDController(kp=0.6, ki=0.12, kd=0.08)  
    
#     target_speed = 3.0  # 目标 3 m/s
#     current_speed = 0.0  # 从静止开始
    
#     print("=== PID 速度控制器测试 ===")
#     print(f"目标速度: {target_speed} m/s")
#     print("-" * 50)
    
#     for step in range(20):  # 模拟 2 秒 (20 * 0.1s)
#         dt = 0.1
#         throttle, brake = pid.compute_throttle_brake(target_speed, current_speed, dt)
        
#         # 模拟车辆加速（简单模型：加速度 = 控制量 * 2.0）
#         acceleration = throttle * 2.0 - brake * 3.0
#         current_speed = max(0.0, current_speed + acceleration * dt)
        
#         print(f"Step {step+1:2d} | 当前速度: {current_speed:.2f} m/s | "
#               f"油门: {throttle:.3f} | 刹车: {brake:.3f}")
        
#         if abs(current_speed - target_speed) < 0.1 and step > 5:
#             print("✓ 已达到目标速度")
#             break

"""
PID 控制器：负责纵向速度控制（油门/刹车）
支持滑行优先策略，避免急刹
"""

class PIDController:
    def __init__(self, kp: float = 0.6, ki: float = 0.12, kd: float = 0.08):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self._integral = 0.0
        self._prev_error = 0.0
        self._dt = 0.1
        
        self.integral_min = -2.0
        self.integral_max = 2.0
        self.output_min = -1.0
        self.output_max = 1.0
    
    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
    
    def compute(self, target_speed: float, current_speed: float, dt: float = 0.1) -> float:
        self._dt = dt
        error = target_speed - current_speed
        
        self._integral += error * self._dt
        self._integral = max(self.integral_min, min(self.integral_max, self._integral))
        
        derivative = (error - self._prev_error) / self._dt if self._dt > 0 else 0.0
        
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._prev_error = error
        
        return max(self.output_min, min(self.output_max, output))
    
    def compute_throttle_brake(self, target_speed: float, current_speed: float, dt: float = 0.1) -> tuple:
        """
        将 PID 输出拆分为 throttle 和 brake，采用滑行优先策略。
        
        策略：
            1. 需要加速（控制量 > 0）→ 输出油门
            2. 轻微超速（速度误差 > -0.5 m/s）→ 滑行（油门刹车均为0）
            3. 严重超速（速度误差 <= -0.5 m/s）→ 按比例刹车，且限制最大刹车力度
        """
        control = self.compute(target_speed, current_speed, dt)
        
        if control >= 0.0:
            throttle = min(1.0, control)
            brake = 0.0
        else:
            speed_error = target_speed - current_speed  # 负值表示超速
            
            # 滑行区：超速小于 0.5 m/s 时不踩刹车
            if speed_error > -0.5:
                throttle = 0.0
                brake = 0.0
            else:
                # 按比例施加刹车，限制最大刹车力度
                max_brake = 0.5
                brake = min(max_brake, abs(control))
                throttle = 0.0
        
        # 低速防锁死
        if current_speed < 0.5 and target_speed > 0.5 and brake > 0.3:
            brake = 0.3
        
        return throttle, brake


# --- 测试入口 ---
if __name__ == "__main__":
    pid = PIDController(kp=0.6, ki=0.12, kd=0.08)
    target = 7.5
    current = 0.0
    
    print("=== 测试PID刹车逻辑（目标7.5 m/s）===")
    for i in range(30):
        t, b = pid.compute_throttle_brake(target, current, 0.1)
        # 模拟加速：油门1.0提供2.0 m/s²加速度，刹车0.5提供3.0 m/s²减速度
        accel = t * 2.0 - b * 3.0
        current = max(0, current + accel * 0.1)
        if i % 5 == 0:
            print(f"Step {i}: speed={current:.2f}, throttle={t:.2f}, brake={b:.2f}")