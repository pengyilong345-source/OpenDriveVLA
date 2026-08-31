# """
# PID 控制器：负责纵向速度控制（油门/刹车）
# 支持滑行优先策略，避免急刹
# """

# class PIDController:
#     def __init__(self, kp: float = 2.0, ki: float = 0.15, kd: float = 0.1):
#         self.kp = kp
#         self.ki = ki
#         self.kd = kd
        
#         self._integral = 0.0
#         self._prev_error = 0.0
#         self._dt = 0.1
        
#         self.integral_min = -2.0
#         self.integral_max = 2.0
#         self.output_min = -1.0
#         self.output_max = 1.0
    
#     def reset(self):
#         self._integral = 0.0
#         self._prev_error = 0.0
    
#     def compute(self, target_speed: float, current_speed: float, dt: float = 0.1) -> float:
#         self._dt = dt
#         error = target_speed - current_speed
        
#         self._integral += error * self._dt
#         self._integral = max(self.integral_min, min(self.integral_max, self._integral))
        
#         derivative = (error - self._prev_error) / self._dt if self._dt > 0 else 0.0
        
#         output = self.kp * error + self.ki * self._integral + self.kd * derivative
#         self._prev_error = error
        
#         return max(self.output_min, min(self.output_max, output))
    
#     def compute_throttle_brake(self, target_speed: float, current_speed: float, dt: float = 0.1) -> tuple:
#         """
#         将 PID 输出拆分为 throttle 和 brake，采用滑行优先策略。
        
#         策略：
#             1. 需要加速（控制量 > 0）→ 输出油门
#             2. 轻微超速（速度误差 > -0.5 m/s）→ 滑行（油门刹车均为0）
#             3. 严重超速（速度误差 <= -0.5 m/s）→ 按比例刹车，且限制最大刹车力度
#         """
#         control = self.compute(target_speed, current_speed, dt)
        
#         if control >= 0.0:
#             throttle = min(1.0, control)
#             brake = 0.0
#         else:
#             speed_error = target_speed - current_speed  # 负值表示超速
            
#             # 滑行区：超速小于 0.5 m/s 时不踩刹车
#             if speed_error > -0.5:
#                 throttle = 0.0
#                 brake = 0.0
#             else:
#                 # 按比例施加刹车，限制最大刹车力度
#                 # 超速越严重，刹车越大，但最大不超过 0.5
#                 max_brake = 1.0  # 最大刹车力度
#                 # 将超速程度映射到刹车力度
#                 # 速度误差从 -0.5 到 -10 m/s 映射到 0 到 max_brake
#                 brake = min(max_brake, (-speed_error - 0.5) * 0.05)  # 0.05 为比例系数
#                 # 同时考虑 PID 输出
#                 brake = max(brake, min(max_brake, abs(control) * 0.4))
#                 throttle = 0.0
        
#         # 低速防锁死
#         if current_speed < 0.5 and target_speed > 0.5 and brake > 0.3:
#             brake = 0.3
        
#         return throttle, brake


# # --- 测试入口 ---
# if __name__ == "__main__":
#     pid = PIDController(kp=1.5, ki=0.15, kd=0.1)
#     target = 7.5
#     current = 0.0
    
#     print("=== 测试PID刹车逻辑（目标7.5 m/s）===")
#     for i in range(30):
#         t, b = pid.compute_throttle_brake(target, current, 0.1)
#         # 模拟加速：油门1.0提供2.0 m/s²加速度，刹车0.5提供3.0 m/s²减速度
#         accel = t * 2.0 - b * 3.0
#         current = max(0, current + accel * 0.1)
#         if i % 5 == 0:
#             print(f"Step {i}: speed={current:.2f}, throttle={t:.2f}, brake={b:.2f}")

"""
PID 控制器：负责纵向速度控制（油门/刹车）
支持滑行优先策略，避免急刹
"""

class PIDController:
    def __init__(self, kp: float = 1.5, ki: float = 0.15, kd: float = 0.1):
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
                # 降低最大刹车力度，避免急刹
                max_brake = 0.35  # 从 0.5 降低到 0.35
                # 将超速程度映射到刹车力度，比例系数降低
                brake = min(max_brake, (-speed_error - 0.5) * 0.03)  # 0.03 降低灵敏度
                # 同时考虑 PID 输出，但降低影响
                brake = max(brake, min(max_brake, abs(control) * 0.25))
                throttle = 0.0
        
        # 低速防锁死：速度很低且目标速度不为 0 时，限制刹车力度
        if current_speed < 0.5 and target_speed > 0.5 and brake > 0.2:
            brake = 0.2
        
        return throttle, brake


# --- 测试入口 ---
if __name__ == "__main__":
    pid = PIDController(kp=1.5, ki=0.15, kd=0.1)
    target = 7.5
    current = 0.0
    
    print("=== 测试PID刹车逻辑（目标7.5 m/s）===")
    for i in range(30):
        t, b = pid.compute_throttle_brake(target, current, 0.1)
        accel = t * 2.0 - b * 3.0
        current = max(0, current + accel * 0.1)
        if i % 5 == 0:
            print(f"Step {i}: speed={current:.2f}, throttle={t:.2f}, brake={b:.2f}")