"""
安全监控模块 (Safety Monitor)

职责：
    1. 限制控制量变化率（防止突变导致车辆抖动）
    2. 输出限幅（确保 steer/throttle/brake 在合法范围内）
    3. 紧急刹车逻辑（基于速度或外部信号）
    4. 防止 throttle 和 brake 同时输出
"""

class SafetyMonitor:
    def __init__(self, 
                 max_steer_rate: float = 0.3,      # 每帧最大转向变化率
                 max_throttle_rate: float = 0.5,   # 每帧最大油门变化率
                 max_brake_rate: float = 0.5,      # 每帧最大刹车变化率
                 emergency_brake_threshold: float = 0.5):  # 紧急刹车速度阈值
        """
        初始化安全监控器。
        
        Args:
            max_steer_rate: 转向角每帧最大变化量 (0.0 ~ 1.0)
            max_throttle_rate: 油门每帧最大变化量 (0.0 ~ 1.0)
            max_brake_rate: 刹车每帧最大变化量 (0.0 ~ 1.0)
            emergency_brake_threshold: 紧急刹车触发速度阈值 (m/s)
                如果当前速度超过此值且刹车指令突然增大，触发紧急刹车
        """
        self.max_steer_rate = max_steer_rate
        self.max_throttle_rate = max_throttle_rate
        self.max_brake_rate = max_brake_rate
        self.emergency_brake_threshold = emergency_brake_threshold
        
        # 上一帧的控制量（用于计算变化率）
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        
        # 紧急刹车标志
        self.emergency_brake_active = False
    
    def reset(self):
        """重置内部状态（用于新场景或重置后）"""
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.emergency_brake_active = False
    
    def filter_control(self, 
                       steer: float, 
                       throttle: float, 
                       brake: float, 
                       current_speed: float) -> tuple:
        """
        对控制量进行安全过滤。
        
        Args:
            steer: 原始转向角 (范围 -1.0 ~ 1.0)
            throttle: 原始油门 (范围 0.0 ~ 1.0)
            brake: 原始刹车 (范围 0.0 ~ 1.0)
            current_speed: 当前车速 (m/s)
        
        Returns:
            (filtered_steer, filtered_throttle, filtered_brake)
        """
        # ===== 1. 限幅（硬安全） =====
        steer = max(-1.0, min(1.0, steer))
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        
        # ===== 2. 防止油门和刹车同时输出 =====
        # 如果刹车 > 0.1，强制油门为0；如果油门 > 0.1，强制刹车为0
        if brake > 0.1 and throttle > 0.1:
            # 如果刹车和油门冲突，优先刹车（安全第一）
            throttle = 0.0
        elif throttle > 0.1 and brake > 0.1:
            # 同上
            brake = 0.0
        
        # ===== 3. 紧急刹车检测 =====
        # 如果车速高于阈值且刹车指令突然增大（从0到>0.5），视为紧急情况
        if (current_speed > self.emergency_brake_threshold and 
            brake > 0.5 and self.prev_brake < 0.1):
            self.emergency_brake_active = True
        
        # 如果紧急刹车标志为真，强制最大刹车
        if self.emergency_brake_active:
            # 当速度降到接近0时，解除紧急刹车标志
            if current_speed < 0.5:
                self.emergency_brake_active = False
            else:
                # 强制最大刹车，油门归零
                throttle = 0.0
                brake = 1.0
        
        # ===== 4. 变化率限制（平滑性保护） =====
        # 计算与上一帧的差值
        steer_diff = steer - self.prev_steer
        throttle_diff = throttle - self.prev_throttle
        brake_diff = brake - self.prev_brake
        
        # 应用变化率限制
        steer = self.prev_steer + max(-self.max_steer_rate, 
                                       min(self.max_steer_rate, steer_diff))
        throttle = self.prev_throttle + max(-self.max_throttle_rate, 
                                            min(self.max_throttle_rate, throttle_diff))
        brake = self.prev_brake + max(-self.max_brake_rate, 
                                      min(self.max_brake_rate, brake_diff))
        
        # 再次限幅（防止变化率限幅后超出边界）
        steer = max(-1.0, min(1.0, steer))
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        
        # ===== 5. 保存当前值供下一帧使用 =====
        self.prev_steer = steer
        self.prev_throttle = throttle
        self.prev_brake = brake
        
        return steer, throttle, brake
    
    def get_status(self) -> dict:
        """获取当前安全状态（用于调试/日志）"""
        return {
            "emergency_brake_active": self.emergency_brake_active,
            "prev_steer": self.prev_steer,
            "prev_throttle": self.prev_throttle,
            "prev_brake": self.prev_brake
        }


# --- 离线测试入口 ---
if __name__ == "__main__":
    import time
    
    print("=== 安全模块测试 ===")
    safety = SafetyMonitor()
    
    # 测试1：正常加速场景
    print("\n1. 正常加速场景 (无突变)")
    safety.reset()
    test_controls = [
        (0.0, 0.0, 0.0, 0.0),   # (steer, throttle, brake, speed)
        (0.1, 0.5, 0.0, 2.0),
        (0.2, 0.8, 0.0, 5.0),
        (0.3, 0.6, 0.0, 8.0),
    ]
    for s, t, b, v in test_controls:
        filtered = safety.filter_control(s, t, b, v)
        print(f"  输入: ({s:.2f}, {t:.2f}, {b:.2f}) 速度={v:.1f} -> 输出: ({filtered[0]:.2f}, {filtered[1]:.2f}, {filtered[2]:.2f})")
    
    # 测试2：紧急刹车场景
    print("\n2. 紧急刹车场景 (高速突踩刹车)")
    safety.reset()
    test_controls = [
        (0.0, 0.5, 0.0, 10.0),   # 巡航
        (0.1, 0.0, 0.8, 10.0),   # 突然踩刹车
        (0.1, 0.0, 0.8, 5.0),    # 速度下降
        (0.1, 0.0, 0.8, 1.0),    # 接近停车
        (0.1, 0.5, 0.0, 0.5),    # 恢复加速（紧急刹车应已解除）
    ]
    for s, t, b, v in test_controls:
        filtered = safety.filter_control(s, t, b, v)
        status = safety.get_status()
        print(f"  输入: ({s:.2f}, {t:.2f}, {b:.2f}) 速度={v:.1f} -> 输出: ({filtered[0]:.2f}, {filtered[1]:.2f}, {filtered[2]:.2f}), 紧急刹车: {status['emergency_brake_active']}")