"""
简易 MPC 控制器（横向控制）
基于运动学自行车模型，使用 casadi 求解最优转向角。
纵向速度由外部 PID 控制，MPC 仅输出 steer。
用法：
    from control_module.core.mpc_controller import MPCController
    mpc = MPCController(wheelbase=2.5, dt=0.1, N=15)
    steer = mpc.compute_steer(trajectory_points, current_speed, x, y, yaw)
"""
import math
import numpy as np
import casadi as ca


class MPCController:
    def __init__(self, wheelbase: float = 2.5, dt: float = 0.1, N: int = 15):
        """
        Args:
            wheelbase: 车辆轴距 (m)
            dt: 控制周期 (s)
            N: 预测时域步数 (总预测时长 = N * dt)
        """
        self.wheelbase = wheelbase
        self.dt = dt
        self.N = N
        
        # 转向角限制 (rad)
        self.max_steer = 0.7       # 约 40 度
        self.max_steer_rate = 0.3  # 每步最大变化率 (rad/s)
        
        # 权重矩阵 (调参重点)
        self.Q = np.diag([5.0, 1.0])   # 横向误差权重, 航向误差权重
        self.R = np.diag([0.1])        # 转向角变化率权重 (平滑性)
        self.P = np.diag([10.0, 2.0])  # 终端代价权重

        # 缓存求解器
        self.solver = None
        self.last_steer = 0.0
        self.opti = None
        
        # 参考轨迹插值缓存
        self._ref_x = None
        self._ref_y = None

    def _build_solver(self):
        """构建 casadi 优化问题（仅在第一次调用时构建）"""
        opti = ca.Opti()
        N = self.N
        dt = self.dt
        L = self.wheelbase

        # --- 1. 决策变量 ---
        # 状态变量: [x, y, psi] (全局坐标)
        X = opti.variable(3, N + 1)
        # 控制变量: steer (转向角)
        U = opti.variable(1, N)

        # --- 2. 参数（外部输入） ---
        # 初始状态 [x0, y0, psi0]
        X0 = opti.parameter(3)
        # 参考轨迹点 (N+1 个点, 每个点 [x_ref, y_ref])
        Ref = opti.parameter(2, N + 1)

        # --- 3. 目标函数 ---
        obj = 0
        # 初始状态约束
        opti.subject_to(X[:, 0] == X0)

        # 动态约束 & 代价累加
        for k in range(N):
            x_k = X[0, k]
            y_k = X[1, k]
            psi_k = X[2, k]
            v_k = self._current_speed  # 使用当前速度 (假设在预测时域内恒定)
            delta_k = U[0, k]

            # 运动学自行车模型 (离散化)
            x_next = x_k + v_k * ca.cos(psi_k) * dt
            y_next = y_k + v_k * ca.sin(psi_k) * dt
            psi_next = psi_k + v_k * ca.tan(delta_k) / L * dt

            # 约束: 状态转移
            opti.subject_to(X[0, k+1] == x_next)
            opti.subject_to(X[1, k+1] == y_next)
            opti.subject_to(X[2, k+1] == psi_next)

            # 控制量约束
            opti.subject_to(opti.bounded(-self.max_steer, delta_k, self.max_steer))

            # 转向变化率约束 (平滑性)
            if k > 0:
                d_delta = delta_k - U[0, k-1]
                opti.subject_to(opti.bounded(-self.max_steer_rate * dt, d_delta, self.max_steer_rate * dt))
            else:
                # 第一步限制变化率相对于上次输出
                d_delta = delta_k - self.last_steer
                opti.subject_to(opti.bounded(-self.max_steer_rate * dt, d_delta, self.max_steer_rate * dt))

            # 代价: 跟踪误差 (横向 + 航向)
            dx = X[0, k] - Ref[0, k]
            dy = X[1, k] - Ref[1, k]
            # 由于轨迹是自车坐标系下的点，这里为了简化，直接跟踪位置误差
            # 更精确的做法是计算法向误差，但位置误差在全局坐标下也有效
            obj += dx * self.Q[0, 0] * dx + dy * self.Q[1, 1] * dy
            
            # 转向变化率代价 (平滑)
            if k > 0:
                obj += (delta_k - U[0, k-1]) * self.R[0, 0] * (delta_k - U[0, k-1])

        # 终端代价
        dxN = X[0, N] - Ref[0, N]
        dyN = X[1, N] - Ref[1, N]
        obj += dxN * self.P[0, 0] * dxN + dyN * self.P[1, 1] * dyN

        # --- 4. 求解器配置 ---
        opti.minimize(obj)
        opts = {
            'ipopt': {
                'print_level': 0,           # 关闭调试输出
                'sb': 'yes',
                'max_iter': 100,
                'tol': 1e-4,
            },
            'print_time': 0
        }
        opti.solver('ipopt', opts)

        self.opti = opti
        self.solver = opti
        # 保存参数引用
        self._X0_param = X0
        self._Ref_param = Ref
        self._X_var = X
        self._U_var = U

    def compute_steer(self, trajectory_points, current_speed, x, y, yaw):
        """
        计算最优转向角
        
        Args:
            trajectory_points: 6个点的列表, 格式 [{"x": 1.0, "y": 0.0}, ...] (自车坐标系)
            current_speed: 当前车速 (m/s)
            x, y, yaw: 当前车辆在世界坐标系下的位置和朝向 (rad)
        
        Returns:
            steer: 最优转向角 (归一化到 [-1, 1])
        """
        if not trajectory_points or len(trajectory_points) < 2:
            return 0.0

        # 缓存速度供构建器使用
        self._current_speed = max(0.5, current_speed)

        # 1. 将自车坐标系的点转换到世界坐标系 (假设车辆当前位置为原点)
        # 将轨迹点从自车坐标 (dx, dy) 转为世界坐标 (wx, wy)
        # 注意：由于 MPC 在全局坐标下运行，我们需要一个绝对的参考线。
        # 最简单的方法：假设车辆当前在 (x, y)，轨迹点相对车辆，那么世界坐标 = 车辆位置 + 旋转后的相对坐标。
        ref_x = []
        ref_y = []
        for pt in trajectory_points:
            dx = pt["x"]
            dy = pt["y"]
            # 旋转到世界坐标系 (yaw 是车辆朝向)
            wx = x + dx * math.cos(yaw) - dy * math.sin(yaw)
            wy = y + dx * math.sin(yaw) + dy * math.cos(yaw)
            ref_x.append(wx)
            ref_y.append(wy)

        # 2. 插值出 N+1 个参考点 (因为 MPC 预测时域 N 步，需要 N+1 个参考点)
        # 如果轨迹点少于 N+1，则用最后一个点外推。
        ref_points = list(zip(ref_x, ref_y))
        if len(ref_points) < self.N + 1:
            # 用最后一个点填充
            last_pt = ref_points[-1]
            while len(ref_points) < self.N + 1:
                ref_points.append(last_pt)
        else:
            # 均匀采样 N+1 个点
            indices = np.linspace(0, len(ref_points)-1, self.N+1).astype(int)
            ref_points = [ref_points[i] for i in indices]

        # 3. 构建求解器 (首次调用时)
        if self.solver is None:
            self._build_solver()

        # 4. 设置参数
        self.opti.set_value(self._X0_param, [x, y, yaw])
        # Ref 是 2 x (N+1) 的矩阵
        ref_matrix = np.array(ref_points).T  # shape: (2, N+1)
        self.opti.set_value(self._Ref_param, ref_matrix)

        # 5. 设置初始猜测 (使用上一帧的解，热启动)
        # 如果没有历史解，用线性外推
        try:
            X_init = self.opti.initial(self._X_var)
            U_init = self.opti.initial(self._U_var)
            # 简单填充：直线行驶
            for k in range(self.N + 1):
                X_init[0, k] = x + k * self.dt * current_speed * math.cos(yaw)
                X_init[1, k] = y + k * self.dt * current_speed * math.sin(yaw)
                X_init[2, k] = yaw
            for k in range(self.N):
                U_init[0, k] = self.last_steer
            self.opti.set_initial(self._X_var, X_init)
            self.opti.set_initial(self._U_var, U_init)
        except:
            pass

        # 6. 求解
        try:
            sol = self.opti.solve()
            # 提取最优转向角 (第一个控制量)
            optimal_steer = sol.value(self._U_var[0, 0])
            # 更新缓存
            self.last_steer = optimal_steer
        except Exception as e:
            # 求解失败时，使用上一帧的转向角，或 fallback 到 0
            print(f"MPC 求解失败: {e}, 使用上一帧转向角 {self.last_steer:.3f}")
            optimal_steer = self.last_steer

        # 7. 归一化到 [-1, 1] (对应 CARLA 的 steer 范围)
        # 假设 max_steer = 0.7 rad 对应 steer = 1.0
        steer_normalized = optimal_steer / self.max_steer
        steer_normalized = max(-1.0, min(1.0, steer_normalized))
        
        return steer_normalized

    def reset(self):
        """重置控制器状态 (用于新场景)"""
        self.last_steer = 0.0
        self.solver = None
        self.opti = None