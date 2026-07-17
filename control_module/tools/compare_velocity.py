"""
车速跟踪对比分析工具
功能：分析实际车速是否跟上目标车速，检查油门/刹车输出是否合理
用法：python compare_velocity.py [日志文件路径]
"""
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def find_latest_log(log_dir='../logs'):
    log_dir = os.path.join(os.path.dirname(__file__), log_dir)
    files = glob.glob(os.path.join(log_dir, 'replay_control_*.csv'))
    if not files:
        raise FileNotFoundError(f"在 {log_dir} 中未找到日志文件")
    return max(files, key=os.path.getmtime)


def main():
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = find_latest_log()
        print(f"自动选择最新日志: {log_path}")

    df = pd.read_csv(log_path)
    print(f"✅ 加载日志: {log_path}，共 {len(df)} 帧")

    # 检查必需字段
    required_cols = ['speed', 'target_speed', 'throttle', 'brake']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠️ 日志缺少字段: {missing}，请确保 run_control_loop.py 记录了这些数据")
        return

    time = df['time'].values
    speed = df['speed'].values
    target_speed = df['target_speed'].values
    throttle = df['throttle'].values
    brake = df['brake'].values

    # 计算速度误差
    error = target_speed - speed
    mae_speed = np.mean(np.abs(error))
    rmse_speed = np.sqrt(np.mean(error**2))
    max_speed_error = np.max(np.abs(error))

    # ====== 绘图 ======
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('纵向控制（速度跟踪）分析', fontsize=16)

    # 图1：速度对比
    ax = axes[0]
    ax.plot(time, target_speed, 'r--', linewidth=1.5, label='目标速度 (专家)', alpha=0.8)
    ax.plot(time, speed, 'b-', linewidth=1.5, label='实际车速', alpha=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度 (m/s)')
    ax.set_title(f'速度跟踪 | MAE={mae_speed:.3f}, RMSE={rmse_speed:.3f}, MaxErr={max_speed_error:.3f}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # 图2：速度误差
    ax = axes[1]
    ax.plot(time, error, 'g-', linewidth=1.0, alpha=0.7, label='速度误差 (目标 - 实际)')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axhline(mae_speed, color='orange', linestyle='--', linewidth=0.8, label=f'MAE = {mae_speed:.3f}')
    ax.axhline(-mae_speed, color='orange', linestyle='--', linewidth=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度误差 (m/s)')
    ax.set_title('速度误差')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # 图3：油门/刹车
    ax = axes[2]
    ax.plot(time, throttle, 'b-', linewidth=1.5, label='油门', alpha=0.8)
    ax.plot(time, brake, 'r-', linewidth=1.5, label='刹车', alpha=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('控制量')
    ax.set_title('油门/刹车输出')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

    # ====== 诊断建议 ======
    print("\n--- 速度跟踪诊断 ---")
    print(f"平均速度误差: {mae_speed:.3f} m/s")
    print(f"最大速度误差: {max_speed_error:.3f} m/s")

    # 检查油门饱和情况
    throttle_saturation_ratio = np.mean(throttle > 0.95) * 100
    print(f"油门饱和 (>0.95) 的比例: {throttle_saturation_ratio:.1f}%")

    if throttle_saturation_ratio > 30:
        print("⚠️ 油门饱和严重：车辆加速能力不足，可能是:")
        print("  1. PID 的 kp 太小（反应慢）")
        print("  2. 车辆最大加速度有限（CARLA 车辆参数）")
        print("  3. 目标速度变化太快")
    else:
        print("✅ 油门未持续饱和，说明加速能力尚可。")

    if mae_speed > 1.0:
        print("⚠️ 速度误差较大 (>1 m/s)，建议增大 PID 的 kp 或 ki。")
    else:
        print("✅ 速度跟踪精度尚可。")


if __name__ == "__main__":
    main()