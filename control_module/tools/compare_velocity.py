# """
# 车速跟踪对比分析工具
# 功能：分析实际车速是否跟上目标车速，检查油门/刹车输出是否合理
# 用法：
#     默认：python compare_velocity.py
#     指定日志：python compare_velocity.py logs/xxx.csv
#     指定输出目录：python compare_velocity.py --save_to_dir ./my_analysis
# """
# import os
# import sys
# import glob
# import argparse
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

# DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), "output")


# def find_latest_log(log_dir='../logs'):
#     log_dir = os.path.join(os.path.dirname(__file__), log_dir)
#     files = glob.glob(os.path.join(log_dir, 'replay_control_*.csv'))
#     if not files:
#         raise FileNotFoundError(f"在 {log_dir} 中未找到日志文件")
#     return max(files, key=os.path.getmtime)


# def main():
#     parser = argparse.ArgumentParser(description="车速跟踪分析")
#     parser.add_argument("log_path", nargs='?', help="日志文件路径（可选，不指定则自动选最新）")
#     parser.add_argument("--save_to_dir", "-s", type=str, default=DEFAULT_SAVE_DIR,
#                         help=f"保存图片的目录（默认: {DEFAULT_SAVE_DIR}）")
#     args = parser.parse_args()

#     if args.log_path:
#         log_path = args.log_path
#     else:
#         try:
#             log_path = find_latest_log()
#             print(f"自动选择最新日志: {log_path}")
#         except FileNotFoundError as e:
#             print(e)
#             return

#     df_raw = pd.read_csv(log_path)
#     print(f"✅ 加载日志: {log_path}，原始共 {len(df_raw)} 帧")

#     # ===== 过滤：排除速度为 0 的点（撞击后停滞） =====
#     speed_threshold = 0.1  # m/s，速度小于此值视为停滞
#     df = df_raw[df_raw['speed'] > speed_threshold].copy()
#     print(f"✅ 过滤后剩余 {len(df)} 帧（剔除速度 <= {speed_threshold} m/s 的点）")

#     if len(df) == 0:
#         print("⚠️ 过滤后没有有效数据，请检查日志")
#         return

#     required_cols = ['speed', 'target_speed', 'throttle', 'brake']
#     missing = [c for c in required_cols if c not in df.columns]
#     if missing:
#         print(f"⚠️ 日志缺少字段: {missing}，请确保 run_control_loop.py 记录了这些数据")
#         return

#     time = df['time'].values
#     speed = df['speed'].values
#     target_speed = df['target_speed'].values
#     throttle = df['throttle'].values
#     brake = df['brake'].values

#     error = target_speed - speed
#     mae = np.mean(np.abs(error))
#     rmse = np.sqrt(np.mean(error**2))
#     max_error = np.max(np.abs(error))
#     std_error = np.std(error)
#     saturation_ratio = np.mean(throttle > 0.95) * 100

#     # 误差超过 1.0 m/s 的比例
#     error_threshold = 1.0
#     large_error_ratio = (np.abs(error) > error_threshold).sum() / len(error) * 100

#     # 找出误差最大的前 5 个点
#     sorted_indices = np.argsort(np.abs(error))[::-1]
#     top5_indices = sorted_indices[:5]

#     # ====== 绘图 ======
#     fig, axes = plt.subplots(3, 1, figsize=(14, 10))
#     fig.suptitle('纵向控制（速度跟踪）分析 (过滤后)', fontsize=16)

#     ax = axes[0]
#     ax.plot(time, target_speed, 'r--', linewidth=1.5, label='目标速度', alpha=0.8)
#     ax.plot(time, speed, 'b-', linewidth=1.5, label='实际车速', alpha=0.8)
#     ax.set_xlabel('时间 (s)')
#     ax.set_ylabel('速度 (m/s)')
#     ax.set_title(f'速度跟踪 | MAE={mae:.3f}, RMSE={rmse:.3f}, MaxErr={max_error:.3f}')
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc='best')

#     ax = axes[1]
#     ax.plot(time, error, 'g-', linewidth=1.0, alpha=0.7, label='速度误差 (目标 - 实际)')
#     ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
#     ax.axhline(mae, color='orange', linestyle='--', linewidth=0.8, label=f'MAE = {mae:.3f}')
#     ax.axhline(-mae, color='orange', linestyle='--', linewidth=0.8)
#     ax.set_xlabel('时间 (s)')
#     ax.set_ylabel('速度误差 (m/s)')
#     ax.set_title('速度误差')
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc='best')

#     ax = axes[2]
#     ax.plot(time, throttle, 'b-', linewidth=1.5, label='油门', alpha=0.8)
#     ax.plot(time, brake, 'r-', linewidth=1.5, label='刹车', alpha=0.8)
#     ax.set_xlabel('时间 (s)')
#     ax.set_ylabel('控制量')
#     ax.set_title(f'油门/刹车输出 (油门饱和: {saturation_ratio:.1f}%)')
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc='best')

#     plt.tight_layout()

#     save_dir = os.path.join(args.save_to_dir, 'velocity')
#     os.makedirs(save_dir, exist_ok=True)
#     base_name = os.path.splitext(os.path.basename(log_path))[0]
#     save_path = os.path.join(save_dir, f"{base_name}.png")
#     plt.savefig(save_path, dpi=150)
#     print(f"✅ 图片已保存: {save_path}")

#     plt.show()

#     # ====== 终端打印详细统计 ======
#     print("\n" + "=" * 70)
#     print("速度跟踪误差详细统计 (已过滤速度为0的点)")
#     print("=" * 70)
#     print(f"原始帧数: {len(df_raw)}")
#     print(f"有效帧数: {len(df)} (剔除速度 <= {speed_threshold} m/s 的点)")
#     print(f"MAE (平均绝对误差): {mae:.3f} m/s")
#     print(f"RMSE (均方根误差): {rmse:.3f} m/s")
#     print(f"标准差: {std_error:.3f} m/s")
#     print(f"最大绝对误差: {max_error:.3f} m/s")
#     print(f"误差 > {error_threshold} m/s 的比例: {large_error_ratio:.2f}%")
#     print(f"油门饱和 (>0.95) 比例: {saturation_ratio:.1f}%")

#     print("\n--- 误差最大的 5 个点 (按绝对误差排序) ---")
#     print(f"{'时间(s)':>10} | {'目标速度':>10} | {'实际速度':>10} | {'误差':>10} | {'油门':>8} | {'刹车':>8}")
#     print("-" * 75)
#     for i in top5_indices:
#         print(f"{time[i]:>10.3f} | {target_speed[i]:>10.2f} | {speed[i]:>10.2f} | {error[i]:>10.2f} | {throttle[i]:>8.3f} | {brake[i]:>8.3f}")
#     print("=" * 70)

#     # 诊断建议
#     if saturation_ratio > 30:
#         print("⚠️ 油门饱和严重，建议增大 PID 的 kp 或降低目标加速度")
#     elif mae > 1.0:
#         print("⚠️ 平均速度误差较大，建议增大 PID 的 kp")
#     else:
#         print("✅ 速度跟踪正常")


# if __name__ == "__main__":
#     main()

"""
车速跟踪对比分析工具
功能：分析实际车速 vs 目标车速 vs 专家车速
用法：
    默认：python compare_velocity.py
    指定日志：python compare_velocity.py logs/xxx.csv
"""
import os
import sys
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), "output")


def find_latest_log(log_dir='../logs'):
    log_dir = os.path.join(os.path.dirname(__file__), log_dir)
    files = glob.glob(os.path.join(log_dir, 'replay_control_*.csv'))
    if not files:
        raise FileNotFoundError(f"在 {log_dir} 中未找到日志文件")
    return max(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="车速对比分析")
    parser.add_argument("log_path", nargs='?', help="日志文件路径（可选）")
    parser.add_argument("--save_to_dir", "-s", type=str, default=DEFAULT_SAVE_DIR,
                        help=f"保存图片的目录（默认: {DEFAULT_SAVE_DIR}）")
    args = parser.parse_args()

    if args.log_path:
        log_path = args.log_path
    else:
        try:
            log_path = find_latest_log()
            print(f"自动选择最新日志: {log_path}")
        except FileNotFoundError as e:
            print(e)
            return

    df = pd.read_csv(log_path)
    print(f"✅ 加载日志: {log_path}，共 {len(df)} 帧")

    # 检查必需字段
    required_cols = ['speed', 'target_speed', 'throttle', 'brake']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠️ 日志缺少字段: {missing}，请确保 run_control_loop.py 记录了这些数据")
        return

    # 检查是否有 expert_speed 字段
    has_expert = 'expert_speed' in df.columns

    time = df['time'].values
    speed = df['speed'].values
    target_speed = df['target_speed'].values
    throttle = df['throttle'].values
    brake = df['brake'].values

    # ====== 绘图 ======
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('纵向控制分析', fontsize=16)

    # 图1：速度对比（三方）
    ax = axes[0]
    ax.plot(time, target_speed, 'r--', linewidth=1.5, label='目标速度 (target_speed)', alpha=0.8)
    ax.plot(time, speed, 'b-', linewidth=1.5, label='实际车速 (speed)', alpha=0.8)
    if has_expert:
        expert_speed = df['expert_speed'].values
        ax.plot(time, expert_speed, 'g-.', linewidth=1.5, label='专家速度 (expert_speed)', alpha=0.6)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度 (m/s)')
    ax.set_title('速度对比')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # 图2：速度误差（实际 vs 目标）
    ax = axes[1]
    error = target_speed - speed
    ax.plot(time, error, 'g-', linewidth=1.0, alpha=0.7, label='误差 (目标 - 实际)')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度误差 (m/s)')
    ax.set_title('速度跟踪误差 (实际 vs 目标)')
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

    # 保存
    save_dir = os.path.join(args.save_to_dir, 'velocity')
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(log_path))[0]
    save_path = os.path.join(save_dir, f"{base_name}.png")
    plt.savefig(save_path, dpi=150)
    print(f"✅ 图片已保存: {save_path}")

    plt.show()

    # 打印统计
    print("\n--- 速度统计 ---")
    print(f"目标速度均值: {np.mean(target_speed):.2f} m/s")
    print(f"实际速度均值: {np.mean(speed):.2f} m/s")
    if has_expert:
        print(f"专家速度均值: {np.mean(expert_speed):.2f} m/s")
        print(f"目标 vs 专家偏差: {np.mean(target_speed) - np.mean(expert_speed):.2f} m/s")
        print(f"实际 vs 专家偏差: {np.mean(speed) - np.mean(expert_speed):.2f} m/s")


if __name__ == "__main__":
    main()