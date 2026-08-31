# """
# 转向角对比分析工具
# 功能：对比控制器输出的 steer 与数据集中专家 steer 的差异
# 用法：
#     默认：python compare_steer.py
#     指定日志：python compare_steer.py logs/xxx.csv
#     指定输出目录：python compare_steer.py --save_to_dir ./my_analysis
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

# # 默认输出目录
# DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), "output")


# def find_latest_log(log_dir='../logs'):
#     log_dir = os.path.join(os.path.dirname(__file__), log_dir)
#     files = glob.glob(os.path.join(log_dir, 'replay_control_*.csv'))
#     if not files:
#         raise FileNotFoundError(f"在 {log_dir} 中未找到日志文件")
#     return max(files, key=os.path.getmtime)


# def main():
#     parser = argparse.ArgumentParser(description="转向角对比分析")
#     parser.add_argument("log_path", nargs='?', help="日志文件路径（可选，不指定则自动选最新）")
#     parser.add_argument("--save_to_dir", "-s", type=str, default=DEFAULT_SAVE_DIR,
#                         help=f"保存图片的目录（默认: {DEFAULT_SAVE_DIR}）")
#     args = parser.parse_args()

#     # 确定日志文件
#     if args.log_path:
#         log_path = args.log_path
#     else:
#         try:
#             log_path = find_latest_log()
#             print(f"自动选择最新日志: {log_path}")
#         except FileNotFoundError as e:
#             print(e)
#             return

#     df = pd.read_csv(log_path)
#     print(f"✅ 加载日志: {log_path}，共 {len(df)} 帧")

#     if 'expert_steer' not in df.columns:
#         print("⚠️ 日志中没有 'expert_steer' 字段，请确保 run_control_loop.py 已记录专家数据")
#         return

#     time = df['time'].values
#     steer = df['steer'].values
#     expert_steer = df['expert_steer'].values  # 不反转，直接使用原始数据
#     speed = df['speed'].values

#     # 计算误差（不反转）
#     error = steer - expert_steer
#     mae = np.mean(np.abs(error))
#     rmse = np.sqrt(np.mean(error**2))
#     max_error = np.max(np.abs(error))
#     std_error = np.std(error)

#     # 符号相反比例（不反转专家数据）
#     sign_diff_ratio = (np.sign(steer) != np.sign(expert_steer)).sum() / len(steer) * 100

#     # 误差超过阈值 0.1 的比例
#     error_threshold = 0.1
#     large_error_ratio = (np.abs(error) > error_threshold).sum() / len(error) * 100

#     # 找出误差最大的前 5 个点
#     sorted_indices = np.argsort(np.abs(error))[::-1]
#     top5_indices = sorted_indices[:5]

#     # ====== 绘图 ======
#     fig, axes = plt.subplots(3, 1, figsize=(14, 10))
#     fig.suptitle('控制器转向角 vs 专家转向角对比 (未反转专家数据)', fontsize=16)

#     ax = axes[0]
#     ax.plot(time, steer, 'b-', linewidth=1.5, label='控制器输出 steer', alpha=0.8)
#     ax.plot(time, expert_steer, 'r--', linewidth=1.5, label='专家 steer (原始数据)', alpha=0.8)
#     ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
#     ax.set_xlabel('时间 (s)')
#     ax.set_ylabel('转向角')
#     ax.set_title(f'转向角对比 | MAE={mae:.4f}, RMSE={rmse:.4f}, Max={max_error:.4f}')
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc='best')

#     ax = axes[1]
#     ax.plot(time, error, 'g-', linewidth=1.0, alpha=0.7, label='误差 (控制器 - 专家)')
#     ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
#     ax.axhline(mae, color='orange', linestyle='--', linewidth=0.8, label=f'MAE = {mae:.4f}')
#     ax.axhline(-mae, color='orange', linestyle='--', linewidth=0.8)
#     ax.set_xlabel('时间 (s)')
#     ax.set_ylabel('转向角误差')
#     ax.set_title('转向角误差')
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc='best')

#     ax = axes[2]
#     ax.plot(time, speed, 'b-', linewidth=1.5, label='实际车速', alpha=0.8)
#     if 'target_speed' in df.columns:
#         ax.plot(time, df['target_speed'].values, 'r--', linewidth=1.5, label='目标速度', alpha=0.8)
#     ax.set_xlabel('时间 (s)')
#     ax.set_ylabel('车速 (m/s)')
#     ax.set_title('车速跟踪')
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc='best')

#     plt.tight_layout()

#     # 保存
#     save_dir = os.path.join(args.save_to_dir, 'steer')
#     os.makedirs(save_dir, exist_ok=True)
#     base_name = os.path.splitext(os.path.basename(log_path))[0]
#     save_path = os.path.join(save_dir, f"{base_name}.png")
#     plt.savefig(save_path, dpi=150)
#     print(f"✅ 图片已保存: {save_path}")

#     plt.show()

#     # ====== 终端打印详细统计 ======
#     print("\n" + "=" * 70)
#     print("转向角误差详细统计 (未反转专家数据)")
#     print("=" * 70)
#     print(f"帧数: {len(df)}")
#     print(f"MAE (平均绝对误差): {mae:.4f}")
#     print(f"RMSE (均方根误差): {rmse:.4f}")
#     print(f"标准差: {std_error:.4f}")
#     print(f"最大绝对误差: {max_error:.4f}")
#     print(f"符号相反比例: {sign_diff_ratio:.2f}%")
#     print(f"误差 > {error_threshold} 的比例: {large_error_ratio:.2f}%")

#     print("\n--- 误差最大的 5 个点 (按绝对误差排序) ---")
#     print(f"{'时间(s)':>10} | {'控制器 steer':>12} | {'专家 steer':>12} | {'误差':>12} | {'车速(m/s)':>10}")
#     print("-" * 70)
#     for i in top5_indices:
#         print(f"{time[i]:>10.3f} | {steer[i]:>12.4f} | {expert_steer[i]:>12.4f} | {error[i]:>12.4f} | {speed[i]:>10.2f}")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()
"""
转向角对比分析工具
功能：对比控制器 steer 与专家 steer，同时对比速度与专家速度
用法：
    默认：python compare_steer.py
    指定日志：python compare_steer.py logs/xxx.csv
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
    parser = argparse.ArgumentParser(description="转向角对比分析")
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

    if 'expert_steer' not in df.columns:
        print("⚠️ 日志中没有 'expert_steer' 字段")
        return

    time = df['time'].values
    steer = df['steer'].values
    expert_steer = df['expert_steer'].values
    speed = df['speed'].values

    has_expert_speed = 'expert_speed' in df.columns

    # 计算误差
    error = steer - expert_steer
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))

    # ====== 绘图 ======
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('转向角对比分析', fontsize=16)

    # 图1：转向角对比
    ax = axes[0]
    ax.plot(time, steer, 'b-', linewidth=1.5, label='控制器 steer', alpha=0.8)
    ax.plot(time, expert_steer, 'r--', linewidth=1.5, label='专家 steer', alpha=0.8)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('转向角')
    ax.set_title(f'转向角对比 | MAE={mae:.4f}, RMSE={rmse:.4f}, Max={max_error:.4f}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # 图2：转向误差
    ax = axes[1]
    ax.plot(time, error, 'g-', linewidth=1.0, alpha=0.7, label='误差 (控制器 - 专家)')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('转向角误差')
    ax.set_title('转向角误差')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # 图3：速度对比（使用 expert_speed 作为基准）
    ax = axes[2]
    ax.plot(time, speed, 'b-', linewidth=1.5, label='实际车速', alpha=0.8)
    if has_expert_speed:
        expert_speed = df['expert_speed'].values
        ax.plot(time, expert_speed, 'g-.', linewidth=1.5, label='专家车速', alpha=0.6)
    elif 'target_speed' in df.columns:
        ax.plot(time, df['target_speed'].values, 'r--', linewidth=1.5, label='目标速度', alpha=0.6)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('车速 (m/s)')
    ax.set_title('车速对比')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()

    # 保存
    save_dir = os.path.join(args.save_to_dir, 'steer')
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(log_path))[0]
    save_path = os.path.join(save_dir, f"{base_name}.png")
    plt.savefig(save_path, dpi=150)
    print(f"✅ 图片已保存: {save_path}")

    plt.show()

    print("\n--- 转向统计 ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"最大误差: {max_error:.4f}")


if __name__ == "__main__":
    main()