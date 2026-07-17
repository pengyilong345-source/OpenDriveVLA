"""
转向角对比分析工具
功能：对比控制器输出的 steer 与数据集中专家 steer 的差异
用法：python compare_steer.py [日志文件路径]
"""
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def find_latest_log(log_dir='../logs'):
    """查找目录下最新的 CSV 日志文件"""
    log_dir = os.path.join(os.path.dirname(__file__), log_dir)
    files = glob.glob(os.path.join(log_dir, 'replay_control_*.csv'))
    if not files:
        raise FileNotFoundError(f"在 {log_dir} 中未找到日志文件")
    latest = max(files, key=os.path.getmtime)
    return latest


def main():
    # 确定日志文件
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        try:
            log_path = find_latest_log()
            print(f"自动选择最新日志: {log_path}")
        except FileNotFoundError as e:
            print(e)
            return

    # 读取日志
    df = pd.read_csv(log_path)
    print(f"✅ 加载日志: {log_path}")
    print(f"   总帧数: {len(df)}")

    # 检查是否有 expert_steer 字段
    if 'expert_steer' not in df.columns:
        print("⚠️ 日志中没有 'expert_steer' 字段，请确保 run_control_loop.py 已记录专家数据")
        return

    time = df['time'].values
    steer = df['steer'].values
    expert_steer = df['expert_steer'].values
    speed = df['speed'].values

    # 计算误差
    error = steer - expert_steer
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))

    # ====== 绘图 ======
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('控制器转向角 vs 专家转向角对比', fontsize=16)

    # 图1：转向角对比
    ax = axes[0]
    ax.plot(time, steer, 'b-', linewidth=1.5, label='控制器输出 steer', alpha=0.8)
    ax.plot(time, expert_steer, 'r--', linewidth=1.5, label='专家 steer (来自数据集)', alpha=0.8)
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
    ax.axhline(mae, color='orange', linestyle='--', linewidth=0.8, label=f'MAE = {mae:.4f}')
    ax.axhline(-mae, color='orange', linestyle='--', linewidth=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('转向角误差')
    ax.set_title('转向角误差')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # 图3：车速对比
    ax = axes[2]
    ax.plot(time, speed, 'b-', linewidth=1.5, label='实际车速', alpha=0.8)
    # 尝试读取目标速度
    if 'target_speed' in df.columns:
        ax.plot(time, df['target_speed'].values, 'r--', linewidth=1.5, label='目标速度', alpha=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('车速 (m/s)')
    ax.set_title('车速跟踪')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

    # ====== 打印统计 ======
    print("\n--- 统计信息 ---")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"最大误差: {max_error:.4f}")
    print(f"转向角符号相反的比例: {(np.sign(steer) != np.sign(expert_steer)).sum() / len(steer) * 100:.2f}%")


if __name__ == "__main__":
    main()