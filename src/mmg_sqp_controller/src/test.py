#!/usr/bin/env python3
import os
import sys

# 将当前文件所在目录加入 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import HydroParams, Acados_Settings
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec


# === 参数设定 ===
horizon = 100
dt = 0.05
Tf = 200
params = HydroParams()
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x, y, psi, u, v, r
t = 0.0
Q = np.diag([1.0, 1.0, 10.0, 0.0, 0.0, 0.0])
R = np.diag([0.0, 0.0])
settings = Acados_Settings(params, 20.0, x0, horizon, dt, Q, R)
model = settings.model
solver = settings.acados_solver
Nsim = int(Tf / dt)

# === 图形布局 ===
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(6, 3, figure=fig)

# 轨迹图
ax1 = fig.add_subplot(gs[0:, 0])
line1, = ax1.plot([], [], 'b-', lw=0.5)  # 实际轨迹
line2, = ax1.plot([], [], 'r-', lw=2.0)  # 预测轨迹
ref_point, = ax1.plot([], [], 'ko', markersize=3)  # 当前参考点
ax1.set_xlim(-4, 4)
ax1.set_ylim(-10, 1)
ax1.set_title('Trajectory')
ax1.set_aspect('equal')

# 速度 u
ax2 = fig.add_subplot(gs[0:2, 1])
line3, = ax2.plot([], [], 'b-')
ax2.set_xlim(0, Tf)
ax2.set_ylim(-2, 2)
ax2.set_title('u')

# 速度 v
ax3 = fig.add_subplot(gs[2:4, 1])
line4, = ax3.plot([], [], 'g-')
ax3.set_xlim(0, Tf)
ax3.set_ylim(-1, 1)
ax3.set_title('v')

# 速度 r
ax4 = fig.add_subplot(gs[4:6, 1])
line5, = ax4.plot([], [], 'm-')
ax4.set_xlim(0, Tf)
ax4.set_ylim(-2, 2)
ax4.set_title('r')

# 推力左
ax5 = fig.add_subplot(gs[0:3, 2])
line6, = ax5.plot([], [], 'c-')
ax5.set_xlim(0, Tf)
ax5.set_ylim(-20., 20.)
ax5.set_title('Thrust_left')

# 推力右
ax6 = fig.add_subplot(gs[3:, 2])
line7, = ax6.plot([], [], 'y-')
ax6.set_xlim(0, Tf)
ax6.set_ylim(-20., 20.)
ax6.set_title('Thrust_right')

# === 数据缓存 ===
x_data, y_data = [], []
u_data, v_data, r_data = [], [], []
Tl_data, Tr_data = [], []
solve_time = 0.0

# === 参考轨迹函数 ===


def get_ref_traj(t: float) -> np.ndarray:
    a = 3.2
    b = 4.8
    x = a * np.sin(0.2 * t)
    y = b * np.cos(0.1 * t) - 4

    x_dot = a * 0.2 * np.cos(0.2 * t)
    y_dot = -b * 0.1 * np.sin(0.1 * t)
    psi = np.arctan2(y_dot, x_dot)

    x_dot_dot = -a * 0.2 * 0.2 * np.sin(0.2 * t)
    y_dot_dot = -b * 0.1 * 0.1 * np.cos(0.1 * t)
    psi_dot = (x_dot * y_dot_dot - y_dot * x_dot_dot) / \
        (x_dot ** 2 + y_dot ** 2)

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    u = cos_psi * x_dot + sin_psi * y_dot
    v = 0.0
    r = psi_dot
    Tl = 0
    Tr = 0

    return np.array([x, y, psi, u, v, r, Tl, Tr])

# === 动画更新函数 ===


def update(frame):
    global x0, t, solve_time

    # 设置状态约束
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    # 设置参考轨迹 yref
    for j in range(horizon):
        yref = get_ref_traj(t + j * dt)
        solver.set(j, "yref", yref)
    yref_N = get_ref_traj(t + horizon * dt)[:6]
    solver.set(horizon, "yref", yref_N)

    # MPC 求解
    t0 = time.time()
    status = solver.solve()
    elapsed = time.time() - t0
    solve_time += elapsed
    print(
        f"[Frame {frame}] status: {status}, time: {elapsed:.4f}, avg: {solve_time/(frame+1):.4f}")

    # 获取控制输入 & 下一状态
    u0 = solver.get(0, "u")
    print(u0)
    x1 = solver.get(1, "x")
    print(x1)

    # 存储轨迹
    x_data.append(x1[0])
    y_data.append(x1[1])
    u_data.append(x1[3])
    v_data.append(x1[4])
    r_data.append(x1[5])
    Tl_data.append(u0[0])
    Tr_data.append(u0[1])

    # 预测轨迹
    x_pred, y_pred = [], []
    for j in range(horizon):
        xj = solver.get(j, "x")
        x_pred.append(xj[0])
        y_pred.append(xj[1])
    line2.set_data(x_pred, y_pred)

    # 当前参考点
    ref_now = get_ref_traj(t)
    ref_point.set_data(ref_now[0], ref_now[1])

    # 更新当前状态与时间
    x0 = x1
    t += dt

    # 更新绘图数据
    line1.set_data(x_data, y_data)
    line3.set_data(np.linspace(0, t, len(u_data)), u_data)
    line4.set_data(np.linspace(0, t, len(v_data)), v_data)
    line5.set_data(np.linspace(0, t, len(r_data)), r_data)
    line6.set_data(np.linspace(0, t, len(Tl_data)), Tl_data)
    line7.set_data(np.linspace(0, t, len(Tr_data)), Tr_data)

    return line1, line2, line3, line4, line5, line6, line7, ref_point


# === 启动动画 ===
ani = animation.FuncAnimation(
    fig, update, frames=Nsim, blit=True, interval=10, repeat=False)
plt.tight_layout()
plt.show()
