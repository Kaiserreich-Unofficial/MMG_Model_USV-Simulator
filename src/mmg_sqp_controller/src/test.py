from settings import Acados_Settings
from model import HydroParams
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import numpy as np

horizon = 100
dt = 0.05
Tf = 200
params = HydroParams()
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
t = 0
Q = np.diag([10.0, 10.0, 100.0, 1.0, 1.0, 1.0])
R = np.diag([0.1, 0.1])
settings = Acados_Settings(params, 20.0, x0, horizon, dt, Q, R)
model = settings.model
solver = settings.acados_solver


# 设置图形和子图布局
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(6, 3, figure=fig)

# 初始化子图1（轨迹）
ax1 = fig.add_subplot(gs[0:, 0])
line1, = ax1.plot([], [], 'b-', lw=0.5)
line2, = ax1.plot([], [], 'r-', lw=3)
# ax1.set_xlim(-radius-2, radius+2+4)
# ax1.set_ylim(-2, 2*radius+2)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-10, 1)
ax1.set_title('Trajectory')
ax1.set_aspect('equal')

# 初始化子图2（速度 u）
ax2 = fig.add_subplot(gs[0:2, 1])
line3, = ax2.plot([], [], 'b-')
ax2.set_xlim(0, Tf)
ax2.set_ylim(0, 0.5)
ax2.set_title('u')

# 初始化子图3（速度 v）
ax3 = fig.add_subplot(gs[2:4, 1])
line4, = ax3.plot([], [], 'g-')
ax3.set_xlim(0, Tf)
ax3.set_ylim(-0.1, 0.1)
ax3.set_title('v')

# 初始化子图4（速度 r）
ax4 = fig.add_subplot(gs[4:6, 1])
line5, = ax4.plot([], [], 'm-')
line_r, = ax4.plot([], [], 'g-', lw=0.5)
ax4.set_xlim(0, Tf)
ax4.set_ylim(-0.5, 0.5)
ax4.set_title('r')

# 初始化子图5（推力输入1）
ax5 = fig.add_subplot(gs[0:3, 2])
line6, = ax5.plot([], [], 'c-')
ax5.set_xlim(0, Tf)
ax5.set_ylim(-0.5, 0.5)
ax5.set_title('Thrust_left')

# 初始化子图6（推力输入2）
ax6 = fig.add_subplot(gs[3:, 2])
line7, = ax6.plot([], [], 'y-')
ax6.set_xlim(0, Tf)
ax6.set_ylim(-0.5, 0.5)
ax6.set_title('Thrust_right')

u_data, v_data, r_data = [], [], []
x_data, y_data = [], []
Tl_data, Tr_data = [], []
r_body_data = []

def get_ref_traj(t: float) -> np.ndarray:
    # 八字形轨迹
    a = 3.2
    b = 4.8
    x = a * np.sin(0.2*t)
    y = b * np.cos(0.1*t) - 4

    # 一阶导数
    x_dot = a * 0.2 * np.cos(0.2 * t)
    y_dot = -b * 0.1 * np.sin(0.1 * t)
    psi = np.arctan2(y_dot, x_dot)

    # 二阶导数
    x_dot_dot = -a * 0.2 * 0.2 * np.sin(0.2 * t)
    y_dot_dot = -b * 0.1 * 0.1 * np.cos(0.1 * t)
    psi_dot = (x_dot * y_dot_dot - y_dot * x_dot_dot) / \
        (x_dot ** 2 + y_dot ** 2)

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    u = cos_psi * x_dot + sin_psi * y_dot
    v = 0
    r = psi_dot

    return np.array([x, y, psi, u, v, r])


if __name__ == '__main__':
    print("Hello")
