import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib as mpl
from scipy.signal import savgol_filter
mpl.rcParams['font.size'] = 20
# ------------------------
# 1. 定义常数和函数
# ------------------------
l1 = 14.95e-3
l2 = 13.49e-3
l3 = 13.75e-3
l4 = 30e-3
l5 = 10e-3
d1 = 4.89e-3
d2 = 0.03      # 米
d3 = 55e-3     # 米
d4 = 3.75e-3   # 米
d5 = 150e-3    # 米
K  = 1.57e-3   # Nm/rad
F  = 3         # N
g  = 9.8
# 质量和转动惯量
m_1 = 0.000340301601  
m_2 = 2e-3            
coef = 0.0000159115937
I = 0.0000039784212
coef1 = 1.90946884e-4

def delta(theta):
    arg = (l1 * np.cos(theta) - l3 + d1) / l2
    return np.arcsin(np.clip(arg, -1, 1))

def alpha(theta):
    return np.arccos((l1 + d1 - l3) / l2) - np.pi/2 + theta + delta(theta)

def beta(theta):
    return (np.pi/2 - delta(theta)) - np.arccos((l1 + d1 - l3) / l2)

def x(theta):
    return (l2 * np.sin(np.arccos((l1 + d1 - l3) / l2))
            - l2 * np.cos(delta(theta)) + l1 * np.sin(theta))

def psi(theta):
    arg = (l4**2 + l5**2 - (d2 + x(theta))**2) / (2 * l4 * l5)
    return np.arccos(np.clip(arg, -1, 1))

def phi(theta):
    arg = (l5 * np.sin(psi(theta))) / (d2 + x(theta))
    return np.arcsin(np.clip(arg, -1, 1))

def F1(theta):
    return F * np.sin(psi(theta)) * np.cos(phi(theta))


def ode_system_1(t, y):
    theta, theta_dot = y
    d_val = delta(theta)
    a_val = alpha(theta)
    b_val = beta(theta)
    F1_val = F1(theta)
    dtheta = theta_dot
    dtheta_dot = - (
        F1_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val))
        + K * (theta + a_val + b_val)
        - coef * theta_dot**2
        + m_1 * np.cos(theta)
    ) / I
    return [dtheta, dtheta_dot]

def ode_system_2(t, y):
    theta, theta_dot = y
    d_val = delta(theta)
    a_val = alpha(theta)
    b_val = beta(theta)
    F1_val = F1(theta)
    dtheta = theta_dot
    dtheta_dot = (
        F1_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val))
        - K * (theta + a_val + b_val)
        - coef * theta_dot**2
        - m_1 * np.cos(theta)
    ) / I
    return [dtheta, dtheta_dot]

def simulate(tf1, tf2, y0_1, y0_2):
    t_total_list = []
    theta_total_list = []
    current_offset = 0.0
    while current_offset < 0.2:
        # 第一段积分
        t_eval1 = np.linspace(0, tf1, 1000)
        sol1 = solve_ivp(ode_system_1, [0, tf1], y0_1, t_eval=t_eval1)
        t1 = sol1.t + current_offset
        theta1 = sol1.y[0]
        t_total_list.append(t1)
        theta_total_list.append(theta1)
        current_offset = t1[-1]
        if current_offset >= 1.0:
            break
        # 第二段积分
        t_eval2 = np.linspace(0, tf2, 1000)
        sol2 = solve_ivp(ode_system_2, [0, tf2], y0_2, t_eval=t_eval2)
        t2 = sol2.t + current_offset
        theta2 = sol2.y[0]
        t_total_list.append(t2)
        theta_total_list.append(theta2)
        current_offset = t2[-1]
    t_total = np.concatenate(t_total_list)
    theta_total = np.concatenate(theta_total_list)
    mask = t_total <= 0.2
    t_total = t_total[mask]
    theta_total = theta_total[mask]
    return t_total, theta_total


F = 4
tf1_4 = 0.05390530482938438
tf2_4 = 0.05548620566389552
y0_1_4 = [0.8704004449728826, 0]
y0_2_4 = [-0.6391148802476694, 0]
t_total4, theta_total4 = simulate(tf1_4, tf2_4, y0_1_4, y0_2_4)

window_length = 11  # 必须为奇数
polyorder = 3
theta_4_smooth = savgol_filter(theta_total4, window_length, polyorder)
# 第二组参数（例如 F = 5 N）
F = 3
tf1_3 = 0.060526589179550463
tf2_3 = 0.06358769972107282 
y0_1_3 = [0.8704004449728826, 0]
y0_2_3 = [-0.6391148802476694, 0]
t_total3, theta_total3 = simulate(tf1_3, tf2_3, y0_1_3, y0_2_3)

window_length = 11  # 必须为奇数
polyorder = 3
theta_3_smooth = savgol_filter(theta_total3, window_length, polyorder)

F = 2
tf1_2 = 0.07097998379966579
tf2_2 = 0.07756925882244137
y0_1_2 = [0.8704004449728826, 0]
y0_2_2 = [-0.6391148802476694, 0]
t_total2, theta_total2 = simulate(tf1_2, tf2_2, y0_1_2, y0_2_2)

window_length = 11  # 必须为奇数
polyorder = 3
theta_2_smooth = savgol_filter(theta_total2, window_length, polyorder)

plt.figure(figsize=(15, 5))
plt.plot(t_total4, theta_4_smooth, color='darkblue', lw=3, label='M = 0.04 N·m')
plt.plot(t_total3, theta_3_smooth, color='blue', lw=3, label='M = 0.03 N·m')
plt.plot(t_total2, theta_2_smooth, color='lightblue', lw=3, label='M = 0.02 N·m')
plt.xlabel('t [s]')
plt.ylabel(r'$\theta_A$ [rad]')

ax = plt.gca()
ylim = ax.get_ylim()
max_abs = max(abs(ylim[0]), abs(ylim[1]))
ax.set_ylim(-max_abs, max_abs)

plt.legend()
plt.grid(True)

fig = plt.gcf()
fig.text(0.01, 0.99, '(a)', transform=fig.transFigure,
         fontsize=20, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.show()