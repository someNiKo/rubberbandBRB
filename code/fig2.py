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
d5 = 120e-3    # 米
F  = 3         # N

# 质量、转动惯量和系数
m_1 = 0.000340301601
coef = 0.0000159115937
I = 0.0000039784212

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

# ------------------------
# 2. 定义 ODE 系统
# ------------------------
def ode_system_1(t, y):
    """
    第一段方程
    """
    theta, theta_dot = y
    d_val = delta(theta)
    a_val = alpha(theta)
    b_val = beta(theta)
    F1_val = F1(theta)
    dtheta = theta_dot
    dtheta_dot = - (F1_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val))
                    + K * (theta + a_val + b_val)
                    - coef * theta_dot**2
                    + m_1 * np.cos(theta)) / I
    return [dtheta, dtheta_dot]

def ode_system_2(t, y):
    """
    第二段方程
    """
    theta, theta_dot = y
    d_val = delta(theta)
    a_val = alpha(theta)
    b_val = beta(theta)
    F1_val = F1(theta)
    dtheta = theta_dot
    dtheta_dot = (F1_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val))
                  - K * (theta + a_val + b_val)
                  - coef * theta_dot**2
                  - m_1 * np.cos(theta)) / I
    return [dtheta, dtheta_dot]

# ------------------------
# 3. 定义模拟函数（将两段积分拼接）
# ------------------------
def simulate(tf1, tf2, y0_1, y0_2, t_final=0.2):
    t_total_list = []
    theta_total_list = []
    current_offset = 0.0
    while current_offset < t_final:
        t_eval1 = np.linspace(0, tf1, 1000)
        sol1 = solve_ivp(ode_system_1, [0, tf1], y0_1, t_eval=t_eval1)
        t1 = sol1.t + current_offset
        theta1 = sol1.y[0]
        t_total_list.append(t1)
        theta_total_list.append(theta1)
        current_offset = t1[-1]
        if current_offset >= t_final:
            break
        t_eval2 = np.linspace(0, tf2, 1000)
        sol2 = solve_ivp(ode_system_2, [0, tf2], y0_2, t_eval=t_eval2)
        t2 = sol2.t + current_offset
        theta2 = sol2.y[0]
        t_total_list.append(t2)
        theta_total_list.append(theta2)
        current_offset = t2[-1]
    t_total = np.concatenate(t_total_list)
    theta_total = np.concatenate(theta_total_list)
    mask = t_total <= t_final
    return t_total[mask], theta_total[mask]

# ------------------------
# 4. F = 3 情况下两种 K 参数的模拟
# ------------------------

# 情况 1: K = 1.57e-3
K = 1.57e-3
tf1_case1 = 0.060526589179550463
tf2_case1 = 0.06358769972107282
y0_1_case1 = [0.8704004449728826, 0]
y0_2_case1 = [-0.6391148802476694, 0]
t_total_case1, theta_total_case1 = simulate(tf1_case1, tf2_case1, y0_1_case1, y0_2_case1)

window_length = 11  # 必须为奇数
polyorder = 3
theta_1_smooth = savgol_filter(theta_total_case1, window_length, polyorder)
# 情况 2: K = 0
K = 0
tf1_case2 = 0.07136425233758165
tf2_case2 = 0.07366674829640933
y0_1_case2 = [0.8704004449728826, 0]
y0_2_case2 = [-0.639, 0]
t_total_case2, theta_total_case2 = simulate(tf1_case2, tf2_case2, y0_1_case2, y0_2_case2)

window_length = 11  # 必须为奇数
polyorder = 3
theta_2_smooth = savgol_filter(theta_total_case2, window_length, polyorder)
# ------------------------
# 5. 绘图：在同一张图中比较两种情况
# ------------------------
plt.figure(figsize=(15, 5))
plt.plot(t_total_case1, theta_1_smooth, color='blue', lw=3, label='K = 1.57mN·m/rad')
plt.plot(t_total_case2, theta_2_smooth, color='blue', linestyle='--',lw=3, label='K = 0')
plt.xlabel('t [s]')
plt.ylabel(r'$\theta_A$ [rad]')
# 使 y 轴关于 0 对称
ax = plt.gca()
ylim = ax.get_ylim()
max_abs = max(abs(ylim[0]), abs(ylim[1]))
ax.set_ylim(-max_abs, max_abs)
plt.legend()
plt.grid(True)

fig = plt.gcf()
fig.text(0.01, 0.99, '(b)', transform=fig.transFigure,
         fontsize=20, fontweight='bold', va='top', ha='left')
plt.tight_layout()  # 自动调整布局
plt.savefig('output.png', bbox_inches='tight') 
plt.show()