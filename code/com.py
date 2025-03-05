import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

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
K  = 1.57e-3   # Nm/rad
F  = 3         # N

# 质量和转动惯量
m_1 = 0.000340301601  # 用于方程中的某项 (示例)
m_2 = 1e-3            # 用于计算 F2 的等效质量
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
# 2. 定义两个 ODE 系统
# ------------------------
def ode_system_1(t, y):
    theta, theta_dot = y
    d_val = delta(theta)
    a_val = alpha(theta)
    b_val = beta(theta)
    F1_val = F1(theta)
    dtheta = theta_dot
    dtheta_dot = - (F1_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val)) + K * (theta + a_val + b_val) - coef * theta_dot**2 + m_1*np.cos(theta)) / I
    return [dtheta, dtheta_dot]

def ode_system_2(t, y):
    theta, theta_dot = y
    d_val = delta(theta)
    a_val = alpha(theta)
    b_val = beta(theta)
    F1_val = F1(theta)
    dtheta = theta_dot
    dtheta_dot = (F1_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val)) - K * (theta + a_val + b_val) - coef * theta_dot**2 - m_1*np.cos(theta)) / I
    return [dtheta, dtheta_dot]

# ------------------------
# 3. 定义一个模拟函数
# ------------------------
def simulate(tf1, tf2, y0_1, y0_2):
    t_total_list = []
    theta_total_list = []
    current_offset = 0.0
    while current_offset < 0.3:
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
    mask = t_total <= 0.3
    t_total = t_total[mask]
    theta_total = theta_total[mask]
    return t_total, theta_total

# ------------------------
# 4. 分别计算两组参数下的模拟结果
# ------------------------

# 第一组参数（例如 F = 8 N）
F = 4
tf1_8 = 0.038396140728502344
tf2_8 = 0.036296671122534714
y0_1_8 = [0.8704004449728826, 0]
y0_2_8 = [-0.6391148802476694, 0]
t_total8, theta_total8 = simulate(tf1_8, tf2_8, y0_1_8, y0_2_8)

# 第二组参数（例如 F = 5 N）
F = 3
tf1_5 = 0.04007961513638783 
tf2_5 = 0.04074981178876041 
y0_1_5 = [0.8704004449728826, 0]
y0_2_5 = [-0.6391148802476694, 0]
t_total5, theta_total5 = simulate(tf1_5, tf2_5, y0_1_5, y0_2_5)

F = 2
tf1_2 = 0.048153459037466734
tf2_2 = 0.04779809658896103
y0_1_2 = [0.8704004449728826, 0]
y0_2_2 = [-0.6391148802476694, 0]
t_total2, theta_total2 = simulate(tf1_2, tf2_2, y0_1_2, y0_2_2)
# ------------------------
# 5. 在同一张图中绘制两条 θ(t) 曲线
# ------------------------
plt.figure(figsize=(16, 5))
plt.plot(t_total8, theta_total8, color='darkblue', lw=2, label='M = 0.04 Nm')
plt.plot(t_total5, theta_total5, color='blue', lw=2, label='M = 0.03 Nm')
plt.plot(t_total2, theta_total2, color='lightblue', lw=2, label='M = 0.02 Nm')
plt.xlabel('Time [s]')
plt.ylabel(r'$\theta$ [rad]')

ax = plt.gca()
ylim = ax.get_ylim()
max_abs = max(abs(ylim[0]), abs(ylim[1]))
ax.set_ylim(-max_abs, max_abs)

plt.legend()
plt.grid(True)
plt.show()