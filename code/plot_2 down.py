import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# ------------------------
# 定义常数（SI 单位）
# ------------------------
l1 = 14.95e-3
l2 = 13.49e-3
l3 = 13.75e-3    # 米
l4 = 30e-3     # 米
l5 = 10e-3     # 米
d1 = 4.89e-3
d2 = 0.03   # 米
d3 = 55e-3
d4 = 3.75e-3
d5 = 120e-3
K  = 0  # Nm/rad
F  = 3   # N
m_1 = 0.000340301601
# 系数（对应于方程中 0.00002619169，但这里用 coef 变量）
coef = 0.0000159115937
I = 0.0000039784212

# 假设质量 m，若公式需要（用户可根据实际情况设置）
m_2 = 1e-3

# ----------------------------
# 定义各个角度及相关函数
# ----------------------------
def delta(theta):
    arg = (l1 * np.cos(theta) - l3 + d1) / l2
    arg = np.clip(arg, -1, 1)
    return np.arcsin(arg)

def alpha(theta):
    return np.arccos((l1 + d1 - l3) / l2) - np.pi/2 + theta + delta(theta)

def beta(theta):
    return (np.pi/2 - delta(theta)) - np.arccos((l1 + d1 - l3) / l2)

def x(theta):
    return (l2 * np.sin(np.arccos((l1 + d1 - l3) / l2))
            - l2 * np.cos(delta(theta)) + l1 * np.sin(theta))

def psi(theta):
    arg = (l4**2 + l5**2 - (d2 + x(theta))**2) / (2 * l4 * l5)
    arg = np.clip(arg, -1, 1)
    return np.arccos(arg)

def phi(theta):
    arg = (l5 * np.sin(psi(theta))) / (d2 + x(theta))
    arg = np.clip(arg, -1, 1)
    return np.arcsin(arg)

def F1(theta):
    return F * np.sin(psi(theta)) * np.cos(phi(theta))

# ----------------------------------------------------------
# 定义 ODE 系统
# 设 y[0] = theta, y[1] = theta_dot
# ----------------------------------------------------------
def ode_system(t, y):
    theta, theta_dot = y
    d_val = delta(theta)
    a_val = alpha(theta)
    b_val = beta(theta)
    F1_val = F1(theta)

    dtheta = theta_dot
    dtheta_dot = (F1_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val))
                  - K * (theta + a_val + b_val)
                  - coef * theta_dot**2 - m_1*np.cos(theta))  / I
    return [dtheta, dtheta_dot]

# -------------------------------
# 数值积分：使用 solve_ivp 求解
# -------------------------------
t0 = 0.0
tf = 0.07366674829640933
t_eval = np.linspace(t0, tf, 1000)

# 初始条件
y0 = [-0.639 , 0]
sol = solve_ivp(ode_system, [t0, tf], y0, t_eval=t_eval)
t_array = sol.t
theta_array = sol.y[0]

# -------------------------------
# 计算其他函数值
# -------------------------------
delta_array = np.array([delta(th) for th in theta_array])
alpha_array = np.array([alpha(th) for th in theta_array])
beta_array  = np.array([beta(th)  for th in theta_array])
F1_array    = np.array([F1(th)    for th in theta_array])
psi_array   = np.array([psi(th)   for th in theta_array])
phi_array   = np.array([phi(th)   for th in theta_array])
x_array     = np.array([x(th)     for th in theta_array])

# -------------------------------
# (1) 根据公式计算 θ_D(t) 及其二阶导数
# -------------------------------
thetaD_array = -np.arctan((x_array - d4) / d3)

# 一阶导数 dθD/dt
thetaD_dot_array = np.gradient(thetaD_array, t_array)

# 二阶导数 d²θD/dt²
thetaD_ddot_array = np.gradient(thetaD_dot_array, t_array)

# -------------------------------
# (2) 根据公式计算 F2(t)
#     F2 = (d5^2 * d3 * m * ddθD) / ((x - d4)^2 + d3^2)
# -------------------------------
F2_array = - (d5**2 * d3 * m_2 * thetaD_ddot_array) / (x_array **2 + d3**2)

# -------------------------------
# (3) 寻找 x(t) = 0.01 的交点（如有需要）
# -------------------------------
cross_indices = np.where(np.diff(np.sign(x_array - 0.01)) != 0)[0]
cross_times = []
for idx in cross_indices:
    t1, t2 = t_array[idx], t_array[idx+1]
    x1, x2 = x_array[idx], x_array[idx+1]
    # 线性插值计算过 0.01 点时间
    t_cross = t1 + (0.01 - x1) * (t2 - t1) / (x2 - x1)
    cross_times.append(t_cross)
print("x = 0.01 crossing times:", cross_times)

# -------------------------------
# (4) 绘制图像
#     2行×4列子图: axs[0..7]
#     将 F2(t) 放在 axs[1]
# -------------------------------
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
axs = axs.flatten()

# θ(t)
axs[0].plot(t_array, theta_array, 'b-')
axs[0].set_title(r'$\theta(t)$')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel(r'$\theta$ [rad]')
axs[0].grid(True)

# F2(t) 新增
axs[1].plot(t_array, F2_array, 'r-')
axs[1].set_title(r'$F_{2}(t)$')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel(r'$F_{2}$ [N?]')
axs[1].grid(True)

# δ(t)
axs[2].plot(t_array, delta_array, 'g-')
axs[2].set_title(r'$\delta(t)$')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel(r'$\delta$ [rad]')
axs[2].grid(True)

# α(t) & β(t)
axs[3].plot(t_array, alpha_array, 'm-', label=r'$\alpha(t)$')
axs[3].plot(t_array, beta_array, 'b-', label=r'$\beta(t)$')
axs[3].set_title(r'$\alpha(t)$ and $\beta(t)$')
axs[3].set_xlabel('Time [s]')
axs[3].set_ylabel('Angle [rad]')
axs[3].legend()
axs[3].grid(True)

# F1(t)
axs[4].plot(t_array, F1_array, 'c-')
axs[4].set_title(r'$F_1(t)$')
axs[4].set_xlabel('Time [s]')
axs[4].set_ylabel(r'$F_1$ [N]')
axs[4].grid(True)

# ψ(t)
axs[5].plot(t_array, psi_array, 'k-')
axs[5].set_title(r'$\psi(t)$')
axs[5].set_xlabel('Time [s]')
axs[5].set_ylabel(r'$\psi$ [rad]')
axs[5].grid(True)

# φ(t)
axs[6].plot(t_array, phi_array, 'k-')
axs[6].set_title(r'$\phi(t)$')
axs[6].set_xlabel('Time [s]')
axs[6].set_ylabel(r'$\phi$ [rad]')
axs[6].grid(True)

# x(t)
axs[7].plot(t_array, x_array, 'k-')
axs[7].set_title(r'$x(t)$')
axs[7].set_xlabel('Time [s]')
axs[7].set_ylabel(r'$x$ [m]')
axs[7].grid(True)

plt.tight_layout()
plt.show()