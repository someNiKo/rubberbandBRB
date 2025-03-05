import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import matplotlib as mpl
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
g = 9.81       # 重力加速度

# 质量和转动惯量
m_1 = 0.000340301601  
m_2 = 2e-3            
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
    dtheta_dot = - ( F1_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val))
                     + K * (theta + a_val + b_val)
                     - coef * theta_dot**2
                     + m_1 * np.cos(theta) ) / I
    return [dtheta, dtheta_dot]

def ode_system_2(t, y):
    theta, theta_dot = y
    d_val = delta(theta)
    a_val = alpha(theta)
    b_val = beta(theta)
    F1_val = F1(theta)
    dtheta = theta_dot
    dtheta_dot = ( F1_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val))
                    - K * (theta + a_val + b_val)
                    - coef * theta_dot**2
                    - m_1 * np.cos(theta) ) / I
    return [dtheta, dtheta_dot]

# ------------------------
# 3. 第一段积分
# ------------------------
t0_1, tf_1 = 0.0, 0.060526589179550463
t_eval_1 = np.linspace(t0_1, tf_1, 1000)
y0_1 = [0.8704004449728826, 0]

sol_1 = solve_ivp(ode_system_1, [t0_1, tf_1], y0_1, t_eval=t_eval_1)
t1 = sol_1.t
theta1 = sol_1.y[0]

delta1 = np.array([delta(th) for th in theta1])
alpha1 = np.array([alpha(th) for th in theta1])
beta1  = np.array([beta(th) for th in theta1])
F1_1   = np.array([-F1(th) for th in theta1])  # 第一段取负
psi1   = np.array([psi(th) for th in theta1])
phi1   = np.array([phi(th) for th in theta1])
x1     = np.array([x(th) for th in theta1])

# ------------------------
# 4. 第二段积分
# ------------------------
t0_2, tf_2 = 0.0, 0.06358769972107282
t_eval_2 = np.linspace(t0_2, tf_2, 1000)
y0_2 = [-0.6391148802476694, 0]

sol_2 = solve_ivp(ode_system_2, [t0_2, tf_2], y0_2, t_eval=t_eval_2)
t2 = sol_2.t
theta2 = sol_2.y[0]

delta2 = np.array([delta(th) for th in theta2])
alpha2 = np.array([alpha(th) for th in theta2])
beta2  = np.array([beta(th) for th in theta2])
F1_2   = np.array([F1(th) for th in theta2])   # 第二段取正
psi2   = np.array([psi(th) for th in theta2])
phi2   = np.array([phi(th) for th in theta2])
x2     = np.array([x(th) for th in theta2])

# ------------------------
# 5. 拼接两段数据
# ------------------------
time_offset = t1[-1]
t2_shifted = t2 + time_offset

t_theta  = np.concatenate((t1, t2_shifted))
theta_12 = np.concatenate((theta1, theta2))

delta_12 = np.concatenate((delta1, delta2))
alpha_12 = np.concatenate((alpha1, alpha2))
beta_12  = np.concatenate((beta1, beta2))
F1_12    = np.concatenate((F1_1, F1_2))
psi_12   = np.concatenate((psi1, psi2))
phi_12   = np.concatenate((phi1, phi2))
x_12     = np.concatenate((x1, x2))

# ------------------------
# 6. 对整段θ数据进行平滑处理，再计算平滑后的 x
# ------------------------
window_length = 11  # 必须为奇数
polyorder = 3
theta_12_smooth = savgol_filter(theta_12, window_length, polyorder)
x_smooth = x(theta_12_smooth)


# ------------------------
# 7. 计算平滑后的 θD 及其导数（对完整数据求导后拆分）
#    θD = - arctan((x - d4) / d3)
# ------------------------
# ------------------------
thetaD_1 = -np.arctan((x1 - d4) / d3)
thetaD_2 = -np.arctan((x2 - d4) / d3)

thetaD_dot_1 = np.gradient(thetaD_1, t1)
thetaD_dot_2 = np.gradient(thetaD_2, t2)
thetaD_ddot_1 = np.gradient(thetaD_dot_1, t1)
thetaD_ddot_2 = np.gradient(thetaD_dot_2, t2)

thetaD_12 = np.concatenate((thetaD_1, thetaD_2))
thetaD_dot_12 = np.concatenate((thetaD_dot_1, thetaD_dot_2))
thetaD_ddot_12 = np.concatenate((thetaD_ddot_1, thetaD_ddot_2))

F2_1 = - (m_2 * d3 * d5) / ((x1 - d4)**2 + d3**2) * (
    d5 * thetaD_ddot_1 + g*np.cos(thetaD_1)
)
F2_2 = - (m_2 * d3 * d5) / ((x2 - d4)**2 + d3**2) * (
    d5 * thetaD_ddot_2 + g*np.cos(thetaD_2)
)
F2_12 = np.concatenate((F2_1, F2_2))

window_length = 101  # 必须为奇数
polyorder = 3

F2_12_smoothed = savgol_filter(F2_12, window_length, polyorder)
N1=len(theta1)
# ------------------------
# 9. 根据平滑后的整段θ拆分计算 F1 分段（第一段取负，第二段取正）
# ------------------------
theta1_smooth = theta_12_smooth[:N1]
theta2_smooth = theta_12_smooth[N1:]

F2_1 = F2_12_smoothed[:N1]
F2_2 = F2_12_smoothed[N1:]
window_length = 11  # 必须为奇数
polyorder = 3
F1_12_smooth = savgol_filter(F1_12, window_length, polyorder)

F1_1 = F1_12_smooth[:N1]
F1_2 = F1_12_smooth[N1:]
# ------------------------
# 10. 计算 F3 = F1 + F2（平滑后的结果）
# ------------------------
F3_12_smoothed = F1_12_smooth + F2_12_smoothed

# 为后续 ODE 求解中插值使用，按段计算 F3（保持原来分段公式）
F3_1 = -(F1_1 + F2_1)
F3_2 = F1_2 + F2_2

# ------------------------
# 11. 用 F3 求解新的θ曲线
# ------------------------
def ode_system_3(t, y):
    # 第一段ODE：使用插值获得当前F3_1值
    theta, theta_dot = y
    d_val = delta(theta)
    a_val = alpha(theta)
    b_val = beta(theta)
    F3_val = np.interp(t, t1, F3_1)
    dtheta = theta_dot
    dtheta_dot = - (F3_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val))
                     + K * (theta + a_val + b_val)
                     - coef * theta_dot**2
                     + m_1 * np.cos(theta)) / I
    return [dtheta, dtheta_dot]

def ode_system_4(t, y):
    # 第二段ODE：使用插值获得当前F3_2值
    theta, theta_dot = y
    d_val = delta(theta)
    a_val = alpha(theta)
    b_val = beta(theta)
    F3_val = np.interp(t, t_eval_2, F3_2)
    dtheta = theta_dot
    dtheta_dot = (F3_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val))
                   - K * (theta + a_val + b_val)
                   - coef * theta_dot**2
                   - m_1 * np.cos(theta)) / I
    return [dtheta, dtheta_dot]

sol_3 = solve_ivp(ode_system_3, [t0_1, tf_1], [0.8704004449728826, 0], t_eval=t_eval_1)
t_3 = sol_3.t
theta3 = sol_3.y[0]

sol_4 = solve_ivp(ode_system_4, [t0_2, tf_2], [-0.6391148802476694, 0], t_eval=t_eval_2)
t_4 = sol_4.t
theta4 = sol_4.y[0]

theta3_start = 0.8704004449728826
theta3_end = -0.6391148802476694
indices_theta3 = np.where((theta3 <= theta3_start) & (theta3 >= theta3_end))[0]
theta3_trimmed = theta3[indices_theta3]
t3_trimmed = t_3[indices_theta3]

indices_theta4 = np.where((theta4 <= theta3_start) & (theta4 >= theta3_end))[0]
theta4_trimmed = theta4[indices_theta4]
t4_trimmed = t_4[indices_theta4]

# 将第二段的时间平移后与裁剪后的第一段拼接
t_4_shifted = t4_trimmed + t1[-1]
t_theta_F3_new = np.concatenate((t3_trimmed, t_4_shifted))
theta_F3_new = np.concatenate((theta3_trimmed, theta4_trimmed))

# ------------------------
# 12. 绘图
# ------------------------
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
axs = axs.flatten()

# (1) θ(t)：显示平滑后的ODE解、几何关系得到的θD以及利用F3反求的新θ曲线
axs[0].plot(t_theta, theta_12_smooth, 'b-', label=r'$\theta_{ODE}$')
axs[0].plot(t_theta, thetaD_12, 'g-', label=r'$\theta_D$')
axs[0].plot(t_theta_F3_new, theta_F3_new, 'r--', label=r'$\theta_{F3}$')
axs[0].set_title(r'$\theta(t)$')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel(r'$\theta$ [rad]')
axs[0].legend()
axs[0].grid(True)

# (2) F2(t)
axs[1].plot(t_theta, F2_12_smoothed, 'r-')
axs[1].set_title(r'$F_2(t)$')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel(r'$F_2$ [N]')
axs[1].grid(True)

# (3) δ(t)
axs[2].plot(t_theta, delta_12, 'g-')
axs[2].set_title(r'$\delta(t)$')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel(r'$\delta$ [rad]')
axs[2].grid(True)

# (4) α(t) & β(t)
axs[3].plot(t_theta, alpha_12, 'm-', label=r'$\alpha(t)$')
axs[3].plot(t_theta, beta_12, 'b-', label=r'$\beta(t)$')
axs[3].set_title(r'$\alpha(t)$ and $\beta(t)$')
axs[3].set_xlabel('Time [s]')
axs[3].set_ylabel('Angle [rad]')
axs[3].legend()
axs[3].grid(True)

# (5) F1, F2, F3
axs[4].plot(t_theta, F1_12, 'c-', label=r'$F_1(t)$')
axs[4].plot(t_theta, 10*F2_12_smoothed, 'r-', label=r'$F_2(t)$')
axs[4].plot(t_theta, F3_12_smoothed, 'b-', label=r'$F_3(t)$')
axs[4].set_title(r'$F_1(t)$, $F_2(t)$ and $F_3(t)$')
axs[4].set_xlabel('Time [s]')
axs[4].set_ylabel('Force [N]')
axs[4].legend()
axs[4].grid(True)

# (6) ψ(t)
axs[5].plot(t_theta, psi_12, 'k-')
axs[5].set_title(r'$\psi(t)$')
axs[5].set_xlabel('Time [s]')
axs[5].set_ylabel(r'$\psi$ [rad]')
axs[5].grid(True)

# (7) φ(t)
axs[6].plot(t_theta, phi_12, 'k-')
axs[6].set_title(r'$\phi(t)$')
axs[6].set_xlabel('Time [s]')
axs[6].set_ylabel(r'$\phi$ [rad]')
axs[6].grid(True)

# (8) x(t)
axs[7].plot(t_theta, x_smooth, 'k-')
axs[7].set_title(r'$x(t)$')
axs[7].set_xlabel('Time [s]')
axs[7].set_ylabel(r'$x$ [m]')
axs[7].grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(t_theta, F1_12_smooth, 'c-', label=r'$F_1$')
plt.plot(t_theta, F2_12_smoothed, 'r-', label=r'$F_2$')
plt.plot(t_theta, F3_12_smoothed, 'b-', label=r'$F_3$')
plt.xlabel('t [s]')
plt.ylabel('Force [N]')
plt.legend()
plt.grid(True)
fig = plt.gcf()
fig.text(0.01, 0.99, '(b)', transform=fig.transFigure,
         fontsize=20, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.savefig('F_plot.png', bbox_inches='tight')
plt.show()