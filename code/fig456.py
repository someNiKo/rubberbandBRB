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
g  = 9.81
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
F1_1   = np.array([-F1(th) for th in theta1])  # 保持原代码符号
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
F1_2   = np.array([F1(th) for th in theta2])
psi2   = np.array([psi(th) for th in theta2])
phi2   = np.array([phi(th) for th in theta2])
x2     = np.array([x(th) for th in theta2])

# ------------------------
# 5. 拼接两段积分结果
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


window_length = 11  # 必须为奇数
polyorder = 3
theta_12_smooth = savgol_filter(theta_12, window_length, polyorder)

N1=len(theta1)
theta_1 = theta_12_smooth[:N1]
theta_2 = theta_12_smooth[N1:]
# ------------------------
# 6. 计算 F2(t)
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


window_length = 11  # 必须为奇数
polyorder = 3

F2_12_smoothed = savgol_filter(F2_12, window_length, polyorder)

N1=len(theta1)
F2_1 = F2_12_smoothed[:N1]
F2_2 = F2_12_smoothed[N1:]

window_length = 11  # 必须为奇数
polyorder = 3
F1_12_smooth = savgol_filter(F1_12, window_length, polyorder)

F1_1 = F1_12_smooth[:N1]
F1_2 = F1_12_smooth[N1:]

F3_12 = F1_12_smooth + F2_12_smoothed

# 定义 F3 分段，用于后续 ODE 积分
F3_1 = -(F1_1 + F2_1)
F3_2 = F1_2 + F2_2

# ------------------------
# 7. 利用 F3 求新的 θ 曲线
# ------------------------
def ode_system_3(t, y):
    theta, theta_dot = y
    d_val = delta(theta)
    a_val = alpha(theta)
    b_val = beta(theta)
    F3_val = np.interp(t, t1, F3_1)
    dtheta = theta_dot
    dtheta_dot = - (
        F3_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val))
        + K * (theta + a_val + b_val)
        - coef * theta_dot**2
        + m_1 * np.cos(theta)
    ) / I
    return [dtheta, dtheta_dot]

def ode_system_4(t, y):
    theta, theta_dot = y
    d_val = delta(theta)
    a_val = alpha(theta)
    b_val = beta(theta)
    F3_val = np.interp(t, t_eval_2, F3_2)
    dtheta = theta_dot
    dtheta_dot = (
        F3_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val))
        - K * (theta + a_val + b_val)
        - coef * theta_dot**2
        - m_1 * np.cos(theta)
    ) / I
    return [dtheta, dtheta_dot]

sol_3 = solve_ivp(ode_system_3, [t0_1, tf_1], [0.8704004449728826, 0], t_eval=t_eval_1)
t_3 = sol_3.t
theta3 = sol_3.y[0]

sol_4 = solve_ivp(ode_system_4, [t0_2, tf_2], [-0.6391148802476694, 0], t_eval=t_eval_2)
t_4 = sol_4.t
theta4 = sol_4.y[0]

# 定义要求区间：从 0.8704004449728826 到 -0.6391148802476694
theta_high = 0.8704004449728826
theta_low  = -0.6391148802476694

# 对 theta3 裁剪，只保留区间内的部分
indices_theta3 = np.where((theta3 <= theta_high) & (theta3 >= theta_low))[0]
theta3_trimmed = theta3[indices_theta3]
t3_trimmed = t_3[indices_theta3]

# 对 theta4 也进行裁剪
indices_theta4 = np.where((theta4 <= theta_high) & (theta4 >= theta_low))[0]
theta4_trimmed = theta4[indices_theta4]
t4_trimmed = t_4[indices_theta4]

# 将第二段的时间平移后与裁剪后的第一段拼接
t_4_shifted = t4_trimmed + t1[-1]
t_theta_F3_new = np.concatenate((t3_trimmed, t_4_shifted))
theta_F3_new = np.concatenate((theta3_trimmed, theta4_trimmed))

window_length = 11  # 必须为奇数
polyorder = 3
theta_F3_new_smooth  = savgol_filter(theta_F3_new , window_length, polyorder)
print("修改后 t 的总时长:", t_theta_F3_new[-1])
# ------------------------
# 8. 分别生成两个独立图像
# ------------------------
# ------------------------
# 计算 F_L 和 F_L'
# ------------------------

# 1) 原拼接解的角速度
# =============================
#  在已有脚本的后半部分插入如下示例
# =============================

# （1）对原拼接解 theta1, theta2 分段求导数并计算 F_L
theta_dot_1 = np.gradient(theta1, t1)  # 第一段导数
theta_dot_2 = np.gradient(theta2, t2)  # 第二段导数

theta_dot_12 = np.concatenate((theta_dot_1, theta_dot_2))
window_length = 301  # 必须为奇数
polyorder = 7
theta_dot_12_smooth = savgol_filter(theta_dot_12, window_length, polyorder)



plt.figure(figsize=(15,5))
plt.plot(t_theta, theta_dot_12, 'b-', label=r'$F_{lift}$')
plt.plot(t_theta, theta_dot_12_smooth, 'b-', label=r'$F_{lift}$')
plt.xlabel('t [s]')
plt.ylabel(r'Force [N]')
plt.legend()
plt.grid(True)
fig = plt.gcf()
fig.text(0.01, 0.99, '(d)', transform=fig.transFigure,
         fontsize=20, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.savefig('F_L_plot.png', bbox_inches='tight')
plt.show()

theta_dot_1 = theta_dot_12_smooth[:N1]
theta_dot_2 = theta_dot_12_smooth[N1:]

F_L_1 = coef1 * (theta_dot_1**2)
F_L_2 = coef1 * (theta_dot_2**2)

# 将时间拼接
time_offset = t1[-1]
t2_shifted = t2 + time_offset
t_theta_FL = np.concatenate((t1, t2_shifted))

# 将 F_L 也拼接
F_L_12 = np.concatenate((F_L_1, F_L_2))

window_length = 301  # 必须为奇数
polyorder = 3
F_L_smooth = savgol_filter(F_L_12, window_length, polyorder)


# （2）对新力 F3 对应的解 theta3, theta4 分段求导数并计算 F_L'
theta3_dot = np.gradient(theta3, t_3)  # 第三段导数
theta4_dot = np.gradient(theta4, t_4)  # 第四段导数

theta_dot_34 = np.concatenate((theta3_dot, theta4_dot))
window_length = 301  # 必须为奇数
polyorder = 7
theta_dot_34_smooth = savgol_filter(theta_dot_34, window_length, polyorder)

theta3_dot = theta_dot_34_smooth[:N1]
ttheta4_dot = theta_dot_34_smooth[N1:]
F_L3 = coef1 * (theta3_dot**2)
F_L4 = coef1 * (theta4_dot**2)

F_L_3_12 = np.concatenate((F_L3, F_L4))

window_length = 301  # 必须为奇数
polyorder = 3
F_L34_smooth = savgol_filter(F_L_3_12, window_length, polyorder)
# 这两段时间也需要拼接（若你想和原解类似地放在一条时间轴）



time_offset_3 = t_3[-1]
t4_shifted = t_4 + time_offset_3
t_theta_F3_FL = np.concatenate((t_3, t4_shifted))

# 拼接 F_L'


T_12 = t_theta_FL[-1] - t_theta_FL[0]  
# 若 t_theta_FL[0] == 0，可直接用 t_theta_FL[-1] 也行

# 使用梯形法数值积分
integral_F_L_12 = np.trapz(F_L_smooth, x=t_theta_FL)

# 平均升力
F_L_12_avg = integral_F_L_12 / T_12

print("原解平均升力 F_L_12_avg =", F_L_12_avg, "N")


# ========== 新解 (F3) 的平均升力 ==========

T_F3_12 = t_theta_F3_FL[-1] - t_theta_F3_FL[0]
integral_F_L_3_12 = np.trapz(F_L34_smooth, x=t_theta_F3_FL)
F_L_3_12_avg = integral_F_L_3_12 / T_F3_12

print("新解平均升力 F_L_3_12_avg =", F_L_3_12_avg, "N")

# （3）示例：可视化 F_L 和 F_L' 随时间的变化
plt.figure(figsize=(15,5))
plt.axvspan(t_theta_FL[0], t_theta_FL[N1-1], color='gray', alpha=0.3)
plt.plot(t_theta_FL, F_L_smooth, 'b-', lw=3,label=r'$F_{lift}$')
plt.plot(t_theta_F3_FL, F_L34_smooth, 'r--', lw=3,label=r'$F^\prime_{lift}$')
plt.xlabel('t [s]')
plt.ylabel(r'Force [N]')
plt.legend()
plt.grid(True)
downstroke_x = (t_theta_FL[0] + t_theta_FL[N1-1]) / 2
plt.text(downstroke_x, max(F_L34_smooth)*0.95, 'downstroke',
         ha='center', fontsize=20, alpha=0.5)

# 在剩余 t2 区间中间添加 "upstroke" 文字
upstroke_x = (t_theta_FL[N1] + t_theta_FL[-1]) / 2
plt.text(upstroke_x, max(F_L34_smooth)*0.95, 'upstroke',
         ha='center', fontsize=20, alpha=0.5)

fig = plt.gcf()
fig.text(0.01, 0.99, '(d)', transform=fig.transFigure,
         fontsize=20, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.savefig('F_L_plot.png', bbox_inches='tight')
plt.show()

M2_2 = m_2*d5* (
    d5 * thetaD_ddot_2 + g*np.cos(thetaD_2)
)
M2_1 = m_2*d5* (
    d5 * thetaD_ddot_1 + g*np.cos(thetaD_1)
)

M2 = np.concatenate((M2_1,M2_2))

window_length = 301  # 必须为奇数
polyorder = 3
M2_smooth = savgol_filter(M2, window_length, polyorder)

M1_1 = F_L3*1.218412478/3.493773751*(0.06-(0.065/4))

M1_2 = -F_L4*1.218412478/3.493773751*(0.06-(0.065/4))

M1=np.concatenate((M1_1,M1_2))

window_length = 201  # 必须为奇数
polyorder = 3
M1_smooth = savgol_filter(M1, window_length, polyorder)

M = M1_smooth+M2_smooth
plt.figure(figsize=(15,5))

plt.axvspan(t_theta[0], t_theta[N1-1], color='gray', alpha=0.3)

plt.plot(t_theta, M, color= 'purple', lw=3,label=r'$M_{total}$')
plt.plot(t_theta, M1_smooth, 'b-', lw=3,label=r'$M_{drag}$')
plt.plot(t_theta, M2_smooth, 'r-', lw=3,label=r'$M_{abdomen}$')
plt.xlabel('t [s]')
plt.ylabel(r'Torque [N·m]')
plt.legend()
plt.grid(True)
# 在 t1 区间中间添加 "downstroke" 文字
downstroke_x = (t_theta[0] + t_theta[N1-1]) / 2
plt.text(downstroke_x, max(M)*0.95, 'downstroke',
         ha='center', fontsize=20, alpha=0.5)

# 在剩余 t2 区间中间添加 "upstroke" 文字
upstroke_x = (t_theta[N1] + t_theta[-1]) / 2
plt.text(upstroke_x, max(M)*0.95, 'upstroke',
         ha='center', fontsize=20, alpha=0.5)

plt.tight_layout()
plt.savefig('F_L_plot.png', bbox_inches='tight')
plt.show()

# 图像1：θ(t) 曲线
plt.figure(figsize=(15,5))

plt.axvspan(t_theta[0], t_theta[N1-1], color='gray', alpha=0.3)
plt.axvspan(t_theta_F3_new[0], t_theta_F3_new[len(t3_trimmed)-1], color='gray', alpha=0.3)

plt.plot(t_theta, theta_12_smooth, 'b-',lw=3, label=r'$\theta_A$')
plt.plot(t_theta_F3_new, theta_F3_new_smooth, 'r--',lw=3, label=r'$\theta_A^\prime$')
plt.xlabel('t [s]')
plt.ylabel(r'Angle [rad]')
plt.legend()
plt.grid(True)

# 在 t1 区间中间添加 "downstroke" 文字
downstroke_x = (t_theta[0] + t_theta[N1-1]) / 2
plt.text(downstroke_x, max(theta_12_smooth)*0.95, 'downstroke',
         ha='center', fontsize=20,alpha=0.5)

# 在剩余 t2 区间中间添加 "upstroke" 文字
upstroke_x = (t_theta[N1] + t_theta[-1]) / 2
plt.text(upstroke_x, max(theta_12_smooth)*0.95, 'upstroke',
         ha='center', fontsize=20,alpha=0.5)

fig = plt.gcf()
fig.text(0.01, 0.99, '(c)', transform=fig.transFigure,
         fontsize=20, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.savefig('theta_plot.png', bbox_inches='tight')
plt.show()

# 图像2：F1, F2, F3 曲线
plt.figure(figsize=(15,5))

plt.axvspan(t_theta[0], t_theta[N1-1], color='gray', alpha=0.3)

plt.plot(t_theta, F1_12_smooth, 'b-', lw=3,label=r'$F_1$')
plt.plot(t_theta, F2_12_smoothed, 'r-', lw=3,label=r'$F_2$')
plt.plot(t_theta, F3_12, color='purple',lw=3, label=r'$F_3$')
plt.xlabel('t [s]')
plt.ylabel('Force [N]')
plt.legend()
plt.grid(True)
downstroke_x = (t_theta[0] + t_theta[N1-1]) / 2
plt.text(downstroke_x, max(F3_12)*0.95, 'downstroke',
         ha='center', fontsize=20,alpha=0.5)

# 在剩余 t2 区间中间添加 "upstroke" 文字
upstroke_x = (t_theta[N1] + t_theta[-1]) / 2
plt.text(upstroke_x, max(F3_12)*0.95, 'upstroke',
         ha='center', fontsize=20,alpha=0.5)

fig = plt.gcf()
fig.text(0.01, 0.99, '(b)', transform=fig.transFigure,
         fontsize=20, fontweight='bold', va='top', ha='left')
plt.tight_layout()
plt.savefig('F_plot.png', bbox_inches='tight')
plt.show()
