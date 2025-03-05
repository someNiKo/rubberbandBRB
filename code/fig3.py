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

# 质量和转动惯量
m_1 = 0.000340301601
m_2 = 1e-3
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
    dtheta_dot = - (F1_val * l1 * np.cos(theta + d_val) / (2 * np.cos(d_val))
                    + K * (theta + a_val + b_val)
                    - coef * theta_dot**2
                    + m_1 * np.cos(theta)) / I
    return [dtheta, dtheta_dot]

def ode_system_2(t, y):
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
# 3. 定义模拟函数（将两段积分拼接，并返回每段的时间区间及标签）
# ------------------------
def simulate(tf1, tf2, y0_1, y0_2):
    t_total_list = []
    theta_total_list = []
    segments = []  # 每个元素为 (t_start, t_end, label)
    current_offset = 0.0
    downstroke = True  # 第一段为下扑
    while current_offset < 0.2:
        if downstroke:
            t_eval1 = np.linspace(0, tf1, 1000)
            sol1 = solve_ivp(ode_system_1, [0, tf1], y0_1, t_eval=t_eval1)
            t1 = sol1.t + current_offset
            theta1 = sol1.y[0]
            t_total_list.append(t1)
            theta_total_list.append(theta1)
            segments.append((t1[0], t1[-1], 'downstroke'))
            current_offset = t1[-1]
            downstroke = False
            if current_offset >= 0.2:
                break
        else:
            t_eval2 = np.linspace(0, tf2, 1000)
            sol2 = solve_ivp(ode_system_2, [0, tf2], y0_2, t_eval=t_eval2)
            t2 = sol2.t + current_offset
            theta2 = sol2.y[0]
            t_total_list.append(t2)
            theta_total_list.append(theta2)
            segments.append((t2[0], t2[-1], 'upstroke'))
            current_offset = t2[-1]
            downstroke = True
    t_total = np.concatenate(t_total_list)
    theta_total = np.concatenate(theta_total_list)
    mask = t_total <= 0.2
    # 过滤掉超出时间范围的片段（若有）
    segments = [seg for seg in segments if seg[1] <= 0.2]
    return t_total[mask], theta_total[mask], segments

# ------------------------
# 4. 计算模拟结果（F = 3）
# ------------------------
tf1_5 = 0.060526589179550463 
tf2_5 = 0.06358769972107282 
y0_1_5 = [0.8704004449728826, 0]
y0_2_5 = [-0.6391148802476694, 0]
t_total, theta_total, segments = simulate(tf1_5, tf2_5, y0_1_5, y0_2_5)

window_length = 11  # 必须为奇数
polyorder = 3
theta_total_smooth = savgol_filter(theta_total, window_length, polyorder)

# ------------------------
# 5. 计算 theta_D 曲线： theta_D = -arctan((x(t)-d4)/d3)
# ------------------------
x_total = x(theta_total_smooth)
theta_D = -np.arctan((x_total - d4) / d3)

# ------------------------
# 6. 绘图：将 theta(t) 和 theta_D(t) 绘制在同一张图中，并将下扑部分标成灰色、上扑部分标上文字
# ------------------------
plt.figure(figsize=(15, 5))

# 对每个区间都添加文字标注，若是下扑则先添加灰色背景
for seg in segments:
    t_start, t_end, label = seg
    mid = (t_start + t_end) / 2
    if label == 'downstroke':
        plt.axvspan(t_start, t_end, color='gray', alpha=0.3)
    plt.text(mid, max(theta_total_smooth)*0.95, label, ha='center', fontsize=20, alpha=0.5)

# 绘制曲线
plt.plot(t_total, theta_total_smooth, color='blue', lw=3, label=r'$\theta_A$')
plt.plot(t_total, theta_D, color='red', linestyle='--', lw=3, label=r'$\theta_D$')
plt.xlabel('t [s]')
plt.ylabel(r'Angle [rad]')

# 使 y 轴关于 0 对称
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
plt.savefig('output.png', bbox_inches='tight')
plt.show()