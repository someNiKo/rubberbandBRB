import numpy as np
from scipy.optimize import fsolve

# ------------------------
# 1) 定义已知常数 (SI 单位)
# ------------------------
l1 = 14.95e-3
l2 = 13.49e-3
l3 = 13.75e-3  
d1 = 4.89e-3
d2 = 0.03 
x_target = -0.01  # 目标 x 值

# ------------------------
# 2) 定义 x(θ) 函数
# ------------------------
def x_of_theta(theta):
    # 先计算 arccos(...) 的参数，并确保数值在 [-1, 1] 范围内
    arg1 = (l1 + d1 - l3) / l2
    arg1 = np.clip(arg1, -1, 1)
    term1 = l2 * np.sin(np.arccos(arg1))
    
    # arcsin(...) 的参数，同样做 clip
    arg2 = (l1 * np.cos(theta) - l3 + d1) / l2
    arg2 = np.clip(arg2, -1, 1)
    term2 = l2 * np.cos(np.arcsin(arg2))
    
    term3 = l1 * np.sin(theta)
    
    # 根据笔记中的形式：x = term1 - term2 + term3
    return term1 - term2 + term3

# ------------------------
# 3) 构造方程 f(θ)=0 并求解
# ------------------------
def equation(theta):
    return x_of_theta(theta) - x_target

# 选取一个初始猜测值（可以多试几个）
theta_guess = 0.01  # 或者 0.5, -0.2 等

# 调用 fsolve 求解
theta_solution = fsolve(equation, theta_guess)

print("解得 θ =", theta_solution[0], " [rad]")
print("对应的 x(θ) =", x_of_theta(theta_solution[0]))