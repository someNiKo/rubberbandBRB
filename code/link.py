import numpy as np
from scipy.optimize import fsolve
from numpy import sin, cos, pi


def equations(vars):
    l1, l2, l3 = vars
    theta_up = 45/180*pi
    theta_down = 35/180*pi
    delta = 1
    l4 = 0.4
    l5 = 0.4
    eq1 = (l4 + delta)**2 + (l2 - l1 + l5)**2 - l3**2
    eq2 = (l2*cos(theta_down) - l1 + l5)**2 + (l4 + 2*delta - l2*sin(theta_down))**2 - l3**2
    eq3 = (l4 + l2*sin(theta_up))**2 + (l2*cos(theta_up) - l1 + l5)**2 - l3**2
    return [eq1, eq2, eq3]


initial_guess = [1, 2, 1]

solution = fsolve(equations, initial_guess)

print("方程组的解为：", solution)