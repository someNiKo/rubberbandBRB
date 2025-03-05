folder_path = 'local_data/'

name = 'proposed'

data_file_name = folder_path + name + '.csv'

import pandas as pd
import numpy as np
data = pd.read_csv(data_file_name)

time_temp = data['time']

quat_avg = np.load('quat_avg.npy')
print('quat_avg:', quat_avg)

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions q1 and q2.
    Each quaternion is represented as a NumPy array [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])

def quat_conjugate(q):
    """ Compute quaternion conjugate """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_error(q, q_d):
    """ Compute quaternion error q_e = q_d* ⊗ q """
    return quaternion_multiply(quat_conjugate(q_d), q)

time = time_temp - time_temp[0] 


pos_x = data['x']
pos_y = data['y']
pos_z = data['z']

vel_x = [0.0] * len(pos_x)
vel_y = [0.0] * len(pos_x)
vel_z = [0.0] * len(pos_x)

time = np.array(time)
pos_x = np.array(pos_x)
pos_y = np.array(pos_y)
pos_z = np.array(pos_z)

for i in range(1, len(pos_x)):
    vel_x[i] = (pos_x[i] - pos_x[i-1]) / (time[i] - time[i-1])
    vel_y[i] = (pos_y[i] - pos_y[i-1]) / (time[i] - time[i-1])
    vel_z[i] = (pos_z[i] - pos_z[i-1]) / (time[i] - time[i-1])
    if abs(vel_x[i]) > 10.0:
        vel_x[i] = vel_x[i-1]
    if abs(vel_y[i]) > 10.0:
        vel_y[i] = vel_y[i-1]
    if abs(vel_z[i]) > 10.0:
        vel_z[i] = vel_z[i-1]
    # print(vel_x[i])
    
# from scipy.signal import  medfilt

# vel_x = savgol_filter(vel_x, 51, 3)
# vel_y = savgol_filter(vel_y, 51, 3)
# vel_z = savgol_filter(vel_z, 51, 3)
from scipy.signal import butter, filtfilt



def butterworth_filter(data, cutoff_freq, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Assuming a sampling frequency fs
# Compute frequency based on the average of the time differences
time_diffs = np.diff(time)
average_time_diff = np.mean(time_diffs)
frequency = 1 / average_time_diff

# print('Computed Frequency:', frequency)


fs = frequency  # Example: 1 Hz sampling frequency
cutoff_freq = 8.0  # Example: 0.1 Hz cutoff frequency

vel_x_filtered = butterworth_filter(vel_x, cutoff_freq, fs)
vel_y_filtered = butterworth_filter(vel_y, cutoff_freq, fs)
vel_z_filtered = butterworth_filter(vel_z, cutoff_freq, fs)

# Adjust the length of the filtered velocity arrays to match the original
vel_x = vel_x_filtered
vel_y = vel_y_filtered
vel_z = vel_z_filtered


# print('len vel x',len(vel_x))

quat_x = data['qx']
quat_y = data['qy']
quat_z = data['qz']
quat_w = data['qw']

quat_x = np.array(quat_x)
quat_y = np.array(quat_y)
quat_z = np.array(quat_z)
quat_w = np.array(quat_w)

quat_avg_conj = np.array([quat_avg[0], -quat_avg[1], -quat_avg[2], -quat_avg[3]])

for i in range(len(quat_x)):
    quat_here = np.array([quat_w[i], quat_x[i], quat_y[i], quat_z[i]])
    quat_novel = quaternion_multiply(quat_avg_conj, quat_here)
    quat_w[i] = quat_novel[0]
    quat_x[i] = quat_novel[1]
    quat_y[i] = quat_novel[2]
    quat_z[i] = quat_novel[3]


###################
###################
# Vital para here #
###################
###################
flight_start_time = 3.5
flight_time_length = 4

time = time - flight_start_time
# print('time before:', time)
time_used_index = np.where((time > 0) & (time < flight_time_length))[0]
# print('time_greater_than_zero_index', time_used_index)

time = time[time_used_index] 
# print('time shape', time.shape)
# print('time:', time)
pos_x = pos_x[time_used_index] - pos_x[time_used_index[0]]
pos_y = pos_y[time_used_index] - pos_y[time_used_index[0]]
pos_z = pos_z[time_used_index]


vel_x = vel_x[time_used_index]
vel_y = vel_y[time_used_index]
vel_z = vel_z[time_used_index]
quat_x = quat_x[time_used_index]
quat_y = quat_y[time_used_index]
quat_z = quat_z[time_used_index]
quat_w = quat_w[time_used_index]

# regularized_energy = []
# for i in range(len(vel_x)):
#     kinematic_energy = 0.5 * (vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i] + vel_z[i] * vel_z[i])
#     potential_energy = pos_z[i] * 9.81
#     regularized_energy.append(potential_energy + kinematic_energy)

# max_energy = max(regularized_energy)


# horizontal_vel = []
# for i in range(len(vel_x)):
#     horizontal_vel.append(np.sqrt(vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i]))
# max_horizontal_vel = max(horizontal_vel)
# ave_horizontal_vel = np.mean(horizontal_vel)

# print(f'max_energy: {max_energy:.4f}')
# print(f'max_horizontal_vel: {max_horizontal_vel:.4f}')
# print(f'ave_horizontal_vel: {ave_horizontal_vel:.4f}')

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20

# fig, ax = plt.subplots(3, 1, sharex=True)

# fig.set_size_inches(16, 9)

# line_width = 1.5

# ax[0].plot(time, pos_x, color='r', label='x', linewidth= line_width)
# ax[0].plot(time, pos_y, color='g', label='y', linewidth=1.5)
# ax[0].plot(time, pos_z, color='b', label='z', linewidth=1.5)
# ax[0].axhline(0, color='black', linestyle='--', linewidth=1.5)
# ax[0].legend(loc='upper right', bbox_to_anchor=(1, 1))
# ax[0].set_ylabel('Position (m)')
# ax[0].tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)


# # ax[1].plot(time, vel_x, color='r', label='x', linewidth=1.5)
# ax[1].plot(time, horizontal_vel, color='r', label='horizontal vel.', linewidth=1.5)
# ax[1].plot(time, vel_z, color='b', label='vertical vel.', linewidth=1.5)
# ax[1].axhline(0, color='black', linestyle='--', linewidth=1.5)
# ax[1].legend(loc='upper right', bbox_to_anchor=(1, 1))
# ax[1].set_ylabel('Velocity (m/s)')
# ax[1].tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)


roll, pitch, yaw = [], [], []

import math
def quaternion_to_rpy(qw, qx, qy, qz):
    """
    Convert quaternion (qw, qx, qy, qz) to roll, pitch, and yaw (radians).
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

quat_here = np. array([quat_w[0], quat_x[0], quat_y[0], quat_z[0]])

angular_velocity_x = []
angular_velocity_y = []
angular_velocity_z = []
k = 80

def quaternion_derivative(q, omega):
    """
    Compute quaternion derivative given angular velocity omega.
    """
    q0, q1, q2, q3 = q
    wx, wy, wz = omega

    dqdt = 0.5 * np.array([
        -q1 * wx - q2 * wy - q3 * wz,
         q0 * wx - q3 * wy + q2 * wz,
         q3 * wx + q0 * wy - q1 * wz,
        -q2 * wx + q1 * wy + q0 * wz
    ])
    
    return dqdt

def integrate_quaternion_euler(q, omega, dt):
    """
    Update quaternion using simple Euler integration.
    """
    dq = quaternion_derivative(q, omega) * dt
    q_new = q + dq
    return q_new / np.linalg.norm(q_new)  # Normalize to maintain unit quaternion

def integrate_quaternion_exp(q, omega, dt):
    """
    Update quaternion using quaternion exponential (accurate for larger dt).
    """
    omega_norm = np.linalg.norm(omega)
    
    if omega_norm > 1e-6:  # Avoid division by zero
        theta = omega_norm * dt / 2
        exp_q = np.hstack([
            np.cos(theta),
            np.sin(theta) * (omega / omega_norm)
        ])
        q_new = quaternion_multiply(q, exp_q)
    else:
        q_new = q  # No significant rotation
    
    return q_new / np.linalg.norm(q_new) 


for i in range(len(quat_w)):
    r, p, y = quaternion_to_rpy(quat_w[i], quat_x[i], quat_y[i], quat_z[i])
    q_d = np.array([quat_w[i], quat_x[i], quat_y[i], quat_z[i]])
    q_e = quat_error(quat_here, q_d)
    e_v = q_e[1:]
    omega_d =  k * e_v
    if i > 0:
        dt = time[i] - time[i-1]
    else:
        dt = 1e-2
        
    quat_here = integrate_quaternion_exp(quat_here, omega_d, dt)
    
    angular_velocity_x.append(omega_d[0])
    angular_velocity_y.append(omega_d[1])
    angular_velocity_z.append(omega_d[2])
    
    roll.append(r)
    pitch.append(p)
    yaw.append(y)



# ax[2].plot(time, roll, color='r', label='roll', linewidth=1.5)
# ax[2].plot(time, pitch, color='g', label='pitch', linewidth=1.5)
# ax[2].plot(time, yaw, color='b', label='yaw', linewidth=1.5)
# ax[2].legend(loc='upper right', bbox_to_anchor=(1, 1))
# ax[2].set_ylabel('Euler angles (rad)')
# ax[2].set_xlabel('time (s)')
# ax[2].tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)

# fig = plt.gcf()
# fig.text(0.01, 0.99, '(c)', transform=fig.transFigure,
#          fontsize=20, fontweight='bold', va='top', ha='left')
# fig.tight_layout()
# plt.show()
# fig.savefig('fig/' + name +'curves.pdf', dpi=300, bbox_inches='tight')




fig = plt.figure()
fig.set_size_inches(10, 7.5)  # Set the figure size to be larger, for example 10x8 inches

ax = fig.add_subplot(111, projection='3d')

darkred = (0.635, 0.078, 0.184, 1.0)
darkpurple = (0.294, 0.0, 0.510, 1.0)

ax.plot(pos_x, pos_y, pos_z, linewidth=3.0, label='Flight Path')
# Mark the start point
ax.scatter(pos_x[0], pos_y[0], pos_z[0] , color=darkred, s=50)
ax.text(pos_x[0], pos_y[0], pos_z[0] +0.4, 'start')

# Mark the end point
ax.scatter(pos_x[-1], pos_y[-1], pos_z[-1], color=darkpurple, s=50)
ax.text(pos_x[-1], pos_y[-1], pos_z[-1] + 0.4, 'end')

# Add legend for start and end points



y_show_width_times = 3
y_shade_times = 3
x_shade_times = 1.02
fig.tight_layout()
ax.plot(pos_x,  pos_y,   0, color='grey', label='Projected Flight Path')
ax.plot(pos_x,      max(pos_y) + ( y_show_width_times -1 ) * np.ptp(pos_y),      pos_z, color='grey')
ax.plot(min(pos_x) - ( x_shade_times -1 ) * np.ptp(pos_x),      pos_y,      pos_z, color='grey')
ax.legend()

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')


# Create a grid of points on the x-y plane
x_diff = max(pos_y) - min(pos_y)


x_grid, y_grid = np.meshgrid(np.linspace(min(pos_x),  max(pos_x) , 10), 
                             np.linspace(min(pos_y)- ( y_show_width_times -1 ) * x_diff, 
                             max(pos_y)+ ( y_show_width_times -1 ) * x_diff, 10))

# Define the z-values for the plane (all zeros)
z_plane = np.zeros_like(x_grid)

# Plot the transparent blue plane at z=0
ax.plot_surface(x_grid, y_grid, z_plane, color='#008792', alpha=0.2, rstride=100, cstride=100)
ax.set_zlim(-0.5, 2.5)

# print(np.ptp(pos_x))
x_times = 1.0
y_times = 3.0
z_times = 1.5

ax.set_box_aspect((x_times * np.ptp(pos_x), y_times * np.ptp(pos_y), z_times * np.ptp(pos_z)))
ax.locator_params(nbins=3)
fig = plt.gcf()
# fig.text(0.01, 0.99, '(c)', transform=fig.transFigure,
#          fontsize=20, fontweight='bold', va='top', ha='left')
fig.tight_layout()


# def handle_close(event):
#     fig.savefig('fig/' + name +'flight_path.svg', dpi=300, bbox_inches='tight')
#     fig.clf()
#     plt.close(fig)

# fig.canvas.mpl_connect('close_event', handle_close)

# 添加图例
# ax.legend()

plt.show()
# fig.savefig('fig/' + name +'flight_path.svg', dpi=300, bbox_inches='tight')