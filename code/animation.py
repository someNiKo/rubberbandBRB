import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import math

folder_path = 'local_data/'
name = 'proposed'
data_file_name = folder_path + name + '.csv'

data = pd.read_csv(data_file_name)
time_temp = data['time']
quat_avg = np.load('quat_avg.npy')

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_to_matrix(qw, qx, qy, qz):
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

time = time_temp - time_temp[0]
pos_x, pos_y, pos_z = data['x'], data['y'], data['z']
quat_x, quat_y, quat_z, quat_w = data['qx'], data['qy'], data['qz'], data['qw']
quat_avg_conj = quat_conjugate(quat_avg)

for i in range(len(quat_x)):
    quat_here = np.array([quat_w[i], quat_x[i], quat_y[i], quat_z[i]])
    quat_novel = quaternion_multiply(quat_avg_conj, quat_here)
    quat_w[i], quat_x[i], quat_y[i], quat_z[i] = quat_novel

flight_start_time, flight_time_length = 3.5, 4
time = time - flight_start_time
time_used_index = np.where((time > 0) & (time < flight_time_length))[0]
time, pos_x, pos_y, pos_z = time[time_used_index].to_numpy(), pos_x[time_used_index].to_numpy(), pos_y[time_used_index].to_numpy(), pos_z[time_used_index].to_numpy()
quat_x, quat_y, quat_z, quat_w = quat_x[time_used_index].to_numpy(), quat_y[time_used_index].to_numpy(), quat_z[time_used_index].to_numpy(), quat_w[time_used_index].to_numpy()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim(min(pos_x), max(pos_x))
# ax.set_ylim(min(pos_y), max(pos_y))

y_show_width_times = 3
y_shade_times = 3
x_shade_times = 1.02
x_diff = max(pos_y) - min(pos_y)
x_grid, y_grid = np.meshgrid(np.linspace(min(pos_x),  max(pos_x) , 10), 
                             np.linspace(min(pos_y)- ( y_show_width_times -1 ) * x_diff, 
                             max(pos_y)+ ( y_show_width_times -1 ) * x_diff, 10))
z_plane = np.zeros_like(x_grid)
ax.plot_surface(x_grid, y_grid, z_plane, color='#008792', alpha=0.2, rstride=100, cstride=100)
ax.set_zlim(-0.5, 2.5)


ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.grid(True)
# ax.plot(pos_x, pos_y, np.zeros_like(pos_z), color='gray', linestyle='dashed', label='Projected Path')
# ax.plot(pos_x, np.zeros_like(pos_y), pos_z, color='gray', linestyle='dashed')
# ax.plot(np.zeros_like(pos_x), pos_y, pos_z, color='gray', linestyle='dashed')

x_times = 1.0
y_times = 3.0
z_times = 1.5
ax.set_box_aspect((x_times * np.ptp(pos_x), y_times * np.ptp(pos_y), z_times * np.ptp(pos_z)))

traj, = ax.plot([], [], [], color='#008792', label='Flight Path')
attitude_lines = [ax.plot([], [], [], color=c, linewidth=2)[0] for c in ['red', 'green', 'blue']]

def update(num):
    traj.set_data(pos_x[:num], pos_y[:num])
    traj.set_3d_properties(pos_z[:num])
    R = quaternion_to_matrix(quat_w[num], quat_x[num], quat_y[num], quat_z[num])
    origin = np.array([pos_x[num], pos_y[num], pos_z[num]])
    axes = np.array([[0.6, 0, 0], [0, 0.6, 0], [0, 0, 0.6]])
    transformed_axes = R @ axes.T + origin[:, None]
    for i in range(3):
        attitude_lines[i].set_data([origin[0], transformed_axes[0, i]], [origin[1], transformed_axes[1, i]])
        attitude_lines[i].set_3d_properties([origin[2], transformed_axes[2, i]])
    return [traj] + attitude_lines

ani = animation.FuncAnimation(fig, update, frames=len(time), interval=50, blit=False)
ani.save('flight_animation.mp4', writer='ffmpeg')
plt.legend()
plt.show()
