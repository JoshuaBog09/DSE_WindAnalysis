import numpy as np
import matplotlib.pyplot as plt

import math

# Define values

m = 5000    # kg
g = 9.81    # m/s^2
rotors: int = 12  # -
blades: int = 7  # -
chord = 0.08  # m
Cl = 0.4548
rho = 1.225

Cd = 0.1366
A_biplane = 56 * 2

T = m * g * (1 + 0.05175)
T_inv = T / rotors

RPM_lim = 2500  # RPM
t = 300  # s
v0 = 0

M_Max = 0.8
Vt_Max = 340 * M_Max

# Dynamically adapt this to incoorporate ceratin limits
a = 0
v = 0
s = 0

RPM = np.arange(100, 3500, 100)
R = ((6*T_inv*30**2)/(Cl*blades*chord*rho*(RPM*math.pi)**2))**(1/3)
V_tip = (RPM*2*math.pi/60)*R

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(RPM, R, 'g-', marker='o')
ax2.plot(RPM, V_tip, 'b-', marker='o')

ax1.set_xlabel('Rotations per minute [-]')
ax1.set_ylabel('Rotor radius [m]', color='g')
ax2.set_ylabel('Blade tip speed [m/s]', color='b')

ax1.axvline(x=2500, color='b', linestyle='--')
ax2.hlines(y=Vt_Max, xmin=1000, xmax=max(RPM), color='b', linestyle='--')

plt.show()

# m = 5000    # kg
# g = 9.81    # m/s^2
# rho = 1.225 # kg/m^3
#
# # Plane data
# Cd = 0.1366
# A_biplane = 56 * 2
#
# # Rotor Data
# rotors: int = 8  # -
# blades: int = 7  # -
# chord = 0.08  # m
# Cl = 0.4548
#
# # Init cond
# v_list = [0]
# a_list = [0]
# s_list = [0]
# t_list = [0]
# D_list = [0]
# T_list = []
#
# dt = 0.1
#
# T_list.append(m * g * (1 + 0.0005))
#
# while s_list[-1] <= 200:
#
#     D = Cd * 0.5 * rho * 1.2 * A_biplane * v_list[-1] ** 2
#
#     t = t_list[-1] + dt
#     a = (T_list[-1] - D - m * g) / m
#
#     # # Velocity cap
#     # if v_list[-1] > 1:
#     #     a = 0
#
#     # # Deceleration
#     # if s_list[-1] > 100:
#     #     a = - 0.005
#
#     v = v_list[-1] + a * dt
#     s = s_list[-1] + v * dt
#
#     t_list.append(t)
#     a_list.append(a)
#     v_list.append(v)
#     s_list.append(s)
#     D_list.append(D)
#     T_list.append((a*m) + D + (m * g))
#
#     # t.append(t[-1] + dt)
#     # a.append((T - D[-1] - m * g) / m)
#     # v.append(v[-1] + a[-1]*dt)
#     # s.append(s[-1] + v[-1]*dt)
#     #
#     # D.append(Cd * 0.5 * rho * 1.2 * A_biplane * v[-1] ** 2)
#
#
# fig, axs = plt.subplots(5, 1)
#
# axs[0].plot(t_list, a_list)
# axs[0].set_ylabel("a")
# axs[1].plot(t_list, v_list)
# axs[1].set_ylabel("v")
# axs[2].plot(t_list, s_list)
# axs[2].set_ylabel("s")
# axs[3].plot(t_list, D_list)
# axs[3].set_ylabel("D")
# axs[4].plot(t_list, T_list)
# axs[4].set_ylabel("T")
# axs[4].set_xlabel("t")
# plt.show()
