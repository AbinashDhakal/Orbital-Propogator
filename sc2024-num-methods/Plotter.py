# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 20:45:44 2024

@author: abudh
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from Integrators import Kepler_RK4  # Ensure this matches the module and function names

def parse_orbit_data(filename):
    with open(filename, "r") as fp:
        if fp.readable():
            data = csv.reader(fp)
            lst = [line for line in data]
            ndata = len(lst) - 1

            time = np.zeros(ndata)
            pos = np.zeros((3, ndata))
            vel = np.zeros((3, ndata))

            for i in range(ndata):
                time[i] = float(lst[i + 1][0])
                for j in range(3):
                    pos[j][i] = float(lst[i + 1][j + 1])
                    vel[j][i] = float(lst[i + 1][j + 4])
        else:
            raise IOError(f"Unreadable data, something's wrong with the file {filename}")

    return time, pos, vel

if __name__ == "__main__":
    # File simulated from GMAT
    filename = "Satellite_PVT_GMAT.csv"
    
    # Parse the orbit data
    time, pos, vel = parse_orbit_data(filename)
    
    # Initial conditions
    pos_0 = pos[:, 0]  # First position vector
    vel_0 = vel[:, 0]  # First velocity vector
    t_params = [time[0], time[-1]]  # Start and end time
    Npoints = len(time)  # Number of points

    # Run the Kepler problem solver using the 4th order Runge-Kutta method
    t_RK, pos_RK, vel_RK = Kepler_RK4(pos_0, vel_0, t_params, Npoints)
    
    # 3D Plot example
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot3D(pos[0], pos[1], pos[2], "blue", label="Actual")
    ax.plot3D(pos_RK[:, 0], pos_RK[:, 1], pos_RK[:, 2], "red", label="RK4")
    ax.set_title("3D Orbit Trajectory")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.legend()
    plt.show()

    # 2D Plot example
    fig, ax = plt.subplots()
    ax.plot(time, pos[0], color="green", linestyle="solid", label="Actual pos X")
    ax.plot(time, pos[1], color="red", linestyle="solid", label="Actual pos Y")
    ax.plot(time, pos[2], color="blue", linestyle="solid", label="Actual pos Z")
    ax.plot(time, pos_RK[:, 0], color="green", linestyle="dashed", label="RK4 pos X")
    ax.plot(time, pos_RK[:, 1], color="red", linestyle="dashed", label="RK4 pos Y")
    ax.plot(time, pos_RK[:, 2], color="blue", linestyle="dashed", label="RK4 pos Z")
    
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Position [km]")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.show()
    
    # Display some initial data points for verification
    print("First two entries of time:")
    print(time[:2])
    
    print("\nFirst two entries of position (x, y, z):")
    print(pos[:, :2])
    
    print("\nFirst two entries of velocity (vx, vy, vz):")
    print(vel[:, :2])
