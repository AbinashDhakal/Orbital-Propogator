# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 20:46:22 2024

@author: abudh
"""
import pandas as pd

from Integrators import *  # Ensure this matches the module and function names
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Harmonic Oscillator Parameters
    k = 1.0  # Spring constant [N/m]
    m = 1.0  # Mass [kg]
    x_0 = 1.0  # Initial position [m]
    v_0 = 0.0  # Initial velocity [m/s]
    Npoints = 100
    t_params = [0.0, 4 * np.pi, Npoints]
    
    # Compute the solution using Euler's method
    t_arr, x_arr, v_arr = Euler_Oscillator(k, m, x_0, v_0, t_params)
    
    # Compute the solution using Half-Step Verlet method
    t_arr_verlet, x_arr_verlet, v_arr_verlet = Verlet_Oscillator(k, m, x_0, v_0, t_params)
    
    # Compute the solution using Runge-Kutta method
    t_arr_rk, x_arr_rk, v_arr_rk = RungeKutta_Oscillator(k, m, x_0, v_0, t_params)
    
    # Compute the real solution for Harmonic Oscillator
    omega = np.sqrt(k / m)
    x_real = x_0 * np.cos(omega * t_arr)
    
    # Plotting Harmonic Oscillator (Euler, Verlet, and Real Solutions)
    plt.figure(figsize=(12, 6))
    plt.plot(t_arr, x_arr, label='Euler Position (x)', linestyle='-', alpha=0.7, color='green')
    plt.plot(t_arr_verlet, x_arr_verlet, label='Verlet Position (x)', linestyle='-', alpha=0.7, color='blue')
    plt.plot(t_arr_rk, x_arr_rk, label='Runge-Kutta Position (x)', linestyle='-', alpha=0.7, color='red')
    plt.plot(t_arr, x_real, label='Real Position (x)', linestyle='-', alpha=0.7, color='black')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Harmonic Oscillator: Euler vs Half-Step Verlet vs Runge-Kutta Methods')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate the relative error
    min_length = min(len(t_arr), len(t_arr_verlet), len(t_arr_rk))
    Euler_Error = np.abs(x_real[:min_length] - np.interp(t_arr[:min_length], t_arr, x_arr)) / np.abs(x_real[:min_length]) * 100
    Verlet_Error = np.abs(x_real[:min_length] - np.interp(t_arr_verlet[:min_length], t_arr_verlet, x_arr_verlet)) / np.abs(x_real[:min_length]) * 100
    Runge_Error = np.abs(x_real[:min_length] - np.interp(t_arr_rk[:min_length], t_arr_rk, x_arr_rk)) / np.abs(x_real[:min_length]) * 100
    
    # Plot the error
    plt.figure(figsize=(12, 6))
    plt.plot(t_arr[:min_length], Euler_Error, color="blue", linestyle="solid", label="Euler Error (%)")
    plt.plot(t_arr_verlet[:min_length], Verlet_Error, color="red", linestyle="solid", label="Verlet Error (%)")
    plt.plot(t_arr_rk[:min_length], Runge_Error, color="orange", linestyle="solid", label="Runge-Kutta Error (%)")
    plt.yscale('log')

    plt.xlabel("Time [s]")
    plt.ylabel("Relative Error (%)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
    
    # Use one of the fixes for the file path
    csv_file = r'C:\Users\abudh\Desktop\SpaceCamp\sc2024-num-methodsv\Satellite_PVT_GMAT.csv'
    data = pd.read_csv(csv_file)
    
    # Extract the time, position, and velocity data
    t_arr = data['Time (UTC)']
    
    pos_0 = [data['x (km)'], data['y (km)'], data['z (km)']]
    vel_0 = [data['vx (km/sec)'], data['vy (km/sec)'], data['vz (km/sec)']]
    
    # Assuming Kepler_RK4 is a function you've defined elsewhere
    pos, vel, t_arr = Kepler_RK4(pos_0, vel_0, t_params)
if __name__ == "__main__":
    main()
