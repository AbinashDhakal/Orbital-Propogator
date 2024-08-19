# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 07:54:00 2024

@author: abudh
"""

import numpy as np

# Constants
G = 6.67e-11  # Gravitational constant [m^3 kg^-1 s^-2]
R_e = 6381370.0  # Radius of Earth [m]
M_e = 5.972e24  # Mass of Earth [kg]

def Kepler_RHS(pos, vel):
    """
    Calculates the right-hand side (RHS) of the differential equation
    for the Kepler problem in dimensionless units.

    Parameters:
    pos (numpy array): Position vector [x, y, z]
    vel (numpy array): Velocity vector [vx, vy, vz]

    Returns:
    rhs_pos: Time derivative of position (which is velocity)
    rhs_vel: Time derivative of velocity
    """
    norm_pos = np.linalg.norm(pos)
    if norm_pos == 0:
        raise ValueError("Position vector norm is zero, leading to division by zero.")
    
    rhs_pos = vel
    rhs_vel = -4.0 * np.pi**2 * pos / norm_pos**3
    return rhs_pos, rhs_vel

def Kepler_RK4(pos_0, vel_0, t_params, Npoints):
    """
    Solves the Kepler problem using the 4th order Runge-Kutta method.

    Parameters:
    pos_0 (list): Initial position [x0, y0, z0]
    vel_0 (list): Initial velocity [vx0, vy0, vz0]
    t_params (list): Initial and final times [t_i, t_f]
    Npoints (int): Number of points in the solution

    Returns:
    t_arr: Array of time points
    pos: Array of positions over time
    vel: Array of velocities over time
    """
    # Renormalization time factor (orbital period T_e)
    T_e = 2.0 * np.pi * np.sqrt(R_e**3 / (M_e * G))
    V_e = R_e / T_e  # Orbital velocity scale [m/s]
    
    # Convert to dimensionless time
    t_i = t_params[0] / T_e
    t_f = t_params[1] / T_e
    dt = (t_f - t_i) / (Npoints - 1)
    t_arr = np.linspace(t_i, t_f, Npoints)
    
    # Initialize position and velocity arrays
    pos = np.zeros((Npoints, 3))
    vel = np.zeros((Npoints, 3))
    
    # Set initial conditions
    pos[0] = np.array(pos_0) / R_e
    vel[0] = np.array(vel_0) / V_e
    
    # Initialize RK4 coefficients
    k_pos = np.zeros((4, 3))
    k_vel = np.zeros((4, 3))
    
    # RK4 integration loop
    for i in range(Npoints - 1):
        rhs_pos, rhs_vel = Kepler_RHS(pos[i], vel[i])
        k_pos[0] = rhs_pos * dt
        k_vel[0] = rhs_vel * dt
        
        rhs_pos, rhs_vel = Kepler_RHS(pos[i] + k_pos[0] / 2.0, vel[i] + k_vel[0] / 2.0)
        k_pos[1] = rhs_pos * dt
        k_vel[1] = rhs_vel * dt
        
        rhs_pos, rhs_vel = Kepler_RHS(pos[i] + k_pos[1] / 2.0, vel[i] + k_vel[1] / 2.0)
        k_pos[2] = rhs_pos * dt
        k_vel[2] = rhs_vel * dt
        
        rhs_pos, rhs_vel = Kepler_RHS(pos[i] + k_pos[2], vel[i] + k_vel[2])
        k_pos[3] = rhs_pos * dt
        k_vel[3] = rhs_vel * dt
        
        # Update position and velocity
        pos[i + 1] = pos[i] + (k_pos[0] + 2.0 * k_pos[1] + 2.0 * k_pos[2] + k_pos[3]) / 6.0
        vel[i + 1] = vel[i] + (k_vel[0] + 2.0 * k_vel[1] + 2.0 * k_vel[2] + k_vel[3]) / 6.0
    
    # Convert back to dimensional units
    return t_arr * T_e, pos * R_e, vel * V_e

# Constants for the harmonic oscillator
k = 1.0  # Spring constant [N/m]
m = 1.0  # Mass [kg]

def rhs_osc(k, m, x, v):
    """
    Calculates the right-hand side (RHS) of the differential equation
    for a harmonic oscillator.

    Parameters:
    k (float): Spring constant
    m (float): Mass
    x (float): Position
    v (float): Velocity

    Returns:
    rhs_x: Time derivative of position (which is velocity)
    rhs_v: Time derivative of velocity (acceleration)
    """
    rhs_x = v
    rhs_v = -k / m * x
    return rhs_x, rhs_v

def Euler_Oscillator(k, m, x_0, v_0, t_params):
    """
    Solves the harmonic oscillator problem using Euler's method.

    Parameters:
    k (float): Spring constant
    m (float): Mass
    x_0 (float): Initial position
    v_0 (float): Initial velocity
    t_params (list): Initial and final times [t_i, t_f]
    
    Returns:
    t_arr: Array of time points
    x_arr: Array of positions over time
    v_arr: Array of velocities over time
    """
    t_i, t_f, Npoints = t_params
    dt = (t_f - t_i) / (Npoints - 1)
    
    x_arr = np.zeros(Npoints)
    v_arr = np.zeros(Npoints)
    t_arr = np.linspace(t_i, t_f, Npoints)
    
    x_arr[0] = x_0
    v_arr[0] = v_0
    
    for i in range(Npoints - 1):
        rhs_x, rhs_v = rhs_osc(k, m, x_arr[i], v_arr[i])
        x_arr[i + 1] = x_arr[i] + rhs_x * dt
        v_arr[i + 1] = v_arr[i] + rhs_v * dt
    
    return t_arr, x_arr, v_arr

def Verlet_Oscillator(k, m, x_0, v_0, t_params):
    """
    Solves the harmonic oscillator problem using the Half-Step Verlet method.

    Parameters:
    k (float): Spring constant
    m (float): Mass
    x_0 (float): Initial position
    v_0 (float): Initial velocity
    t_params (list): Initial and final times [t_i, t_f]
    
    Returns:
    t_arr: Array of time points
    x_arr: Array of positions over time
    v_arr: Array of velocities over time
    """
    t_i, t_f, Npoints = t_params
    dt = (t_f - t_i) / (Npoints - 1)
    
    x_arr = np.zeros(Npoints)
    v_arr = np.zeros(Npoints)
    t_arr = np.linspace(t_i, t_f, Npoints)
    
    x_arr[0] = x_0
    v_arr[0] = v_0
    
    # Compute initial acceleration
    rhs_x, rhs_v = rhs_osc(k, m, x_0, v_0)
    a_n = rhs_v
    
    # Compute the first position update
    x_arr[1] = x_0 + v_0 * dt + 0.5 * a_n * dt**2
    
    # Compute the initial half-step velocity
    rhs_x, rhs_v = rhs_osc(k, m, x_arr[1], v_0)
    v_half = v_0 + 0.5 * rhs_v * dt
    
    for i in range(1, Npoints - 1):
        rhs_x, rhs_v = rhs_osc(k, m, x_arr[i], v_half)
        
        x_arr[i + 1] = x_arr[i] + v_half * dt + 0.5 * rhs_v * dt**2
        
        rhs_x, rhs_v = rhs_osc(k, m, x_arr[i + 1], v_half)
        v_half = v_half + 0.5 * rhs_v * dt
        
        v_arr[i + 1] = v_half
    
    return t_arr, x_arr, v_arr

def RungeKutta_Oscillator(k, m, x_0, v_0, t_params):
    """
    Solves the harmonic oscillator problem using the 4th order Runge-Kutta method.

    Parameters:
    k (float): Spring constant
    m (float): Mass
    x_0 (float): Initial position
    v_0 (float): Initial velocity
    t_params (list): Initial and final times [t_i, t_f]
    
    Returns:
    t_arr: Array of time points
    x_arr: Array of positions over time
    v_arr: Array of velocities over time
    """
    t_i, t_f, Npoints = t_params
    dt = (t_f - t_i) / (Npoints - 1)
    
    t_arr = np.linspace(t_i, t_f, Npoints)
    x_arr = np.zeros(Npoints)
    v_arr = np.zeros(Npoints)
    
    x_arr[0] = x_0
    v_arr[0] = v_0
    
    for i in range(Npoints - 1):
        # Compute k1
        rhs_x, rhs_v = rhs_osc(k, m, x_arr[i], v_arr[i])
        k1_x = rhs_x * dt
        k1_v = rhs_v * dt
        
        # Compute k2
        x_half = x_arr[i] + k1_x / 2.0
        v_half = v_arr[i] + k1_v / 2.0
        rhs_x, rhs_v = rhs_osc(k, m, x_half, v_half)
        k2_x = rhs_x * dt
        k2_v = rhs_v * dt
        
        # Compute k3
        x_half = x_arr[i] + k2_x / 2.0
        v_half = v_arr[i] + k2_v / 2.0
        rhs_x, rhs_v = rhs_osc(k, m, x_half, v_half)
        k3_x = rhs_x * dt
        k3_v = rhs_v * dt
        
        # Compute k4
        x_full = x_arr[i] + k3_x
        v_full = v_arr[i] + k3_v
        rhs_x, rhs_v = rhs_osc(k, m, x_full, v_full)
        k4_x = rhs_x * dt
        k4_v = rhs_v * dt
        
        # Update position and velocity
        x_arr[i + 1] = x_arr[i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0
        v_arr[i + 1] = v_arr[i] + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0
    
    return t_arr, x_arr, v_arr
