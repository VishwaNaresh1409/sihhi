"""
Ballistic projectile motion calculator.

Solves for launch angle and velocity needed to hit a target,
assuming NO aerodynamic effects (simple projectile motion).

This is intentionally simplified - it WILL miss in reality
because it ignores drag.
"""

import math
import config


def calculate_ballistic_solution(target_x, target_y, launch_x=0.0, launch_y=0.0,
                                 initial_velocity_magnitude=None):
    """
    Calculate launch angle to hit a target using ballistic equations.

    Solves: y = x*tan(θ) - (g*x²)/(2*v²*cos²(θ))

    Ignores aerodynamic drag completely.

    Args:
        target_x: target horizontal position (m)
        target_y: target altitude (m)
        launch_x: launch horizontal position (m)
        launch_y: launch altitude (m)
        initial_velocity_magnitude: launch speed (m/s), if None uses config reference

    Returns:
        dict with keys:
        - 'launch_angle_rad': launch angle in radians (positive = nose up)
        - 'launch_angle_deg': launch angle in degrees
        - 'launch_velocity': launch speed in m/s
        - 'time_to_target': predicted flight time in seconds
        - 'success': whether a solution was found
        - 'reason': explanation if no solution
    """

    if initial_velocity_magnitude is None:
        initial_velocity_magnitude = config.reference_velocity_x

    # Relative target position
    dx = target_x - launch_x
    dy = target_y - launch_y

    g = config.gravity
    v = initial_velocity_magnitude

    # Special case: target directly overhead/below
    if dx < 0.1:
        return {
            'success': False,
            'reason': 'Target behind launch point',
            'launch_angle_rad': 0.0,
            'launch_angle_deg': 0.0,
            'launch_velocity': v,
            'time_to_target': 0.0
        }

    # Ballistic equation: y = x*tan(θ) - (g*x²)/(2*v²*cos²(θ))
    # Rearrange to standard form and solve using quadratic formula

    # Let tan(θ) = t, then: y = x*t - (g*x²)/(2*v²*(1+t²))
    # Multiply by (1+t²): y*(1+t²) = x*t*(1+t²) - (g*x²)/(2*v²)
    # Expand: y + y*t² = x*t + x*t³ - (g*x²)/(2*v²)
    # Rearrange: x*t³ + t²*(y-x²*g/(2*v²)) + t*x - y = 0

    # For simplicity, use numerical method (trial and error for launch angle)
    best_angle = None
    best_error = float('inf')

    # Try angles from -45° to +90°
    for angle_deg in range(-45, 91):
        angle_rad = math.radians(angle_deg)

        # Calculate where projectile lands
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        if cos_a == 0:
            continue

        # Ballistic trajectory formula
        # y = x*tan(θ) - (g*x²)/(2*v²*cos²(θ))
        predicted_y = dx * math.tan(angle_rad) - (g * dx ** 2) / (2 * v ** 2 * cos_a ** 2)

        # Error from target
        error = abs(predicted_y - dy)

        if error < best_error:
            best_error = error
            best_angle = angle_rad

    if best_angle is None:
        return {
            'success': False,
            'reason': 'No ballistic solution found',
            'launch_angle_rad': 0.0,
            'launch_angle_deg': 0.0,
            'launch_velocity': v,
            'time_to_target': 0.0
        }

    # Calculate time to target using: x = v*cos(θ)*t
    time_to_target = dx / (v * math.cos(best_angle))

    if time_to_target <= 0:
        return {
            'success': False,
            'reason': 'Negative time to target',
            'launch_angle_rad': best_angle,
            'launch_angle_deg': math.degrees(best_angle),
            'launch_velocity': v,
            'time_to_target': time_to_target
        }

    return {
        'success': True,
        'reason': 'Solution found',
        'launch_angle_rad': best_angle,
        'launch_angle_deg': math.degrees(best_angle),
        'launch_velocity': v,
        'time_to_target': time_to_target,
        'predicted_range': dx,
        'predicted_final_alt': launch_y + best_angle,
    }


def calculate_ballistic_trajectory(launch_x, launch_y, launch_velocity,
                                   launch_angle_rad, num_points=100):
    """
    Calculate ballistic trajectory (no drag) for visualization.

    Args:
        launch_x: launch X position (m)
        launch_y: launch Y position (m)
        launch_velocity: launch speed (m/s)
        launch_angle_rad: launch angle (radians)
        num_points: number of trajectory points

    Returns:
        tuple of (x_list, y_list) trajectory points
    """
    g = config.gravity
    v = launch_velocity
    theta = launch_angle_rad

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # Time to apex (peak altitude)
    t_apex = (v * sin_theta) / g

    # Total flight time (when y returns to launch altitude)
    # Using: y = y0 + v*sin(θ)*t - 0.5*g*t²
    # At ground: 0 = 0 + v*sin(θ)*t - 0.5*g*t²
    # t = 2*v*sin(θ)/g
    t_total = 2 * t_apex

    # Extend time slightly beyond landing
    t_max = t_total * 1.2

    times = np.linspace(0, t_max, num_points)

    x_list = []
    y_list = []

    for t in times:
        x = launch_x + v * cos_theta * t
        y = launch_y + v * sin_theta * t - 0.5 * g * t ** 2

        x_list.append(x)
        y_list.append(y)

    return x_list, y_list


import numpy as np