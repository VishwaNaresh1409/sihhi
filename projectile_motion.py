"""
Ballistic projectile motion calculator.

Solves for launch angle using closed-form quadratic solution.
Falls back to fine-grained numerical search only if closed form fails.

FIXES:
- Closed-form solution instead of 1-degree integer search
- Added validity check: warns if target is beyond max ballistic range
- Picks flatter (lower-energy) of the two solution angles
"""

import math
import numpy as np
import config

def calculate_ballistic_solution(target_x, target_y, launch_x=0.0, launch_y=0.0,
                                 initial_velocity_magnitude=None):
    """
    Calculate launch angle to hit a target using ballistic equations.
    Ignores aerodynamic drag completely.
    """
    if initial_velocity_magnitude is None:
        initial_velocity_magnitude = config.reference_velocity_x

    dx = target_x - launch_x
    dy = target_y - launch_y
    g = config.gravity
    v = initial_velocity_magnitude

    # Validity check: Is target beyond maximum possible ballistic range?
    max_range = (v ** 2) / g
    if dx > max_range:
        return {
            'success': False,
            'reason': f'Target ({dx}m) is beyond max ballistic range ({max_range:.1f}m) for v={v}m/s',
            'launch_angle_rad': 0.0,
            'launch_angle_deg': 0.0,
            'launch_velocity': v,
            'time_to_target': 0.0,
            'predicted_range': max_range,
        }

    if dx < 0.1:
        return {
            'success': False,
            'reason': 'Target behind or at launch point',
            'launch_angle_rad': 0.0,
            'launch_angle_deg': 0.0,
            'launch_velocity': v,
            'time_to_target': 0.0,
            'predicted_range': dx,
        }

        # Check if target is physically reachable in a vacuum
        max_ballistic_range = (v ** 2) / g
        if dx > max_ballistic_range:
            return {
                "success": False,
                "reason": (
                    f"Target ({dx:.1f}m) exceeds max ballistic range"
                    f" ({max_ballistic_range:.1f}m) at {v}m/s"
                ),
                "launch_angle_rad": 0.0,
                "launch_angle_deg": 0.0,
                "launch_velocity": v,
                "time_to_target": 0.0,
                "predicted_range": max_ballistic_range,
            }

    # Closed-form ballistic solution
    # y = x*tan(θ) - g*x²/(2v²cos²θ)
    # Using 1/cos²θ = 1 + tan²(θ), we get a quadratic in terms of tan(θ):
    A = (g * dx ** 2) / (2.0 * v ** 2)
    B = dx
    C = dy + (g * dx ** 2) / (2.0 * v ** 2)

    discriminant = B ** 2 - 4.0 * A * C

    best_angle = None
    best_time = 0.0

    if discriminant >= 0:
        sqrt_disc = math.sqrt(discriminant)
        t1 = (B + sqrt_disc) / (2.0 * A)
        t2 = (B - sqrt_disc) / (2.0 * A)

        angle1 = math.atan(t1)
        angle2 = math.atan(t2)

        candidates = []
        for angle in [angle1, angle2]:
            cos_a = math.cos(angle)
            if cos_a > 0.01:
                t_flight = dx / (v * cos_a)
                if t_flight > 0:
                    candidates.append((angle, t_flight))

        if candidates:
            # Choose the flatter trajectory (smaller absolute angle) to minimize drag error
            candidates.sort(key=lambda x: abs(x[0]))
            best_angle, best_time = candidates[0]

    # Fallback: fine-grained numerical search if closed-form fails mathematically
    if best_angle is None:
        best_error = float('inf')
        for angle_deg_100 in range(-4500, 9001):
            angle_rad = math.radians(angle_deg_100 / 100.0)
            cos_a = math.cos(angle_rad)
            if cos_a < 0.01:
                continue
            predicted_y = dx * math.tan(angle_rad) - (g * dx ** 2) / (2 * v ** 2 * cos_a ** 2)
            error = abs(predicted_y - dy)
            if error < best_error:
                best_error = error
                best_angle = angle_rad
                best_time = dx / (v * cos_a)

    if best_angle is None or best_time <= 0:
        return {
            'success': False,
            'reason': 'No valid trajectory found',
            'launch_angle_rad': 0.0,
            'launch_angle_deg': 0.0,
            'launch_velocity': v,
            'time_to_target': 0.0,
            'predicted_range': dx,
        }

    return {
        'success': True,
        'reason': 'Solution found',
        'launch_angle_rad': best_angle,
        'launch_angle_deg': math.degrees(best_angle),
        'launch_velocity': v,
        'time_to_target': best_time,
        'predicted_range': dx,
        'predicted_final_alt': launch_y + dy,
    }


def calculate_ballistic_trajectory(launch_x, launch_y, launch_velocity,
                                   launch_angle_rad, num_points=100):
    """
    Calculate full ballistic trajectory points for plotting.
    """
    g = config.gravity
    v = launch_velocity
    theta = launch_angle_rad

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    if abs(sin_theta) < 1e-6:
        t_total = 2.0
    else:
        t_apex = (v * sin_theta) / g
        t_total = 2.0 * t_apex

    t_max = max(t_total * 1.2, 5.0)

    times = np.linspace(0, t_max, num_points)
    x_list = []
    y_list = []

    for t in times:
        x = launch_x + v * cos_theta * t
        y = launch_y + v * sin_theta * t - 0.5 * g * t ** 2
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list