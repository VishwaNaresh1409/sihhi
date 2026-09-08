"""
Aerodynamic and environmental force calculations.

Computes lift, drag, and other forces acting on the vehicle and control surfaces.
All aerodynamic forces use AIRSPEED (velocity relative to wind), not ground speed.

FIXES:
- All aero functions now accept wind_x, wind_y parameters
- Lift/drag computed from airspeed vector, not inertial velocity vector
- calculate_total_force passes wind through to body and surface calculations
"""

import math
import numpy as np
import config
from atmosphere import get_air_density


def calculate_drag_force(airspeed_magnitude, reference_area, drag_coefficient):
    """
    Calculate aerodynamic drag force magnitude.

    Drag equation: F_D = 0.5 * rho * V_air^2 * C_D * A

    Args:
        airspeed_magnitude: airspeed (velocity relative to air/wind) in m/s
        reference_area: reference planform area in m^2
        drag_coefficient: dimensionless drag coefficient

    Returns:
        drag_force: magnitude of drag force in Newtons
    """
    air_density = config.air_density
    dynamic_pressure = 0.5 * air_density * airspeed_magnitude ** 2
    drag_force = dynamic_pressure * drag_coefficient * reference_area
    return drag_force


def calculate_lift_force(airspeed_magnitude, reference_area, lift_coefficient):
    """
    Calculate aerodynamic lift force magnitude.

    Lift equation: F_L = 0.5 * rho * V_air^2 * C_L * A

    Args:
        airspeed_magnitude: airspeed in m/s
        reference_area: reference planform area in m^2
        lift_coefficient: dimensionless lift coefficient

    Returns:
        lift_force: magnitude of lift force in Newtons
    """
    air_density = config.air_density
    dynamic_pressure = 0.5 * air_density * airspeed_magnitude ** 2
    lift_force = dynamic_pressure * lift_coefficient * reference_area
    return lift_force


def calculate_body_aerodynamic_forces(vehicle_state, altitude, wind_x=0.0, wind_y=0.0):
    """
    Calculate lift and drag forces on the main vehicle body.

    Uses airspeed (velocity relative to wind) for all aerodynamic calculations.
    Lift is perpendicular to airspeed vector; drag opposes airspeed vector.

    Args:
        vehicle_state: Vehicle object with current kinematics
        altitude: current altitude in meters
        wind_x: wind velocity in X direction (m/s)
        wind_y: wind velocity in Y direction (m/s)

    Returns:
        force_x, force_y: force components in Newtons (inertial frame)
    """
    # Airspeed = vehicle velocity - wind velocity
    airspeed_x = vehicle_state.velocity_x - wind_x
    airspeed_y = vehicle_state.velocity_y - wind_y
    airspeed_magnitude = math.sqrt(airspeed_x ** 2 + airspeed_y ** 2)

    if airspeed_magnitude < 0.1:
        return 0.0, 0.0

    # Direction of airspeed vector (angle from horizontal)
    airspeed_direction = math.atan2(airspeed_y, airspeed_x)

    # Angle of attack = pitch angle - airspeed direction
    # Angle of attack = pitch angle - airspeed direction
    angle_of_attack = vehicle_state.pitch_angle - airspeed_direction
    while angle_of_attack > math.pi:
        angle_of_attack -= 2.0 * math.pi
    while angle_of_attack < -math.pi:
        angle_of_attack += 2.0 * math.pi

    angle_of_attack_deg = math.degrees(angle_of_attack)
    effective_aoa_deg = max(-15.0, min(15.0, angle_of_attack_deg))

    c_l = config.body_lift_coefficient_per_angle * effective_aoa_deg
    c_d = config.body_drag_coefficient

    drag_magnitude = calculate_drag_force(airspeed_magnitude, config.vehicle_reference_area, c_d)
    lift_magnitude = calculate_lift_force(airspeed_magnitude, config.vehicle_reference_area, c_l)

    # Drag opposes airspeed direction
    drag_force_x = -drag_magnitude * math.cos(airspeed_direction)
    drag_force_y = -drag_magnitude * math.sin(airspeed_direction)

    # Lift is perpendicular to airspeed (90 degrees counterclockwise)
    lift_direction = airspeed_direction + math.pi / 2.0
    lift_force_x = lift_magnitude * math.cos(lift_direction)
    lift_force_y = lift_magnitude * math.sin(lift_direction)

    force_x = drag_force_x + lift_force_x
    force_y = drag_force_y + lift_force_y

    return force_x, force_y


def calculate_control_surface_force(vehicle_state, surface_angle_rad, altitude,
                                    wind_x=0.0, wind_y=0.0):
    """
    Calculate force from a single control surface.

    Surface force depends on airspeed (wind-relative velocity) and deflection angle.

    Args:
        vehicle_state: Vehicle object
        surface_angle_rad: deflection angle in radians
        altitude: current altitude in meters
        wind_x: wind velocity in X (m/s)
        wind_y: wind velocity in Y (m/s)

    Returns:
        force_x, force_y: force components in Newtons (inertial frame)
    """
    airspeed_x = vehicle_state.velocity_x - wind_x
    airspeed_y = vehicle_state.velocity_y - wind_y
    airspeed_magnitude = math.sqrt(airspeed_x ** 2 + airspeed_y ** 2)

    if airspeed_magnitude < 0.1:
        return 0.0, 0.0

    airspeed_direction = math.atan2(airspeed_y, airspeed_x)

    surface_angle_deg = math.degrees(surface_angle_rad)
    c_l_surface = config.control_surface_lift_coefficient_per_angle * surface_angle_deg
    c_d_surface = config.control_surface_drag_coefficient_increase * (surface_angle_deg ** 2)

    drag_magnitude = calculate_drag_force(airspeed_magnitude, config.control_surface_area, c_d_surface)
    lift_magnitude = calculate_lift_force(airspeed_magnitude, config.control_surface_area, c_l_surface)

    drag_force_x = -drag_magnitude * math.cos(airspeed_direction)
    drag_force_y = -drag_magnitude * math.sin(airspeed_direction)

    lift_direction = airspeed_direction + math.pi / 2.0
    lift_force_x = lift_magnitude * math.cos(lift_direction)
    lift_force_y = lift_magnitude * math.sin(lift_direction)

    force_x = drag_force_x + lift_force_x
    force_y = drag_force_y + lift_force_y

    return force_x, force_y


def calculate_gravity_force(vehicle_mass):
    """
    Calculate gravitational force.

    Args:
        vehicle_mass: mass in kg

    Returns:
        force_x, force_y: force components in Newtons (force_y negative = downward)
    """
    force_x = 0.0
    force_y = -config.gravity * vehicle_mass
    return force_x, force_y


def calculate_total_force(vehicle_state, surface_angles, altitude,
                          wind_x=0.0, wind_y=0.0):
    """
    Calculate total force on the vehicle.

    Sums body aerodynamics, control surface forces, and gravity.
    All aerodynamic terms use airspeed (wind-relative velocity).

    Args:
        vehicle_state: Vehicle object
        surface_angles: list of control surface deflection angles in radians
        altitude: current altitude in meters
        wind_x: wind X velocity (m/s)
        wind_y: wind Y velocity (m/s)

    Returns:
        total_force_x, total_force_y: total force components in Newtons
    """
    # Body aerodynamics (uses airspeed)
    aero_fx, aero_fy = calculate_body_aerodynamic_forces(
        vehicle_state, altitude, wind_x=wind_x, wind_y=wind_y
    )

    # Control surface forces (uses airspeed)
    surface_fx_total = 0.0
    surface_fy_total = 0.0
    for surface_angle in surface_angles:
        surf_fx, surf_fy = calculate_control_surface_force(
            vehicle_state, surface_angle, altitude,
            wind_x=wind_x, wind_y=wind_y
        )
        surface_fx_total += surf_fx
        surface_fy_total += surf_fy

    # Gravity
    grav_fx, grav_fy = calculate_gravity_force(vehicle_state.mass)

    total_fx = aero_fx + surface_fx_total + grav_fx
    total_fy = aero_fy + surface_fy_total + grav_fy

    return total_fx, total_fy