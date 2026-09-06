"""
Aerodynamic and environmental force calculations.

Computes lift, drag, and other forces acting on the vehicle and control surfaces.
All calculations are explicit and physics-based.
"""

import math
import numpy as np
import config
from atmosphere import get_air_density


def calculate_drag_force(velocity_magnitude, reference_area, drag_coefficient):
    """
    Calculate aerodynamic drag force magnitude.

    Drag equation: F_D = 0.5 * rho * V^2 * C_D * A

    Args:
        velocity_magnitude: total velocity in m/s
        reference_area: reference planform area in m^2
        drag_coefficient: dimensionless drag coefficient

    Returns:
        drag_force: magnitude of drag force in Newtons
    """
    air_density = config.air_density  # kg/m^3

    # Dynamic pressure
    dynamic_pressure = 0.5 * air_density * velocity_magnitude ** 2

    # Drag force (always opposes velocity)
    drag_force = dynamic_pressure * drag_coefficient * reference_area

    return drag_force


def calculate_lift_force(velocity_magnitude, reference_area, lift_coefficient):
    """
    Calculate aerodynamic lift force magnitude.

    Lift equation: F_L = 0.5 * rho * V^2 * C_L * A

    Args:
        velocity_magnitude: total velocity in m/s
        reference_area: reference planform area in m^2
        lift_coefficient: dimensionless lift coefficient

    Returns:
        lift_force: magnitude of lift force in Newtons
    """
    air_density = config.air_density  # kg/m^3

    # Dynamic pressure
    dynamic_pressure = 0.5 * air_density * velocity_magnitude ** 2

    # Lift force
    lift_force = dynamic_pressure * lift_coefficient * reference_area

    return lift_force


def calculate_body_aerodynamic_forces(vehicle_state, altitude):
    """
    Calculate lift and drag forces on the main vehicle body.

    Uses simplified linear aerodynamic model:
    - Drag coefficient is constant
    - Lift coefficient varies linearly with angle of attack

    Args:
        vehicle_state: Vehicle object with current kinematics
        altitude: current altitude in meters

    Returns:
        force_x, force_y: force components in Newtons
    """
    # Get velocity
    velocity_x = vehicle_state.velocity_x
    velocity_y = vehicle_state.velocity_y
    velocity_magnitude = vehicle_state.get_velocity_magnitude()
    velocity_direction = vehicle_state.get_velocity_direction()

    # Avoid division by zero
    if velocity_magnitude < 0.1:
        return 0.0, 0.0

    # Calculate angle of attack
    angle_of_attack = vehicle_state.get_angle_of_attack()
    angle_of_attack_deg = math.degrees(angle_of_attack)

    # Lift coefficient (linear with angle of attack)
    c_l = config.body_lift_coefficient_per_angle * angle_of_attack_deg

    # Drag coefficient (constant for this simplified model)
    c_d = config.body_drag_coefficient

    # Calculate forces
    drag_magnitude = calculate_drag_force(
        velocity_magnitude,
        config.vehicle_reference_area,
        c_d
    )

    lift_magnitude = calculate_lift_force(
        velocity_magnitude,
        config.vehicle_reference_area,
        c_l
    )

    # Drag acts opposite to velocity direction
    drag_force_x = -drag_magnitude * math.cos(velocity_direction)
    drag_force_y = -drag_magnitude * math.sin(velocity_direction)

    # Lift acts perpendicular to velocity direction (90 degrees counterclockwise)
    # Perpendicular direction: rotate velocity angle by 90 degrees
    lift_direction = velocity_direction + math.pi / 2.0
    lift_force_x = lift_magnitude * math.cos(lift_direction)
    lift_force_y = lift_magnitude * math.sin(lift_direction)

    # Total body aerodynamic force
    force_x = drag_force_x + lift_force_x
    force_y = drag_force_y + lift_force_y

    return force_x, force_y


def calculate_control_surface_force(vehicle_state, surface_angle_rad, altitude):
    """
    Calculate force from a single control surface.

    Surface force depends on:
    - Current airflow (velocity magnitude and direction)
    - Surface deflection angle

    Args:
        vehicle_state: Vehicle object
        surface_angle_rad: deflection angle in radians
        altitude: current altitude in meters

    Returns:
        force_x, force_y: force components in Newtons
    """
    velocity_magnitude = vehicle_state.get_velocity_magnitude()
    velocity_direction = vehicle_state.get_velocity_direction()

    # Avoid division by zero
    if velocity_magnitude < 0.1:
        return 0.0, 0.0

    # Lift coefficient of control surface depends on its deflection angle
    surface_angle_deg = math.degrees(surface_angle_rad)
    c_l_surface = config.control_surface_lift_coefficient_per_angle * surface_angle_deg

    # Additional drag from surface deflection
    c_d_surface = config.control_surface_drag_coefficient_increase * (surface_angle_deg ** 2)

    # Calculate forces
    drag_magnitude = calculate_drag_force(
        velocity_magnitude,
        config.control_surface_area,
        c_d_surface
    )

    lift_magnitude = calculate_lift_force(
        velocity_magnitude,
        config.control_surface_area,
        c_l_surface
    )

    # Drag acts opposite to velocity
    drag_force_x = -drag_magnitude * math.cos(velocity_direction)
    drag_force_y = -drag_magnitude * math.sin(velocity_direction)

    # Lift acts perpendicular to velocity
    lift_direction = velocity_direction + math.pi / 2.0
    lift_force_x = lift_magnitude * math.cos(lift_direction)
    lift_force_y = lift_magnitude * math.sin(lift_direction)

    # Total surface force
    force_x = drag_force_x + lift_force_x
    force_y = drag_force_y + lift_force_y

    return force_x, force_y


def calculate_gravity_force(vehicle_mass):
    """
    Calculate gravitational force.

    Args:
        vehicle_mass: mass in kg

    Returns:
        force_x, force_y: force components in Newtons (force_y is negative/downward)
    """
    force_x = 0.0
    force_y = -config.gravity * vehicle_mass  # Negative because gravity acts downward

    return force_x, force_y


def calculate_total_force(vehicle_state, surface_angles, altitude):
    """
    Calculate total force on the vehicle.

    Sums:
    - Body aerodynamic forces
    - Control surface forces
    - Gravitational force
    - (Environmental disturbances would be added here)

    Args:
        vehicle_state: Vehicle object
        surface_angles: list of control surface deflection angles in radians
        altitude: current altitude in meters

    Returns:
        total_force_x, total_force_y: total force components in Newtons
    """
    # Body aerodynamics
    aero_fx, aero_fy = calculate_body_aerodynamic_forces(vehicle_state, altitude)

    # Control surfaces
    surface_fx_total = 0.0
    surface_fy_total = 0.0
    for surface_angle in surface_angles:
        surf_fx, surf_fy = calculate_control_surface_force(
            vehicle_state, surface_angle, altitude
        )
        surface_fx_total += surf_fx
        surface_fy_total += surf_fy

    # Gravity
    grav_fx, grav_fy = calculate_gravity_force(vehicle_state.mass)

    # Total force
    total_fx = aero_fx + surface_fx_total + grav_fx
    total_fy = aero_fy + surface_fy_total + grav_fy

    return total_fx, total_fy