"""
Control surface aerodynamic force calculations.

Handles lift and drag of control surfaces (elevators, flaps, etc.).
"""

import math
import config


def get_surface_lift_coefficient(deflection_angle_deg):
    """
    Get lift coefficient of control surface as function of deflection.

    Simplified linear model relating surface angle to lift.

    Args:
        deflection_angle_deg: surface deflection angle in degrees

    Returns:
        lift_coefficient: dimensionless C_L
    """
    c_l = config.control_surface_lift_coefficient_per_angle * deflection_angle_deg

    return c_l


def get_surface_drag_coefficient(deflection_angle_deg):
    """
    Get drag coefficient increase from control surface deflection.

    Deflecting a surface increases drag (parabolic relationship).

    Args:
        deflection_angle_deg: surface deflection angle in degrees

    Returns:
        additional_drag_coefficient: increase in C_D due to deflection
    """
    # Parabolic increase in drag with deflection
    c_d_delta = config.control_surface_drag_coefficient_increase * (deflection_angle_deg ** 2)

    return c_d_delta