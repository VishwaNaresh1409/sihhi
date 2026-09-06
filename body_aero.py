"""
Body aerodynamic force calculations and models.

Handles lift and drag of the main fuselage/body.
"""

import math
import config


def get_body_lift_coefficient(angle_of_attack_deg):
    """
    Get lift coefficient for body as function of angle of attack.

    Simplified linear model:
    C_L = C_L_slope * (alpha - alpha_zero_lift)

    Args:
        angle_of_attack_deg: angle of attack in degrees

    Returns:
        lift_coefficient: dimensionless C_L
    """
    alpha_offset = angle_of_attack_deg - config.body_zero_lift_pitch_angle
    c_l = config.body_lift_coefficient_per_angle * alpha_offset

    return c_l


def get_body_drag_coefficient(angle_of_attack_deg):
    """
    Get drag coefficient for body.

    Simplified model: drag is constant (parabolic model could be added).

    Args:
        angle_of_attack_deg: angle of attack in degrees

    Returns:
        drag_coefficient: dimensionless C_D
    """
    # Simple constant model
    c_d = config.body_drag_coefficient

    # Could add parabolic component here if needed:
    # c_d += config.body_drag_parabolic_coefficient * (angle_of_attack_deg ** 2)

    return c_d