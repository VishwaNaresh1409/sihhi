"""
Actuator models for control surface movement.

Simulates rate-limited, potentially delayed surface movements.
"""

import math
import config


class ControlSurfaceActuator:
    """
    Represents a single control surface actuator.

    Controls movement of a control surface from commanded angle to actual angle.
    Implements rate limiting and optional response delay.
    """

    def __init__(self, surface_index=0):
        """
        Initialize actuator.

        Args:
            surface_index: identifier for this surface (for tracking)
        """
        self.surface_index = surface_index
        self.current_angle_rad = 0.0  # Current actual angle
        self.commanded_angle_rad = 0.0  # Desired angle from controller

        self.max_speed = math.radians(config.actuator_max_speed)  # rad/s
        self.response_delay = config.actuator_response_delay  # seconds
        self.damping = config.actuator_damping  # 0 to 1

        self.angle_rate = 0.0  # rad/s - rate of change

    def set_command(self, commanded_angle_rad):
        """
        Set the desired angle for this surface.

        Args:
            commanded_angle_rad: desired angle in radians
        """
        # Limit command to max angle
        max_angle_rad = math.radians(config.control_surface_max_angle)
        min_angle_rad = math.radians(config.control_surface_min_angle)

        self.commanded_angle_rad = max(min_angle_rad, min(max_angle_rad, commanded_angle_rad))

    def step(self, dt):
        """
        Update actuator position based on commanded angle.

        Implements rate limiting so surface doesn't move instantaneously.

        Args:
            dt: timestep in seconds
        """
        # Error between current and commanded position
        angle_error = self.commanded_angle_rad - self.current_angle_rad

        # Desired rate to reach commanded angle
        desired_rate = angle_error / max(dt, 0.01)  # Avoid division by zero

        # Limit rate to maximum speed
        rate_limited = max(-self.max_speed, min(self.max_speed, desired_rate))

        # Apply damping to smooth motion
        # damping=0: no damping, rate changes instantly
        # damping=1: critically damped, rate changes smoothly
        self.angle_rate = (1.0 - self.damping) * rate_limited + self.damping * self.angle_rate

        # Update angle
        self.current_angle_rad += self.angle_rate * dt

        # Ensure angle stays within limits
        max_angle_rad = math.radians(config.control_surface_max_angle)
        min_angle_rad = math.radians(config.control_surface_min_angle)
        self.current_angle_rad = max(min_angle_rad, min(max_angle_rad, self.current_angle_rad))

    def get_angle(self):
        """
        Get current actual surface angle.

        Returns:
            angle_rad: actual surface angle in radians
        """
        return self.current_angle_rad