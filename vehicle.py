"""
Vehicle state and properties.

Represents the aerodynamic vehicle with position, velocity, orientation,
and kinematic state. Does not calculate forces; that's handled by forces.py.
"""

import math
import numpy as np
import config


class Vehicle:
    """
    Represents a 2D aerodynamic vehicle.

    State vector: [x, y, vx, vy, pitch_angle_rad]
    - x, y: position in meters
    - vx, vy: velocity components in m/s
    - pitch_angle_rad: pitch angle in radians (positive = nose up)
    """

    def __init__(self):
        """Initialize vehicle with config parameters."""

        # Kinematics and dynamics
        self.mass = config.vehicle_mass  # kg
        self.pitch_inertia = config.vehicle_pitch_inertia  # kg*m^2

        # Aerodynamic properties
        self.reference_area = config.vehicle_reference_area  # m^2
        self.reference_length = config.vehicle_reference_length  # m

        # State (will be initialized by reset())
        self.position_x = 0.0  # m
        self.position_y = 0.0  # m
        self.velocity_x = 0.0  # m/s
        self.velocity_y = 0.0  # m/s
        self.pitch_angle = 0.0  # radians
        self.pitch_rate = 0.0  # radians/second

        # Initialize with config values
        self.reset()

    def reset(self):
        """Reset vehicle to initial conditions from config."""
        self.position_x = config.initial_position_x
        self.position_y = config.initial_position_y
        self.velocity_x = config.initial_velocity_x
        self.velocity_y = config.initial_velocity_y
        self.pitch_angle = math.radians(config.initial_pitch_angle)
        self.pitch_rate = 0.0

    def get_state_vector(self):
        """
        Return complete state as a list.

        Returns:
            [x, y, vx, vy, pitch_angle_rad, pitch_rate]
        """
        return [
            self.position_x,
            self.position_y,
            self.velocity_x,
            self.velocity_y,
            self.pitch_angle,
            self.pitch_rate
        ]

    def get_velocity_magnitude(self):
        """Get total velocity (airspeed magnitude)."""
        v_mag = math.sqrt(self.velocity_x ** 2 + self.velocity_y ** 2)
        return v_mag

    def get_velocity_direction(self):
        """
        Get velocity direction in radians.

        Returns angle from horizontal (positive = nose up).
        """
        if self.velocity_x < 1e-6:  # Avoid division by zero
            if self.velocity_y > 0:
                return math.pi / 2.0
            else:
                return -math.pi / 2.0

        direction = math.atan2(self.velocity_y, self.velocity_x)
        return direction

    def get_angle_of_attack(self):
        """
        Calculate angle of attack (angle between velocity and pitch angle).

        Simplified calculation assuming velocity vector represents aerodynamic reference.

        Returns:
            angle_of_attack in radians
        """
        velocity_direction = self.get_velocity_direction()
        angle_of_attack = self.pitch_angle - velocity_direction

        # Normalize to [-pi, pi]
        while angle_of_attack > math.pi:
            angle_of_attack -= 2.0 * math.pi
        while angle_of_attack < -math.pi:
            angle_of_attack += 2.0 * math.pi

        return angle_of_attack

    def update_kinematics(self, accel_x, accel_y, angular_accel, dt):
        """
        Update vehicle state using kinematic equations.

        Uses simple Euler integration.

        Args:
            accel_x: acceleration in X direction (m/s^2)
            accel_y: acceleration in Y direction (m/s^2)
            angular_accel: angular acceleration about pitch axis (rad/s^2)
            dt: timestep in seconds
        """
        # Update velocities from accelerations
        self.velocity_x += accel_x * dt
        self.velocity_y += accel_y * dt

        # Update positions from velocities
        self.position_x += self.velocity_x * dt
        self.position_y += self.velocity_y * dt

        # Update pitch rate and angle
        self.pitch_rate += angular_accel * dt
        self.pitch_angle += self.pitch_rate * dt

    def get_altitude(self):
        """Get current altitude (Y position)."""
        return self.position_y

    def get_position(self):
        """Get current position as (x, y) tuple."""
        return (self.position_x, self.position_y)