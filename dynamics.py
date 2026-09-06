"""
Dynamics integration and time-stepping for the vehicle simulation.

Implements the main numerical integration loop that advances vehicle state
based on forces and accelerations.
"""

import math
import numpy as np
import config
from vehicle import Vehicle
from forces import calculate_total_force, calculate_gravity_force


class VehicleDynamics:
    """
    Manages vehicle dynamics integration and state updates.

    Separates the physics calculation from state management.
    """

    def __init__(self, vehicle):
        """
        Initialize dynamics simulator.

        Args:
            vehicle: Vehicle object to simulate
        """
        self.vehicle = vehicle
        self.current_time = 0.0

    def calculate_accelerations(self, total_force_x, total_force_y):
        """
        Calculate linear accelerations from forces using F=ma.

        Args:
            total_force_x: total force in X direction (Newtons)
            total_force_y: total force in Y direction (Newtons)

        Returns:
            accel_x, accel_y: accelerations in m/s^2
        """
        mass = self.vehicle.mass

        accel_x = total_force_x / mass
        accel_y = total_force_y / mass

        return accel_x, accel_y

    def step(self, dt, surface_angles):
        """
        Advance vehicle state by one timestep.

        This is the main physics integration function.

        Process:
        1. Calculate forces based on current state
        2. Calculate accelerations from forces
        3. Update velocities and positions
        4. Update orientation (pitch angle)

        Args:
            dt: timestep in seconds
            surface_angles: list of current control surface angles in radians
        """
        # Get current altitude for atmospheric calculations
        altitude = self.vehicle.get_altitude()

        # Calculate total force on vehicle
        total_fx, total_fy = calculate_total_force(
            self.vehicle,
            surface_angles,
            altitude
        )

        # Calculate linear accelerations
        accel_x, accel_y = self.calculate_accelerations(total_fx, total_fy)

        # For pitch angle, we need rotational dynamics
        # Simplified: no rotational dynamics in this version
        # Pitch angle follows from control surface positions or pilot input
        # If you want to add rotational inertia, implement a torque calculation
        angular_accel = 0.0  # rad/s^2 (could be calculated from surface moments)

        # Update vehicle kinematics
        self.vehicle.update_kinematics(accel_x, accel_y, angular_accel, dt)

        # Update time
        self.current_time += dt

    def reset(self):
        """Reset dynamics to initial state."""
        self.vehicle.reset()
        self.current_time = 0.0