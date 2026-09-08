"""
Dynamics integration and time-stepping for the vehicle simulation.

Implements the main numerical integration loop that advances vehicle state
based on forces and accelerations, with aerodynamic pitch damping.
"""

import math
import config
from vehicle import Vehicle
from forces import calculate_total_force

class VehicleDynamics:
    """
    Manages vehicle dynamics integration and state updates.
    """

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.current_time = 0.0
        self.wind_x = 0.0
        self.wind_y = 0.0

    def set_wind(self, wind_x, wind_y):
        self.wind_x = wind_x
        self.wind_y = wind_y

    def calculate_accelerations(self, total_force_x, total_force_y):
        mass = self.vehicle.mass
        accel_x = total_force_x / mass
        accel_y = total_force_y / mass
        return accel_x, accel_y

    def calculate_pitch_torque(self, surface_angles):
        """
        Calculate net torque about the pitch axis from control surface deflections
        and apply aerodynamic rotational damping.
        """
        airspeed_x = self.vehicle.velocity_x - self.wind_x
        airspeed_y = self.vehicle.velocity_y - self.wind_y
        airspeed_mag = math.sqrt(airspeed_x ** 2 + airspeed_y ** 2)

        if airspeed_mag < 0.1:
            return 0.0

        dynamic_pressure = 0.5 * config.air_density * airspeed_mag ** 2
        moment_arm = config.vehicle_reference_length * 0.4

        total_torque = 0.0
        for angle_rad in surface_angles:
            angle_deg = math.degrees(angle_rad)
            cl = config.control_surface_lift_coefficient_per_angle * angle_deg
            lift = dynamic_pressure * cl * config.control_surface_area
            total_torque += lift * moment_arm

        # Add physical aerodynamic pitch damping opposing rotational rate
        pitch_damping_torque = 4.0 * self.vehicle.pitch_rate
        total_torque -= pitch_damping_torque

        # Ensure a float is always returned to prevent NoneType math errors
        return float(total_torque)

    def step(self, dt, surface_angles):
        altitude = self.vehicle.get_altitude()

        total_fx, total_fy = calculate_total_force(
            self.vehicle,
            surface_angles,
            altitude,
            wind_x=self.wind_x,
            wind_y=self.wind_y
        )

        accel_x, accel_y = self.calculate_accelerations(total_fx, total_fy)
        pitch_torque = self.calculate_pitch_torque(surface_angles)

        # Fallback safety catch
        if pitch_torque is None:
            pitch_torque = 0.0

        angular_accel = pitch_torque / self.vehicle.pitch_inertia

        self.vehicle.update_kinematics(accel_x, accel_y, angular_accel, dt)
        self.current_time += dt

    def reset(self):
        self.vehicle.reset()
        self.current_time = 0.0
        self.wind_x = 0.0
        self.wind_y = 0.0