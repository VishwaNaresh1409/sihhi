"""
Main control system for the aerodynamic vehicle.

Implements cascaded control loops:
1. Altitude control loop (outer)
2. Pitch angle control loop (inner)
3. Surface actuation
"""

import math
import config
from pid import PIDController
from actuator import ControlSurfaceActuator


class VehicleController:
    """
    Hierarchical control system for the vehicle.

    Control architecture:
    - Altitude error -> Altitude PID -> Desired pitch command
    - Pitch error -> Pitch PID -> Surface deflection command
    - Surface command -> Actuator dynamics -> Actual surface position
    """

    def __init__(self, num_surfaces=config.num_control_surfaces):
        """
        Initialize control system.

        Args:
            num_surfaces: number of control surfaces
        """
        self.num_surfaces = num_surfaces

        # Altitude control loop (outer loop)
        self.altitude_controller = PIDController(
            kp=config.altitude_controller_kp,
            ki=config.altitude_controller_ki,
            kd=config.altitude_controller_kd,
            output_limit=config.altitude_controller_output_limit,
            integral_limit=config.pid_integral_limit
        )

        # Pitch control loop (inner loop)
        self.pitch_controller = PIDController(
            kp=config.pitch_controller_kp,
            ki=config.pitch_controller_ki,
            kd=config.pitch_controller_kd,
            output_limit=config.pitch_controller_output_limit,
            integral_limit=config.pid_integral_limit
        )

        # Actuators for each surface
        self.actuators = [
            ControlSurfaceActuator(i) for i in range(num_surfaces)
        ]

        # State tracking
        self.is_enabled = False
        self.commanded_pitch = 0.0  # degrees
        self.commanded_surface_angles = [0.0] * num_surfaces  # radians

    def enable(self):
        """Enable the control system."""
        self.is_enabled = True
        self.altitude_controller.reset()
        self.pitch_controller.reset()

    def disable(self):
        """Disable the control system."""
        self.is_enabled = False

    def update(self, vehicle_state, measured_altitude, measured_surface_angles, dt,
               target_position=None, guidance_trajectory=None, guidance_mode="altitude"):
        """
        Main control update function with guidance modes.

        Args:
            vehicle_state: Vehicle object with current state
            measured_altitude: measured current altitude
            measured_surface_angles: list of measured surface angles
            dt: timestep in seconds
            target_position: (x, y) target for trajectory guidance
            guidance_trajectory: reference trajectory to follow
            guidance_mode: "altitude" (maintain), "trajectory" (follow path), or "terminal" (chase target)
        """
        if not self.is_enabled:
            return False

        # ========== MODE 1: ALTITUDE HOLD (default) ==========
        if guidance_mode == "altitude" or target_position is None:
            altitude_error = config.reference_altitude - measured_altitude
            pitch_command_deg = self.altitude_controller.update(altitude_error, dt)
            self.commanded_pitch = pitch_command_deg

        # ========== MODE 2: TRAJECTORY FOLLOWING ==========
        elif guidance_mode == "trajectory" and guidance_trajectory is not None:
            # Follow a pre-calculated trajectory to target
            current_x = vehicle_state.position_x

            # Find closest point on trajectory
            closest_idx = 0
            closest_dist = float('inf')
            for i, (traj_x, traj_y) in enumerate(guidance_trajectory):
                dist = abs(traj_x - current_x)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i

            # Target altitude from trajectory
            target_alt_from_traj = guidance_trajectory[closest_idx][1]
            altitude_error = target_alt_from_traj - measured_altitude

            pitch_command_deg = self.altitude_controller.update(altitude_error, dt)
            self.commanded_pitch = pitch_command_deg

        # ========== MODE 3: TERMINAL GUIDANCE ==========
        elif guidance_mode == "terminal" and target_position is not None:
            # Chase the target directly
            target_x, target_y = target_position
            distance_to_target = math.sqrt(
                (vehicle_state.position_x - target_x) ** 2 +
                (vehicle_state.position_y - target_y) ** 2
            )

            # More aggressive control near target
            altitude_error = target_y - measured_altitude

            # Use stronger gains for terminal guidance
            temp_kp = self.altitude_controller.kp
            self.altitude_controller.kp *= 1.5  # More aggressive

            pitch_command_deg = self.altitude_controller.update(altitude_error, dt)

            self.altitude_controller.kp = temp_kp  # Restore
            self.commanded_pitch = pitch_command_deg

        # ========== PITCH CONTROL LOOP ==========
        current_pitch_deg = math.degrees(vehicle_state.pitch_angle)
        pitch_error_deg = pitch_command_deg - current_pitch_deg
        surface_command_deg = self.pitch_controller.update(pitch_error_deg, dt)

        # ========== SURFACE ACTUATION ==========
        surface_command_rad = math.radians(surface_command_deg)

        for i, actuator in enumerate(self.actuators):
            actuator.set_command(surface_command_rad)
            self.commanded_surface_angles[i] = surface_command_rad

        return True

    def step_actuators(self, dt):
        """
        Step actuators (separate from control law for timing flexibility).

        Args:
            dt: timestep in seconds
        """
        for actuator in self.actuators:
            actuator.step(dt)

    def get_surface_angles(self):
        """
        Get current actual surface angles.

        Returns:
            angles: list of surface angles in radians
        """
        angles = [actuator.get_angle() for actuator in self.actuators]
        return angles

    def reset(self):
        """Reset controller to initial state."""
        self.altitude_controller.reset()
        self.pitch_controller.reset()
        self.is_enabled = False

        for actuator in self.actuators:
            actuator.current_angle_rad = 0.0
            actuator.angle_rate = 0.0