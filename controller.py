"""
Intermediate Fixed-Canard Guidance Compatibility Controller.

This controller preserves the existing and validated guidance architecture:

    Trajectory error
            ↓
    Flight-path-angle error
            ↓
    Desired pitch correction
            ↓
    Inner PD correction
            ↓
    Equivalent aerodynamic correction demand

The old simulation interpreted this demand as a movable control-surface
deflection.

For the fixed-canard demonstration, the same demand is additionally
translated into the required clock orientation of a fixed canard assembly.

IMPORTANT SIMPLIFICATION
------------------------

The existing actuator and aerodynamic pipeline is intentionally preserved
so that the previously validated simulation behaviour, plots, Monte Carlo
results, and interfaces remain unchanged.

The old surface command is therefore treated as an EQUIVALENT AERODYNAMIC
CORRECTION COMMAND.

A virtual fixed-canard clock angle is calculated using:

    correction_fraction = correction / maximum_correction

    correction_fraction = cos(clock_angle)

Therefore:

    +1.0 correction ->   0 degrees
     0.0 correction ->  90 degrees
    -1.0 correction -> 180 degrees

The physical interpretation for the hackathon model is:

    "If a fixed canard assembly were rapidly phased to this clock angle,
     its projected aerodynamic authority in the pitch plane would produce
     the required fraction of the maximum correction."

Rapid phase alignment is assumed instantaneous relative to the simulation
time step.

The existing actuator output is retained only as an equivalent aerodynamic
correction variable, allowing the rest of the existing simulation code to
remain unchanged.
"""

import math
import config
from actuator import ControlSurfaceActuator


class VehicleController:
    """
    Guidance controller with an equivalent fixed-canard orientation model.

    External API is intentionally identical to the original controller so
    that main.py, dynamics.py, plotting, and Monte Carlo code do not need
    to change.
    """

    def __init__(self, num_surfaces=config.num_control_surfaces):

        # Retained for backwards compatibility with the existing simulation.
        self.num_surfaces = num_surfaces

        self.actuators = [
            ControlSurfaceActuator(i)
            for i in range(num_surfaces)
        ]

        self.is_enabled = False

        # Existing telemetry variables.
        self.commanded_pitch = 0.0

        self.commanded_surface_angles = [
            0.0
        ] * num_surfaces

        # ============================================================
        # NEW: FIXED-CANARD EQUIVALENT GUIDANCE VARIABLES
        # ============================================================

        # Equivalent aerodynamic correction calculated by the old PD loop.
        # Units: degrees.
        self.required_correction_deg = 0.0

        # Normalised correction from -1 to +1.
        self.required_correction_fraction = 0.0

        # Required orientation of the fixed canard assembly.
        # Radians, range 0 to pi for this 2D pitch-plane model.
        self.required_canard_clock_angle = math.pi / 2.0

        # Degrees version for telemetry/debugging.
        self.required_canard_clock_angle_deg = 90.0

        # Current assumed canard orientation.
        #
        # Under the rapid-phase-alignment assumption this immediately
        # becomes equal to the required angle.
        self.current_canard_clock_angle = math.pi / 2.0

        self.current_canard_clock_angle_deg = 90.0

        # Store useful guidance errors for debugging/presentation.
        self.flight_path_error_deg = 0.0
        self.pitch_error_deg = 0.0

    def enable(self):
        """Enable guidance controller."""
        self.is_enabled = True

    def disable(self):
        """Disable guidance controller."""
        self.is_enabled = False

    def update(
        self,
        vehicle_state,
        measured_altitude,
        measured_surface_angles,
        dt,
        target_position=None,
        guidance_trajectory=None,
        guidance_mode="trajectory"
    ):
        """
        Update guidance.

        The trajectory tracking and PD mathematics are intentionally kept
        the same as the original working implementation.

        The resulting aerodynamic correction demand is additionally mapped
        to a fixed-canard clock orientation.
        """

        if not self.is_enabled:
            return False

        # ============================================================
        # CURRENT VEHICLE STATE
        # ============================================================

        current_x = vehicle_state.position_x
        current_y = vehicle_state.position_y

        vx = vehicle_state.velocity_x
        vy = vehicle_state.velocity_y

        # ============================================================
        # STEP 1: CURRENT FLIGHT-PATH ANGLE
        # ============================================================

        current_fpa = math.atan2(
            vy,
            max(vx, 0.1)
        )

        # Default desired path is the current path.
        desired_fpa = current_fpa

        # ============================================================
        # STEP 2: TRAJECTORY FOLLOWING
        # ============================================================

        if (
            guidance_trajectory
            and len(guidance_trajectory) > 0
        ):

            look_ahead = 15.0

            target_wp = guidance_trajectory[-1]

            # Find waypoint ahead of the projectile.
            for wx, wy in guidance_trajectory:

                if wx >= current_x + look_ahead:

                    target_wp = (wx, wy)
                    break

            dx = target_wp[0] - current_x
            dy = target_wp[1] - current_y

            distance_to_waypoint = math.sqrt(
                dx ** 2 +
                dy ** 2
            )

            # Prevent unstable behaviour extremely close to waypoint.
            if distance_to_waypoint < 5.0:

                desired_fpa = current_fpa

            else:

                desired_fpa = math.atan2(
                    dy,
                    max(dx, 0.1)
                )

        # ============================================================
        # STEP 3: FLIGHT-PATH ERROR
        # ============================================================

        fpa_error_deg = math.degrees(
            desired_fpa - current_fpa
        )

        # Wrap error into [-180, +180].
        while fpa_error_deg > 180.0:
            fpa_error_deg -= 360.0

        while fpa_error_deg < -180.0:
            fpa_error_deg += 360.0

        self.flight_path_error_deg = (
            fpa_error_deg
        )

        # ============================================================
        # STEP 4: OUTER GUIDANCE LOOP
        #
        # EXACT SAME IDEA AS THE ORIGINAL WORKING CONTROLLER
        # ============================================================

        pitch_command_deg = (
            math.degrees(desired_fpa)
            +
            1.5 * fpa_error_deg
        )

        pitch_command_deg = max(
            -config.altitude_controller_output_limit,
            min(
                config.altitude_controller_output_limit,
                pitch_command_deg
            )
        )

        self.commanded_pitch = (
            pitch_command_deg
        )

        # ============================================================
        # STEP 5: PITCH ERROR
        # ============================================================

        current_pitch_deg = math.degrees(
            vehicle_state.pitch_angle
        )

        pitch_error_deg = (
            pitch_command_deg -
            current_pitch_deg
        )

        pitch_rate_deg = math.degrees(
            vehicle_state.pitch_rate
        )

        self.pitch_error_deg = (
            pitch_error_deg
        )

        # ============================================================
        # STEP 6: INNER PD CORRECTION
        #
        # THIS IS THE SAME CALCULATION THAT PREVIOUSLY GENERATED THE
        # CONTROL-SURFACE DEFLECTION.
        #
        # NOW WE INTERPRET IT AS THE REQUIRED EQUIVALENT AERODYNAMIC
        # CORRECTION.
        # ============================================================

        equivalent_correction_deg = (
            config.pitch_controller_kp
            *
            pitch_error_deg
            -
            config.pitch_controller_kd
            *
            pitch_rate_deg
        )

        equivalent_correction_deg = max(
            -config.pitch_controller_output_limit,
            min(
                config.pitch_controller_output_limit,
                equivalent_correction_deg
            )
        )

        # Store for telemetry.
        self.required_correction_deg = (
            equivalent_correction_deg
        )

        # ============================================================
        # STEP 7: CONVERT REQUIRED CORRECTION TO CANARD CLOCK ANGLE
        #
        # In this simplified 2D model:
        #
        # correction_fraction = cos(clock_angle)
        #
        # Therefore:
        #
        # clock_angle = acos(correction_fraction)
        #
        # 0°   -> maximum positive correction
        # 90°  -> zero pitch-plane correction
        # 180° -> maximum negative correction
        # ============================================================

        maximum_correction = max(
            abs(config.pitch_controller_output_limit),
            1e-9
        )

        correction_fraction = (
            equivalent_correction_deg /
            maximum_correction
        )

        correction_fraction = max(
            -1.0,
            min(
                1.0,
                correction_fraction
            )
        )

        self.required_correction_fraction = (
            correction_fraction
        )

        required_clock_angle = math.acos(
            correction_fraction
        )

        self.required_canard_clock_angle = (
            required_clock_angle
        )

        self.required_canard_clock_angle_deg = (
            math.degrees(
                required_clock_angle
            )
        )

        # ============================================================
        # STEP 8: RAPID PHASE ALIGNMENT ASSUMPTION
        #
        # Projectile rotational phase adjustment is assumed much faster
        # than the simulation/control time scale.
        #
        # Therefore orientation is treated as instantaneous.
        # ============================================================

        self.current_canard_clock_angle = (
            required_clock_angle
        )

        self.current_canard_clock_angle_deg = (
            self.required_canard_clock_angle_deg
        )

        # ============================================================
        # STEP 9: BACKWARDS-COMPATIBILITY ADAPTER
        #
        # We preserve the existing actuator pipeline so that:
        #
        # main.py
        # dynamics.py
        # forces.py
        # plots.py
        # Monte Carlo
        #
        # continue to run unchanged.
        #
        # The actuator command now represents the equivalent aerodynamic
        # correction associated with the required canard orientation.
        # ============================================================

        equivalent_command_rad = math.radians(
            equivalent_correction_deg
        )

        for i, actuator in enumerate(
            self.actuators
        ):

            actuator.set_command(
                equivalent_command_rad
            )

            self.commanded_surface_angles[i] = (
                equivalent_command_rad
            )

        return True

    def step_actuators(self, dt):
        """
        Retained unchanged for compatibility.

        In the intermediate model, this represents the response of the
        equivalent correction channel.
        """

        for actuator in self.actuators:

            actuator.step(dt)

    def get_surface_angles(self):
        """
        Existing API retained.

        These angles should now be interpreted as equivalent aerodynamic
        correction commands, not literal physical fin deflections.
        """

        return [
            actuator.get_angle()
            for actuator in self.actuators
        ]

    # ================================================================
    # NEW TELEMETRY / PRESENTATION METHODS
    # ================================================================

    def get_required_canard_clock_angle(self):
        """Return required fixed-canard clock angle in radians."""

        return (
            self.required_canard_clock_angle
        )

    def get_required_canard_clock_angle_deg(self):
        """Return required fixed-canard clock angle in degrees."""

        return (
            self.required_canard_clock_angle_deg
        )

    def get_current_canard_clock_angle_deg(self):
        """
        Return assumed current canard clock angle.

        Under rapid-phase-alignment assumption this equals the required
        orientation.
        """

        return (
            self.current_canard_clock_angle_deg
        )

    def get_required_correction_deg(self):
        """
        Return equivalent aerodynamic correction demand.

        This is the quantity previously interpreted as fin deflection.
        """

        return (
            self.required_correction_deg
        )

    def get_correction_fraction(self):
        """
        Return required correction as a fraction of maximum authority.

        Range: -1 to +1.
        """

        return (
            self.required_correction_fraction
        )

    def get_guidance_errors(self):
        """Return useful guidance diagnostics."""

        return {
            "flight_path_error_deg":
                self.flight_path_error_deg,

            "pitch_error_deg":
                self.pitch_error_deg,

            "required_correction_deg":
                self.required_correction_deg,

            "required_canard_clock_angle_deg":
                self.required_canard_clock_angle_deg,
        }

    def reset(self):
        """Reset controller."""

        self.is_enabled = False

        self.commanded_pitch = 0.0

        self.required_correction_deg = 0.0

        self.required_correction_fraction = 0.0

        self.required_canard_clock_angle = (
            math.pi / 2.0
        )

        self.required_canard_clock_angle_deg = (
            90.0
        )

        self.current_canard_clock_angle = (
            math.pi / 2.0
        )

        self.current_canard_clock_angle_deg = (
            90.0
        )

        self.flight_path_error_deg = 0.0
        self.pitch_error_deg = 0.0

        for actuator in self.actuators:

            actuator.current_angle_rad = 0.0

            actuator.commanded_angle_rad = 0.0

            actuator.angle_rate = 0.0