"""
Main simulation runner.

Orchestrates the complete simulation including:
- Scenario setup
- Physics integration
- Control system execution
- Data recording
- Visualization

FIXES:
- Guidance trajectory is now a direct line from launch to target,
  NOT the ballistic parabola (which is already the wrong path)
- Wind is passed to dynamics each step so aerodynamics use airspeed
- pitch_command stored directly in degrees (no erroneous math.degrees() call)
- run_simulation() passes target_position and guidance_mode to controller
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import config

from vehicle import Vehicle
from dynamics import VehicleDynamics
from atmosphere import get_wind_vector
from wind import WindModel
from imu import IMU
from encoder import SurfaceAngleEncoder
from controller import VehicleController
from plots import (
    create_trajectory_comparison_plot,
    create_altitude_error_plot,
    create_surface_angle_plot,
    create_controller_output_plot,
    create_velocity_plot,
    create_dashboard,
    display_summary_statistics
)
from animation import create_simple_animation


class SimulationRecorder:
    """Records simulation data for analysis."""

    def __init__(self):
        """Initialize data storage."""
        self.data = {
            'time': [],
            'x': [],
            'y': [],
            'vx': [],
            'vy': [],
            'pitch': [],
            'pitch_rate': [],
            'surface_commanded': [],
            'surface_actual': [],
            'altitude_error': [],
            'pitch_command': [],
            'wind_x': [],
            'wind_y': [],
            # Fixed-canard phase guidance telemetry
            'required_correction_deg': [],
            'required_correction_fraction': [],
            'required_canard_clock_angle_deg': [],
            'current_canard_clock_angle_deg': [],
            'flight_path_error_deg': [],
            'pitch_error_deg': [],
        }

    def record(self, time_s, vehicle, controller=None, wind_x=0.0, wind_y=0.0):
        """
        Record current simulation state.

        Args:
            time_s: current time in seconds
            vehicle: Vehicle object
            controller: VehicleController object (optional)
            wind_x, wind_y: wind velocity components
        """
        self.data['time'].append(time_s)
        self.data['x'].append(vehicle.position_x)
        self.data['y'].append(vehicle.position_y)
        self.data['vx'].append(vehicle.velocity_x)
        self.data['vy'].append(vehicle.velocity_y)
        self.data['pitch'].append(vehicle.pitch_angle)
        self.data['pitch_rate'].append(vehicle.pitch_rate)
        self.data['wind_x'].append(wind_x)
        self.data['wind_y'].append(wind_y)

        if controller is not None and controller.is_enabled:
            if controller.actuators:
                self.data['surface_commanded'].append(
                    controller.commanded_surface_angles[0]
                )
                self.data['surface_actual'].append(controller.actuators[0].get_angle())
            else:
                self.data['surface_commanded'].append(0.0)
                self.data['surface_actual'].append(0.0)

            self.data['altitude_error'].append(
                vehicle.position_y - config.target_position_y
            )
            self.data['pitch_command'].append(controller.commanded_pitch)

            # Fixed-canard phase guidance telemetry
            self.data['required_correction_deg'].append(
                controller.required_correction_deg
            )
            self.data['required_correction_fraction'].append(
                controller.required_correction_fraction
            )
            self.data['required_canard_clock_angle_deg'].append(
                controller.required_canard_clock_angle_deg
            )
            self.data['current_canard_clock_angle_deg'].append(
                controller.current_canard_clock_angle_deg
            )
            self.data['flight_path_error_deg'].append(
                controller.flight_path_error_deg
            )
            self.data['pitch_error_deg'].append(
                controller.pitch_error_deg
            )
        else:
            self.data['surface_commanded'].append(0.0)
            self.data['surface_actual'].append(0.0)
            self.data['altitude_error'].append(
                vehicle.position_y - config.target_position_y
            )
            self.data['pitch_command'].append(0.0)

            # Fixed-canard telemetry: neutral/zero when controller inactive
            self.data['required_correction_deg'].append(0.0)
            self.data['required_correction_fraction'].append(0.0)
            self.data['required_canard_clock_angle_deg'].append(90.0)
            self.data['current_canard_clock_angle_deg'].append(90.0)
            self.data['flight_path_error_deg'].append(0.0)
            self.data['pitch_error_deg'].append(0.0)


def make_direct_guidance_trajectory(launch_x, launch_y, target_x, target_y, num_points=100):
    """
    Build a straight-line waypoint list from launch to target.

    This is the correct reference path for the guided projectile.
    Using the ballistic parabola as guidance is wrong because:
    - The ballistic solution ignores drag (so the parabola is already wrong)
    - Telling the guided shot to follow the wrong path makes CEP worse

    Args:
        launch_x, launch_y: launch position (m)
        target_x, target_y: target position (m)
        num_points: number of waypoints

    Returns:
        list of (x, y) tuples from launch to target
    """
    xs = np.linspace(launch_x, target_x, num_points)
    ys = np.linspace(launch_y, target_y, num_points)
    return list(zip(xs, ys))


def run_simulation_guided(ballistic_solution, guidance_trajectory):
    """
    Run guided simulation with trajectory following.

    Args:
        ballistic_solution: dict with ballistic launch parameters
        guidance_trajectory: list of (x, y) points to follow (direct line to target)

    Returns:
        recorder: SimulationRecorder with simulation data
    """
    print(f"\n{'=' * 70}")
    print(f"Running: GUIDED WITH TRAJECTORY FOLLOWING")
    print(f"{'=' * 70}")

    vehicle = Vehicle()
    dynamics = VehicleDynamics(vehicle)
    controller = VehicleController()
    wind_model = WindModel()
    recorder = SimulationRecorder()

    # Launch with ballistic solution angles
    vehicle.position_x = 0.0
    vehicle.position_y = config.ballistic_launch_altitude
    vehicle.velocity_x = ballistic_solution['launch_velocity'] * math.cos(
        ballistic_solution['launch_angle_rad']
    )
    vehicle.velocity_y = ballistic_solution['launch_velocity'] * math.sin(
        ballistic_solution['launch_angle_rad']
    )
    vehicle.pitch_angle = ballistic_solution['launch_angle_rad']

    controller.enable()

    simulation_time = config.simulation_time
    physics_dt = config.time_step
    sensor_update_dt = config.sensor_update_period
    controller_update_dt = config.controller_update_period

    next_sensor_update = sensor_update_dt
    next_controller_update = controller_update_dt
    current_time = 0.0
    step_count = 0

    print(f"Guidance trajectory: {len(guidance_trajectory)} waypoints (direct line to target)")
    print(f"Target: ({config.target_position_x}, {config.target_position_y})")

    wind_x, wind_y = 0.0, 0.0

    while current_time < simulation_time:
        if vehicle.position_y < 0:
            print(f"  Vehicle hit ground at {current_time:.2f}s")
            break
        if vehicle.position_x >= config.target_position_x:
            print(f"  Reached target x-range at {current_time:.2f}s")
            break

        # Update wind
        wind_model.update(current_time)
        wind_x, wind_y = wind_model.get_wind(vehicle.position_y)

        # Tell dynamics what the wind is (so airspeed is used in force calcs)
        dynamics.set_wind(wind_x, wind_y)

        # Sensor + controller update
        if current_time >= next_sensor_update:
            measured_altitude = vehicle.position_y
            measured_surface_angles = controller.get_surface_angles()

            if current_time >= next_controller_update:
                # Switch to terminal guidance in the last stretch
                dist_to_target = math.sqrt(
                    (vehicle.position_x - config.target_position_x) ** 2 +
                    (vehicle.position_y - config.target_position_y) ** 2
                )

                if (config.terminal_guidance_enable and
                        dist_to_target < config.terminal_guidance_distance):
                    mode = "terminal"
                else:
                    mode = "trajectory"

                controller.update(
                    vehicle,
                    measured_altitude,
                    measured_surface_angles,
                    controller_update_dt,
                    target_position=(config.target_position_x, config.target_position_y),
                    guidance_trajectory=guidance_trajectory,
                    guidance_mode=mode
                )
                next_controller_update += controller_update_dt

            next_sensor_update += sensor_update_dt

        controller.step_actuators(physics_dt)
        surface_angles = controller.get_surface_angles()
        dynamics.step(physics_dt, surface_angles)

        if step_count % max(1, int(controller_update_dt / physics_dt)) == 0:
            recorder.record(current_time, vehicle, controller, wind_x, wind_y)

        current_time += physics_dt
        step_count += 1

        if step_count % 1000 == 0:
            distance_to_target = math.sqrt(
                (vehicle.position_x - config.target_position_x) ** 2 +
                (vehicle.position_y - config.target_position_y) ** 2
            )
            print(f"  {current_time:.1f}s: Pos=({vehicle.position_x:.0f}, "
                  f"{vehicle.position_y:.0f}) | Distance to target: {distance_to_target:.0f}m")

    recorder.record(current_time, vehicle, controller, wind_x, wind_y)

    final_x_error = abs(vehicle.position_x - config.target_position_x)
    final_y_error = abs(vehicle.position_y - config.target_position_y)
    miss_distance = math.sqrt(final_x_error ** 2 + final_y_error ** 2)

    print(f"\nSimulation complete: {current_time:.2f} s")
    print(f"  Final position: ({vehicle.position_x:.2f}, {vehicle.position_y:.2f}) m")
    print(f"  Target position: ({config.target_position_x:.2f}, {config.target_position_y:.2f}) m")
    print(f"  Horizontal error: {final_x_error:.2f} m")
    print(f"  Vertical error: {final_y_error:.2f} m")
    print(f"  TOTAL MISS DISTANCE: {miss_distance:.2f} m")

    recorder.target_x = config.target_position_x
    recorder.target_y = config.target_position_y
    recorder.miss_distance = miss_distance
    recorder.ballistic_solution = ballistic_solution

    return recorder


def run_simulation(enable_control=False, ballistic_solution=None, is_ballistic_demo=False):
    """
    Run a complete simulation scenario.

    Args:
        enable_control: if True, control system is active
        ballistic_solution: dict with ballistic trajectory data
        is_ballistic_demo: if True, this is the unguided ballistic shot

    Returns:
        recorder: SimulationRecorder with all simulation data
    """
    scenario_name = "BALLISTIC UNGUIDED" if is_ballistic_demo else (
        "BALLISTIC + GUIDED CONTROL" if enable_control and ballistic_solution else
        "NO CONTROL"
    )

    print(f"\n{'=' * 70}")
    print(f"Running: {scenario_name}")
    print(f"{'=' * 70}")

    vehicle = Vehicle()
    dynamics = VehicleDynamics(vehicle)
    controller = VehicleController()
    wind_model = WindModel()
    imu = IMU()
    encoders = [SurfaceAngleEncoder() for _ in range(config.num_control_surfaces)]
    recorder = SimulationRecorder()

    if ballistic_solution:
        vehicle.position_x = 0.0
        vehicle.position_y = config.ballistic_launch_altitude
        vehicle.velocity_x = ballistic_solution['launch_velocity'] * math.cos(
            ballistic_solution['launch_angle_rad']
        )
        vehicle.velocity_y = ballistic_solution['launch_velocity'] * math.sin(
            ballistic_solution['launch_angle_rad']
        )
        vehicle.pitch_angle = ballistic_solution['launch_angle_rad']

        print(f"\nBallistic Solution:")
        print(f"  Launch velocity: {ballistic_solution['launch_velocity']:.2f} m/s")
        print(f"  Launch angle: {ballistic_solution['launch_angle_deg']:.2f}°")
        print(f"  Predicted range: {ballistic_solution['predicted_range']:.2f} m")
        print(f"  Predicted flight time: {ballistic_solution['time_to_target']:.2f} s")

    if enable_control:
        controller.enable()

    simulation_time = config.simulation_time
    physics_dt = config.time_step
    sensor_update_dt = config.sensor_update_period
    controller_update_dt = config.controller_update_period

    next_sensor_update = sensor_update_dt
    next_controller_update = controller_update_dt
    current_time = 0.0
    step_count = 0

    wind_x, wind_y = 0.0, 0.0

    while current_time < simulation_time:
        if vehicle.position_y < 0:
            print(f"  Vehicle hit ground at {current_time:.2f}s")
            break
        if vehicle.position_x >= config.target_position_x:
            print(f"  Reached target x-range at {current_time:.2f}s")
            break

        wind_model.update(current_time)
        wind_x, wind_y = wind_model.get_wind(vehicle.position_y)

        # FIX: tell dynamics about current wind every step
        dynamics.set_wind(wind_x, wind_y)

        if current_time >= next_sensor_update:
            measured_altitude = vehicle.position_y
            measured_surface_angles = [
                encoders[i].measure(controller.get_surface_angles()[i])
                for i in range(config.num_control_surfaces)
            ] if enable_control else [0.0] * config.num_control_surfaces

            if current_time >= next_controller_update and enable_control:
                # Ballistic demo with control: hold altitude only
                controller.update(
                    vehicle,
                    measured_altitude,
                    measured_surface_angles,
                    controller_update_dt,
                    guidance_mode="altitude"
                )
                next_controller_update += controller_update_dt

            next_sensor_update += sensor_update_dt

        if enable_control:
            controller.step_actuators(physics_dt)
            surface_angles = controller.get_surface_angles()
        else:
            surface_angles = [0.0] * config.num_control_surfaces

        dynamics.step(physics_dt, surface_angles)

        if step_count % max(1, int(controller_update_dt / physics_dt)) == 0:
            recorder.record(current_time, vehicle, controller if enable_control else None,
                            wind_x, wind_y)

        current_time += physics_dt
        step_count += 1

        if step_count % 1000 == 0:
            distance_to_target = abs(vehicle.position_x - config.target_position_x)
            alt_error = abs(vehicle.position_y - config.target_position_y)
            print(f"  {current_time:.1f}s: Pos=({vehicle.position_x:.0f}, "
                  f"{vehicle.position_y:.0f}) | Distance to target: {distance_to_target:.0f}m | "
                  f"Alt error: {alt_error:.0f}m")

    recorder.record(current_time, vehicle, controller if enable_control else None, wind_x, wind_y)

    final_x_error = abs(vehicle.position_x - config.target_position_x)
    final_y_error = abs(vehicle.position_y - config.target_position_y)
    miss_distance = math.sqrt(final_x_error ** 2 + final_y_error ** 2)

    print(f"\nSimulation complete: {current_time:.2f} s")
    print(f"  Final position: ({vehicle.position_x:.2f}, {vehicle.position_y:.2f}) m")
    print(f"  Target position: ({config.target_position_x:.2f}, {config.target_position_y:.2f}) m")
    print(f"  Horizontal error: {final_x_error:.2f} m")
    print(f"  Vertical error: {final_y_error:.2f} m")
    print(f"  TOTAL MISS DISTANCE: {miss_distance:.2f} m")

    recorder.target_x = config.target_position_x
    recorder.target_y = config.target_position_y
    recorder.miss_distance = miss_distance
    recorder.ballistic_solution = ballistic_solution

    return recorder


def main():
    """Main entry point for the simulation."""
    print("\n" + "=" * 70)
    print("AERODYNAMIC VEHICLE SIMULATION - BALLISTIC vs GUIDED")
    print("=" * 70)

    results = {}
    ballistic_solution = None

    if config.ballistic_mode_enabled:
        print("\n" + "=" * 70)
        print("CALCULATING BALLISTIC SOLUTION")
        print("=" * 70)

        from projectile_motion import calculate_ballistic_solution, calculate_ballistic_trajectory

        ballistic_solution = calculate_ballistic_solution(
            target_x=config.target_position_x,
            target_y=config.target_position_y,
            launch_x=0.0,
            launch_y=config.ballistic_launch_altitude,
            initial_velocity_magnitude=config.ballistic_launch_velocity
        )

        if ballistic_solution['success']:
            print(f"\n✓ Ballistic solution FOUND:")
            print(f"  Launch angle: {ballistic_solution['launch_angle_deg']:.2f}°")
            print(f"  Launch velocity: {ballistic_solution['launch_velocity']:.2f} m/s")
            print(f"  Predicted flight time: {ballistic_solution['time_to_target']:.2f} s")
            print(f"\n  ⚠️  WARNING: This assumes NO AERODYNAMIC DRAG")
            print(f"      In reality, drag will cause the projectile to MISS!")

    if config.run_scenario_ballistic and ballistic_solution and ballistic_solution['success']:
        print("\n[1/3] Running BALLISTIC (unguided) scenario...")
        results['ballistic'] = run_simulation(
            enable_control=False,
            ballistic_solution=ballistic_solution,
            is_ballistic_demo=True
        )

    if config.run_scenario_guided and ballistic_solution['success']:
        print("\n[2/3] Running GUIDED (controlled) scenario...")

        from projectile_motion import calculate_ballistic_trajectory
        bx, by = calculate_ballistic_trajectory(
            launch_x=0.0,
            launch_y=config.ballistic_launch_altitude,
            launch_velocity=ballistic_solution['launch_velocity'],
            launch_angle_rad=ballistic_solution['launch_angle_rad'],
            num_points=200
        )

        # Clip guidance points at target so controller does not track downward past the target
        guidance_trajectory = [
            (x, y) for x, y in zip(bx, by) if x <= config.target_position_x
        ]
        guidance_trajectory.append((config.target_position_x, config.target_position_y))

        results['guided'] = run_simulation_guided(
            ballistic_solution=ballistic_solution,
            guidance_trajectory=guidance_trajectory
        )

    if 'ballistic' in results and 'guided' in results:
        print("\n[3/3] Creating visualizations...")

        ballistic_traj = None
        if ballistic_solution and ballistic_solution['success']:
            ballistic_traj = calculate_ballistic_trajectory(
                launch_x=0.0,
                launch_y=config.ballistic_launch_altitude,
                launch_velocity=ballistic_solution['launch_velocity'],
                launch_angle_rad=ballistic_solution['launch_angle_rad'],
                num_points=100
            )

        from plots import (
            create_ballistic_vs_guided_plot,
            create_cep_heatmap,
            create_pid_telemetry_dashboard,
            create_canard_phase_telemetry_dashboard,
        )

        fig_comparison = create_ballistic_vs_guided_plot(
            results['ballistic'].data,
            results['guided'].data,
            ballistic_trajectory=ballistic_traj
        )

        fig_cep = create_cep_heatmap(
            results['ballistic'].data,
            results['guided'].data
        )

        # Figure 1: Guided Projectile Telemetry (existing, relabelled)
        fig_telemetry = create_pid_telemetry_dashboard(results['guided'].data)

        # Figure 2: Fixed-Canard Phase Guidance Telemetry (new)
        fig_canard = create_canard_phase_telemetry_dashboard(results['guided'].data)

        ballistic_miss = results['ballistic'].miss_distance
        guided_miss = results['guided'].miss_distance
        improvement = ballistic_miss - guided_miss
        improvement_percent = (improvement / ballistic_miss * 100) if ballistic_miss > 0 else 0

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"\nTARGET LOCATION: ({config.target_position_x:.0f}m, {config.target_position_y:.0f}m)")

        print(f"\nBALLISTIC SHOT (Naive Projectile Math):")
        print(f"  Final position: ({results['ballistic'].data['x'][-1]:.2f}, {results['ballistic'].data['y'][-1]:.2f})")
        print(f"  Miss distance: {ballistic_miss:.2f} m ❌")

        print(f"\nGUIDED SHOT (With Aerodynamic Control & Trajectory Guidance):")
        print(f"  Final position: ({results['guided'].data['x'][-1]:.2f}, {results['guided'].data['y'][-1]:.2f})")
        print(f"  Miss distance: {guided_miss:.2f} m {'✓' if guided_miss < ballistic_miss else '⚠'}")

        print(f"\nIMPROVEMENT:")
        print(f"  Accuracy gain: {improvement:.2f} m ({improvement_percent:.1f}%)")

        if config.plot_show:
            import matplotlib.pyplot as plt
            plt.show()



if __name__ == "__main__":
    main()