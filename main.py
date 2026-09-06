"""
Main simulation runner.

Orchestrates the complete simulation including:
- Scenario setup
- Physics integration
- Control system execution
- Data recording
- Visualization
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
        }

    def record(self, time_s, vehicle, controller=None, wind_x=0.0, wind_y=0.0):
        """
        Record current simulation state.

        Args:
            time_s: current time in seconds
            vehicle: Vehicle object
            controller: VehicleController object (optional)
            wind_x, wind_y: wind velocity
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

        # Surface angles (if controller exists)
        if controller is not None:
            if controller.actuators:
                self.data['surface_commanded'].append(controller.commanded_surface_angles[0])
                self.data['surface_actual'].append(controller.actuators[0].get_angle())
            else:
                self.data['surface_commanded'].append(0.0)
                self.data['surface_actual'].append(0.0)

            # Error tracking
            altitude_error = vehicle.position_y - config.reference_altitude
            self.data['altitude_error'].append(altitude_error)
            self.data['pitch_command'].append(math.degrees(controller.commanded_pitch))
        else:
            self.data['surface_commanded'].append(0.0)
            self.data['surface_actual'].append(0.0)
            self.data['altitude_error'].append(0.0)
            self.data['pitch_command'].append(0.0)


def run_simulation_guided(ballistic_solution, guidance_trajectory):
    """
    Run guided simulation with trajectory following.

    Args:
        ballistic_solution: dict with ballistic launch parameters
        guidance_trajectory: list of (x, y) points to follow

    Returns:
        recorder: SimulationRecorder with simulation data
    """
    print(f"\n{'=' * 70}")
    print(f"Running: GUIDED WITH TRAJECTORY FOLLOWING")
    print(f"{'=' * 70}")

    # Initialize
    vehicle = Vehicle()
    dynamics = VehicleDynamics(vehicle)
    controller = VehicleController()
    wind_model = WindModel()
    recorder = SimulationRecorder()

    # Launch with ballistic solution
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

    # Timing
    simulation_time = config.simulation_time
    physics_dt = config.time_step
    sensor_update_dt = config.sensor_update_period
    controller_update_dt = config.controller_update_period

    next_sensor_update = sensor_update_dt
    next_controller_update = controller_update_dt
    current_time = 0.0
    step_count = 0

    print(f"Guidance trajectory: {len(guidance_trajectory)} points")
    print(f"Target: ({config.target_position_x}, {config.target_position_y})")

    # Main loop
    while current_time < simulation_time:
        if vehicle.position_y < 0:
            break
        if vehicle.position_x > config.target_position_x + 500:
            break

        wind_model.update(current_time)
        wind_x, wind_y = wind_model.get_wind(vehicle.position_y)

        if current_time >= next_sensor_update:
            measured_altitude = vehicle.position_y
            measured_surface_angles = [vehicle.pitch_angle]

            if current_time >= next_controller_update:
                # Use trajectory guidance mode
                controller.update(
                    vehicle,
                    measured_altitude,
                    measured_surface_angles,
                    controller_update_dt,
                    target_position=(config.target_position_x, config.target_position_y),
                    guidance_trajectory=guidance_trajectory,
                    guidance_mode="trajectory"  # Follow guidance trajectory
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
            distance_to_target = abs(vehicle.position_x - config.target_position_x)
            alt_error = abs(vehicle.position_y - config.target_position_y)
            print(f"  {current_time:.1f}s: Pos=({vehicle.position_x:.0f}, "
                  f"{vehicle.position_y:.0f}) | Distance to target: {distance_to_target:.0f}m | "
                  f"Alt error: {alt_error:.0f}m")

    recorder.record(current_time, vehicle, controller, wind_x, wind_y)

    # Calculate miss
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
        ballistic_solution: dict with ballistic trajectory data (for guided mode)
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

    # Initialize components
    vehicle = Vehicle()
    dynamics = VehicleDynamics(vehicle)
    controller = VehicleController()
    wind_model = WindModel()
    imu = IMU()
    encoders = [SurfaceAngleEncoder() for _ in range(config.num_control_surfaces)]
    recorder = SimulationRecorder()

    # Set initial conditions
    if ballistic_solution:
        # Launch with ballistic solution
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

    # Enable control if requested
    if enable_control:
        controller.enable()

    # Simulation timing
    simulation_time = config.simulation_time
    physics_dt = config.time_step
    sensor_update_dt = config.sensor_update_period
    controller_update_dt = config.controller_update_period

    # Timing counters
    next_sensor_update = sensor_update_dt
    next_controller_update = controller_update_dt

    current_time = 0.0
    step_count = 0

    # Main simulation loop
    while current_time < simulation_time:
        # Stop if vehicle hits ground
        if vehicle.position_y < 0:
            print(f"  Vehicle hit ground at {current_time:.2f}s")
            break

        # Stop if vehicle is far beyond target (won't come back)
        if vehicle.position_x > config.target_position_x + 500:
            print(f"  Vehicle passed target zone at {current_time:.2f}s")
            break

        # Update wind model
        wind_model.update(current_time)
        wind_x, wind_y = wind_model.get_wind(vehicle.position_y)

        # Update sensors if it's time
        if current_time >= next_sensor_update:
            measured_altitude = vehicle.position_y
            measured_surface_angles = [vehicle.pitch_angle]

            # Update controller if it's time
            if current_time >= next_controller_update and enable_control:
                controller.update(vehicle, measured_altitude, measured_surface_angles,
                                  controller_update_dt)
                next_controller_update += controller_update_dt

            next_sensor_update += sensor_update_dt

        # Step actuators
        if enable_control:
            controller.step_actuators(physics_dt)
            surface_angles = controller.get_surface_angles()
        else:
            surface_angles = [0.0] * config.num_control_surfaces

        # Physics integration
        dynamics.step(physics_dt, surface_angles)

        # Record data
        if step_count % max(1, int(controller_update_dt / physics_dt)) == 0:
            recorder.record(current_time, vehicle, controller, wind_x, wind_y)

        # Advance time
        current_time += physics_dt
        step_count += 1

        # Progress indicator
        if step_count % 1000 == 0:
            distance_to_target = abs(vehicle.position_x - config.target_position_x)
            alt_error = abs(vehicle.position_y - config.target_position_y)
            print(f"  {current_time:.1f}s: Pos=({vehicle.position_x:.0f}, "
                  f"{vehicle.position_y:.0f}) | Distance to target: {distance_to_target:.0f}m | "
                  f"Alt error: {alt_error:.0f}m")

    # Final recording
    recorder.record(current_time, vehicle, controller, wind_x, wind_y)

    # Calculate miss distance
    final_x_error = abs(vehicle.position_x - config.target_position_x)
    final_y_error = abs(vehicle.position_y - config.target_position_y)
    miss_distance = math.sqrt(final_x_error ** 2 + final_y_error ** 2)

    print(f"\nSimulation complete: {current_time:.2f} s")
    print(f"  Final position: ({vehicle.position_x:.2f}, {vehicle.position_y:.2f}) m")
    print(f"  Target position: ({config.target_position_x:.2f}, {config.target_position_y:.2f}) m")
    print(f"  Horizontal error: {final_x_error:.2f} m")
    print(f"  Vertical error: {final_y_error:.2f} m")
    print(f"  TOTAL MISS DISTANCE: {miss_distance:.2f} m")

    # Store target and ballistic info for visualization
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
        print("      (This will MISS the target due to aerodynamic drag)")
        results['ballistic'] = run_simulation(
            enable_control=False,
            ballistic_solution=ballistic_solution,
            is_ballistic_demo=True
        )

    if config.run_scenario_guided and ballistic_solution and ballistic_solution['success']:
        print("\n[2/3] Running GUIDED (controlled) scenario...")
        print("      (Control system corrects trajectory to HIT target)")

        # Calculate guidance trajectory from ballistic solution to target
        from projectile_motion import calculate_ballistic_trajectory

        guidance_traj = calculate_ballistic_trajectory(
            launch_x=0.0,
            launch_y=config.ballistic_launch_altitude,
            launch_velocity=ballistic_solution['launch_velocity'],
            launch_angle_rad=ballistic_solution['launch_angle_rad'],
            num_points=50
        )

        # Convert to list of (x, y) tuples
        guidance_trajectory = list(zip(guidance_traj[0], guidance_traj[1]))

        # Blend: follow ballistic trajectory, then descend to target
        # For last part of trajectory, guide to target instead of ballistic path
        blend_start_idx = int(len(guidance_trajectory) * 0.7)
        target_x = config.target_position_x
        target_y = config.target_position_y

        for i in range(blend_start_idx, len(guidance_trajectory)):
            # Linear interpolation from ballistic to target
            blend_factor = (i - blend_start_idx) / (len(guidance_trajectory) - blend_start_idx)
            ballistic_x, ballistic_y = guidance_trajectory[i]
            blended_x = ballistic_x + (target_x - ballistic_x) * blend_factor
            blended_y = ballistic_y + (target_y - ballistic_y) * blend_factor
            guidance_trajectory[i] = (blended_x, blended_y)

        results['guided'] = run_simulation_guided(
            ballistic_solution=ballistic_solution,
            guidance_trajectory=guidance_trajectory
        )

    # Create visualizations
    if 'ballistic' in results and 'guided' in results:
        print("\n[3/3] Creating visualizations...")

        ballistic_traj = None
        if ballistic_solution['success']:
            from projectile_motion import calculate_ballistic_trajectory
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

        # Print summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        ballistic_miss = results['ballistic'].miss_distance
        guided_miss = results['guided'].miss_distance
        improvement = ballistic_miss - guided_miss
        improvement_percent = (improvement / ballistic_miss * 100) if ballistic_miss > 0 else 0

        print(f"\nTARGET LOCATION: ({config.target_position_x:.0f}m, {config.target_position_y:.0f}m)")
        print(f"\nBALLISTIC SHOT (Naive Projectile Math):")
        print(f"  Final position: ({results['ballistic'].data['x'][-1]:.2f}, "
              f"{results['ballistic'].data['y'][-1]:.2f})")
        print(f"  Miss distance: {ballistic_miss:.2f} m ❌")

        print(f"\nGUIDED SHOT (With Aerodynamic Control & Trajectory Guidance):")
        print(f"  Final position: ({results['guided'].data['x'][-1]:.2f}, "
              f"{results['guided'].data['y'][-1]:.2f})")
        print(f"  Miss distance: {guided_miss:.2f} m ✓")

        print(f"\nIMPROVEMENT:")
        print(f"  Accuracy gain: {improvement:.2f} m ({improvement_percent:.1f}%)")

        print("\n" + "=" * 70)
        print("KEY INSIGHT:")
        print("=" * 70)
        print("Ballistic predictions are wrong because they ignore aerodynamic drag.")
        print("The guided system:")
        print("  1. Follows the ballistic trajectory initially")
        print("  2. Gradually blends to direct target approach")
        print("  3. Uses control surfaces to correct trajectory in real-time")
        print("  4. Achieves precision through active guidance")
        print("=" * 70 + "\n")

        if config.plot_show:
            plt.show()


if __name__ == "__main__":
    main()