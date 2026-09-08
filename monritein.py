import math
import random
import numpy as np
import matplotlib.pyplot as plt

import config
from vehicle import Vehicle
from dynamics import VehicleDynamics
from controller import VehicleController
from projectile_motion import calculate_ballistic_solution, calculate_ballistic_trajectory
from wind import WindModel
from imu import IMU
from encoder import SurfaceAngleEncoder


def run_monte_carlo_validation(num_simulations=50):
    print(f"Running Monte Carlo Validation: {num_simulations} randomized scenarios...")

    results = []

    for i in range(num_simulations):
        # Randomize environmental and target variables
        test_target_x = random.uniform(300.0, 800.0)
        test_wind_x = random.uniform(-10.0, 10.0)  # Headwind/Tailwind
        test_wind_y = random.uniform(-5.0, 5.0)  # Updraft/Downdraft

        # Temporarily override config for this iteration
        config.target_position_x = test_target_x
        config.base_wind_speed_x = test_wind_x
        config.base_wind_speed_y = test_wind_y

        # Calculate ideal ballistic solution for the new target
        b_sol = calculate_ballistic_solution(
            target_x=test_target_x,
            target_y=config.target_position_y,
            launch_x=0.0,
            launch_y=config.ballistic_launch_altitude,
            initial_velocity_magnitude=config.ballistic_launch_velocity
        )

        if not b_sol['success']:
            continue

        # Generate the specific ballistic arc for the controller to track
        bx, by = calculate_ballistic_trajectory(
            launch_x=0.0,
            launch_y=config.ballistic_launch_altitude,
            launch_velocity=b_sol['launch_velocity'],
            launch_angle_rad=b_sol['launch_angle_rad'],
            num_points=150
        )
        guidance_trajectory = [(x, y) for x, y in zip(bx, by) if x <= test_target_x]
        guidance_trajectory.append((test_target_x, config.target_position_y))

        # Initialize Simulation Entities
        vehicle = Vehicle()
        vehicle.position_x = 0.0
        vehicle.position_y = config.ballistic_launch_altitude
        vehicle.velocity_x = b_sol['launch_velocity'] * math.cos(b_sol['launch_angle_rad'])
        vehicle.velocity_y = b_sol['launch_velocity'] * math.sin(b_sol['launch_angle_rad'])
        vehicle.pitch_angle = b_sol['launch_angle_rad']

        dynamics = VehicleDynamics(vehicle)
        controller = VehicleController()
        wind_model = WindModel()
        controller.enable()

        t = 0.0
        dt = config.time_step

        trajectory_x = []
        trajectory_y = []

        # Run Physics Loop
        while t < config.simulation_time:
            if vehicle.position_y < 0 or vehicle.position_x >= test_target_x:
                break

            wind_model.update(t)
            wind_x, wind_y = wind_model.get_wind(vehicle.position_y)
            dynamics.set_wind(wind_x, wind_y)

            # Controller Update
            controller.update(
                vehicle_state=vehicle,
                measured_altitude=vehicle.position_y,
                measured_surface_angles=controller.get_surface_angles(),
                dt=dt,
                target_position=(test_target_x, config.target_position_y),
                guidance_trajectory=guidance_trajectory,
                guidance_mode="trajectory"
            )

            controller.step_actuators(dt)
            dynamics.step(dt, controller.get_surface_angles())

            trajectory_x.append(vehicle.position_x)
            trajectory_y.append(vehicle.position_y)
            t += dt

        miss_distance = math.sqrt(
            (vehicle.position_x - test_target_x) ** 2 + (vehicle.position_y - config.target_position_y) ** 2)

        results.append({
            'target_x': test_target_x,
            'wind_x': test_wind_x,
            'wind_y': test_wind_y,
            'miss_distance': miss_distance,
            'traj_x': trajectory_x,
            'traj_y': trajectory_y
        })

        print(
            f"Run {i + 1:02d} | Target: {test_target_x:.0f}m | Wind: {test_wind_x:+.1f}m/s | Miss: {miss_distance:.2f}m")

    plot_monte_carlo_results(results)


def plot_monte_carlo_results(results):
    fig = plt.figure(figsize=(15, 10))

    # 1. Trajectory Overlay
    ax1 = plt.subplot(2, 1, 1)
    for res in results:
        ax1.plot(res['traj_x'], res['traj_y'], alpha=0.3, color='blue')
        ax1.plot(res['target_x'], config.target_position_y, 'g*', markersize=10)

    ax1.axhline(config.target_position_y, color='gray', linestyle='--')
    ax1.set_title("Monte Carlo Trajectory Density (Variable Targets & Wind)", fontweight='bold')
    ax1.set_xlabel("Horizontal Distance (m)")
    ax1.set_ylabel("Altitude (m)")
    ax1.grid(True, alpha=0.3)

    # 2. Miss Distance vs Wind Shear
    ax2 = plt.subplot(2, 2, 3)
    winds = [r['wind_x'] for r in results]
    misses = [r['miss_distance'] for r in results]
    ax2.scatter(winds, misses, c='red', alpha=0.6)
    ax2.set_title("Robustness: Miss Distance vs Wind Shear", fontweight='bold')
    ax2.set_xlabel("Crosswind X (m/s)")
    ax2.set_ylabel("Miss Distance (m)")
    ax2.grid(True, alpha=0.3)

    # 3. CEP Distribution
    ax3 = plt.subplot(2, 2, 4)
    ax3.hist(misses, bins=15, color='purple', alpha=0.7, edgecolor='black')
    ax3.set_title("Circular Error Probable (CEP) Distribution", fontweight='bold')
    ax3.set_xlabel("Miss Distance (m)")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_monte_carlo_validation(num_simulations=50)