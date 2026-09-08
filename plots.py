"""
Visualization and plotting of simulation results.

Creates matplotlib figures showing trajectory, errors, and control signals.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import config


def create_trajectory_comparison_plot(baseline_data, controlled_data):
    """
    Create main trajectory comparison plot.

    Shows baseline vs controlled trajectories with reference.

    Args:
        baseline_data: dict with keys 'time', 'x', 'y', etc.
        controlled_data: dict with same structure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot reference altitude as horizontal line
    ax.axhline(y=config.reference_altitude, color='green', linestyle='--',
               linewidth=2, label='Reference Altitude', alpha=0.7)

    # Plot baseline trajectory
    ax.plot(baseline_data['x'], baseline_data['y'], 'b-', linewidth=2,
            label='Baseline (No Control)', alpha=0.8)
    ax.plot(baseline_data['x'][0], baseline_data['y'][0], 'bo', markersize=10,
            label='Baseline Start')
    ax.plot(baseline_data['x'][-1], baseline_data['y'][-1], 'bs', markersize=10,
            label='Baseline End')

    # Plot controlled trajectory
    ax.plot(controlled_data['x'], controlled_data['y'], 'r-', linewidth=2,
            label='Controlled', alpha=0.8)
    ax.plot(controlled_data['x'][0], controlled_data['y'][0], 'ro', markersize=10,
            label='Controlled Start')
    ax.plot(controlled_data['x'][-1], controlled_data['y'][-1], 'rs', markersize=10,
            label='Controlled End')

    # Labels and formatting
    ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
    ax.set_ylabel('Altitude (m)', fontsize=12)
    ax.set_title('Vehicle Trajectory: Baseline vs Controlled', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig


def create_altitude_error_plot(baseline_data, controlled_data):
    """
    Create plot showing altitude error vs time.

    Args:
        baseline_data: dict with 'time' and 'y' keys
        controlled_data: dict with 'time' and 'y' keys
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate errors
    baseline_error = np.array(baseline_data['y']) - config.reference_altitude
    controlled_error = np.array(controlled_data['y']) - config.reference_altitude

    # Plot errors
    ax.plot(baseline_data['time'], baseline_error, 'b-', linewidth=2,
            label='Baseline Error', alpha=0.8)
    ax.plot(controlled_data['time'], controlled_error, 'r-', linewidth=2,
            label='Controlled Error', alpha=0.8)

    # Reference line at zero
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Target')

    # Labels and formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Altitude Error (m)', fontsize=12)
    ax.set_title('Altitude Control Error vs Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    return fig


def create_surface_angle_plot(controlled_data):
    """
    Create plot of surface angles over time.

    Args:
        controlled_data: dict with 'time' and surface angle keys
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot commanded and actual surface angles
    if 'surface_commanded' in controlled_data and controlled_data['surface_commanded']:
        commanded = np.array(controlled_data['surface_commanded']) * (180 / math.pi)  # Convert to degrees
        actual = np.array(controlled_data['surface_actual']) * (180 / math.pi)  # Convert to degrees

        ax.plot(controlled_data['time'], commanded, 'g--', linewidth=2,
                label='Commanded Surface Angle', alpha=0.8)
        ax.plot(controlled_data['time'], actual, 'r-', linewidth=2,
                label='Actual Surface Angle', alpha=0.8)

    # Labels and formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Surface Angle (degrees)', fontsize=12)
    ax.set_title('Control Surface Deflection', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    return fig


def create_controller_output_plot(controlled_data):
    """
    Create plot of PID controller outputs.

    Args:
        controlled_data: dict with controller output data
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot altitude error and controller output
    if 'altitude_error' in controlled_data and controlled_data['altitude_error']:
        altitude_error = np.array(controlled_data['altitude_error'])

        ax.plot(controlled_data['time'], altitude_error, 'b-', linewidth=2,
                label='Altitude Error', alpha=0.8)

        # Create second y-axis for pitch command
        ax2 = ax.twinx()
        if 'pitch_command' in controlled_data and controlled_data['pitch_command']:
            pitch_cmd = np.array(controlled_data['pitch_command'])
            ax2.plot(controlled_data['time'], pitch_cmd, 'r--', linewidth=2,
                     label='Pitch Command', alpha=0.8)
            ax2.set_ylabel('Pitch Command (degrees)', fontsize=12, color='r')
            ax2.tick_params(axis='y', labelcolor='r')

    # Labels and formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Altitude Error (m)', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title('Controller Outputs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    return fig


def create_velocity_plot(baseline_data, controlled_data):
    """
    Create plot of velocity vs time.

    Args:
        baseline_data: dict with velocity components
        controlled_data: dict with velocity components
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate velocity magnitude
    baseline_v = np.sqrt(np.array(baseline_data['vx']) ** 2 + np.array(baseline_data['vy']) ** 2)
    controlled_v = np.sqrt(np.array(controlled_data['vx']) ** 2 + np.array(controlled_data['vy']) ** 2)

    # Plot velocities
    ax.plot(baseline_data['time'], baseline_v, 'b-', linewidth=2,
            label='Baseline Airspeed', alpha=0.8)
    ax.plot(controlled_data['time'], controlled_v, 'r-', linewidth=2,
            label='Controlled Airspeed', alpha=0.8)

    # Reference speed
    ax.axhline(y=config.reference_velocity_x, color='green', linestyle='--',
               linewidth=1.5, label='Reference Speed', alpha=0.7)

    # Labels and formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Airspeed (m/s)', fontsize=12)
    ax.set_title('Vehicle Airspeed', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    return fig


def create_dashboard(baseline_data, controlled_data):
    """
    Create a comprehensive dashboard with multiple subplots.

    Args:
        baseline_data: dict with simulation results
        controlled_data: dict with simulation results
    """
    fig = plt.figure(figsize=(16, 12))

    # Trajectory plot (large, top)
    ax1 = plt.subplot(2, 3, (1, 4))
    ax1.plot(baseline_data['x'], baseline_data['y'], 'b-', linewidth=2,
             label='Baseline', alpha=0.8)
    ax1.plot(controlled_data['x'], controlled_data['y'], 'r-', linewidth=2,
             label='Controlled', alpha=0.8)
    ax1.axhline(y=config.reference_altitude, color='green', linestyle='--',
                linewidth=1.5, label='Reference Alt', alpha=0.7)
    ax1.plot(baseline_data['x'][-1], baseline_data['y'][-1], 'bs', markersize=8)
    ax1.plot(controlled_data['x'][-1], controlled_data['y'][-1], 'rs', markersize=8)
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('Trajectory Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Altitude error
    ax2 = plt.subplot(2, 3, 2)
    baseline_error = np.array(baseline_data['y']) - config.reference_altitude
    controlled_error = np.array(controlled_data['y']) - config.reference_altitude
    ax2.plot(baseline_data['time'], baseline_error, 'b-', linewidth=1.5, label='Baseline')
    ax2.plot(controlled_data['time'], controlled_error, 'r-', linewidth=1.5, label='Controlled')
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude Error (m)')
    ax2.set_title('Altitude Error')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Velocity
    ax3 = plt.subplot(2, 3, 3)
    baseline_v = np.sqrt(np.array(baseline_data['vx']) ** 2 + np.array(baseline_data['vy']) ** 2)
    controlled_v = np.sqrt(np.array(controlled_data['vx']) ** 2 + np.array(controlled_data['vy']) ** 2)
    ax3.plot(baseline_data['time'], baseline_v, 'b-', linewidth=1.5, label='Baseline')
    ax3.plot(controlled_data['time'], controlled_v, 'r-', linewidth=1.5, label='Controlled')
    ax3.axhline(y=config.reference_velocity_x, color='green', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Airspeed (m/s)')
    ax3.set_title('Airspeed')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Surface angles
    ax4 = plt.subplot(2, 3, 5)
    if 'surface_actual' in controlled_data and controlled_data['surface_actual']:
        surface_deg = np.array(controlled_data['surface_actual']) * (180 / math.pi)
        ax4.plot(controlled_data['time'], surface_deg, 'r-', linewidth=1.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Surface Angle (deg)')
        ax4.set_title('Control Surface Deflection')
        ax4.grid(True, alpha=0.3)

    # Pitch angle
    ax5 = plt.subplot(2, 3, 6)
    baseline_pitch = np.array(baseline_data['pitch']) * (180 / math.pi)
    controlled_pitch = np.array(controlled_data['pitch']) * (180 / math.pi)
    ax5.plot(baseline_data['time'], baseline_pitch, 'b-', linewidth=1.5, label='Baseline')
    ax5.plot(controlled_data['time'], controlled_pitch, 'r-', linewidth=1.5, label='Controlled')
    ax5.axhline(y=config.reference_pitch_angle, color='green', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Pitch Angle (degrees)')
    ax5.set_title('Pitch Angle')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    plt.suptitle('Flight Control Simulation - Comprehensive Dashboard',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig


def create_ballistic_vs_guided_plot(ballistic_data, guided_data, ballistic_trajectory=None):
    """
    Create plot comparing ballistic miss vs guided hit.

    This is the key visualization showing WHY aerodynamics matter.

    Args:
        ballistic_data: unguided ballistic shot data
        guided_data: guided control shot data
        ballistic_trajectory: predicted ballistic path (no aerodynamics)
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    # Target location and CEP circle
    target_x = config.target_position_x
    target_y = config.target_position_y

    # Draw target
    ax.plot(target_x, target_y, 'g*', markersize=30, label='TARGET', zorder=5)
    ax.plot(target_x, target_y, 'g+', markersize=20, markeredgewidth=2)

    # Draw CEP circle (error radius)
    cep_circle = plt.Circle((target_x, target_y), config.cep_radius,
                            color='green', fill=False, linestyle='--',
                            linewidth=2, alpha=0.5, label=f'CEP Radius ({config.cep_radius}m)')
    ax.add_patch(cep_circle)

    # Plot predicted ballistic trajectory (no aerodynamics)
    if ballistic_trajectory:
        ax.plot(ballistic_trajectory[0], ballistic_trajectory[1], 'g--',
                linewidth=2, alpha=0.6, label='Predicted Ballistic Path (no drag)')

    # Plot actual ballistic shot (misses due to drag)
    ax.plot(ballistic_data['x'], ballistic_data['y'], 'b-', linewidth=2.5,
            label='BALLISTIC SHOT (Unguided - MISSES)', alpha=0.8)
    ax.plot(ballistic_data['x'][0], ballistic_data['y'][0], 'bo', markersize=12,
            label='Ballistic Launch', zorder=4)
    ax.plot(ballistic_data['x'][-1], ballistic_data['y'][-1], 'bs', markersize=12,
            label=f"Ballistic Impact (Miss: {ballistic_data.get('miss_distance', 0):.1f}m)",
            zorder=4)

    # Plot guided shot (hits or near-hits target)
    ax.plot(guided_data['x'], guided_data['y'], 'r-', linewidth=2.5,
            label='GUIDED SHOT (With Control - HITS)', alpha=0.8)
    ax.plot(guided_data['x'][0], guided_data['y'][0], 'ro', markersize=12,
            label='Guided Launch', zorder=4)
    ax.plot(guided_data['x'][-1], guided_data['y'][-1], 'r^', markersize=12,
            label=f"Guided Impact (Miss: {guided_data.get('miss_distance', 0):.1f}m)",
            zorder=4)

    # Reference altitude band
    ax.axhline(y=config.target_position_y, color='green', linestyle=':',
               linewidth=1, alpha=0.3)

    # Labels and formatting
    ax.set_xlabel('Horizontal Distance (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Altitude (m)', fontsize=13, fontweight='bold')
    ax.set_title('BALLISTIC vs GUIDED: Why Aerodynamic Control Matters\n'
                 'Naive projectile math misses; control systems correct',
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

    # Add annotation box
    textstr = (
        f"Target: ({config.target_position_x:.0f}m, {config.target_position_y:.0f}m)\n"
        f"Ballistic Miss: {ballistic_data.get('miss_distance', 0):.1f}m\n"
        f"Guided Miss: {guided_data.get('miss_distance', 0):.1f}m\n"
        f"Improvement: {ballistic_data.get('miss_distance', 0) - guided_data.get('miss_distance', 0):.1f}m"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')

    plt.tight_layout()
    return fig


def create_cep_heatmap(ballistic_data, guided_data):
    """
    Create a view showing miss distances on a circular target.

    Args:
        ballistic_data: ballistic shot results
        guided_data: guided shot results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    target_x = config.target_position_x
    target_y = config.target_position_y

    # Ballistic subplot
    ax1.set_xlim(target_x - 100, target_x + 100)
    ax1.set_ylim(target_y - 100, target_y + 100)
    ax1.set_aspect('equal')

    # Target circle
    circle1 = plt.Circle((target_x, target_y), config.cep_radius,
                         color='green', fill=False, linestyle='--', linewidth=2)
    ax1.add_patch(circle1)
    ax1.plot(target_x, target_y, 'g*', markersize=25)

    # Ballistic impact
    ballistic_impact_x = ballistic_data['x'][-1]
    ballistic_impact_y = ballistic_data['y'][-1]
    ax1.plot(ballistic_impact_x, ballistic_impact_y, 'bs', markersize=15,
             label=f"Miss: {ballistic_data.get('miss_distance', 0):.1f}m")
    ax1.arrow(target_x, target_y, ballistic_impact_x - target_x,
              ballistic_impact_y - target_y, head_width=5, head_length=5,
              fc='blue', ec='blue', alpha=0.5)

    ax1.set_xlabel('Horizontal Distance (m)')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('BALLISTIC SHOT - MISSES\n(Aerodynamic drag not calculated in launch)',
                  fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Guided subplot
    ax2.set_xlim(target_x - 100, target_x + 100)
    ax2.set_ylim(target_y - 100, target_y + 100)
    ax2.set_aspect('equal')

    circle2 = plt.Circle((target_x, target_y), config.cep_radius,
                         color='green', fill=False, linestyle='--', linewidth=2)
    ax2.add_patch(circle2)
    ax2.plot(target_x, target_y, 'g*', markersize=25)

    # Guided impact
    guided_impact_x = guided_data['x'][-1]
    guided_impact_y = guided_data['y'][-1]
    ax2.plot(guided_impact_x, guided_impact_y, 'r^', markersize=15,
             label=f"Miss: {guided_data.get('miss_distance', 0):.1f}m")
    ax2.arrow(target_x, target_y, guided_impact_x - target_x,
              guided_impact_y - target_y, head_width=5, head_length=5,
              fc='red', ec='red', alpha=0.5)

    ax2.set_xlabel('Horizontal Distance (m)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('GUIDED SHOT - HITS\n(Control system corrects for aerodynamics)',
                  fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.suptitle('CEP (Circular Error Probable) Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def display_summary_statistics(baseline_data, controlled_data):
    """
    Print summary statistics comparing scenarios.

    Args:
        baseline_data: dict with simulation results
        controlled_data: dict with simulation results
    """
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY STATISTICS")
    print("=" * 70)

    # Altitude statistics
    baseline_alt_final = baseline_data['y'][-1]
    controlled_alt_final = controlled_data['y'][-1]
    baseline_alt_error = abs(baseline_alt_final - config.reference_altitude)
    controlled_alt_error = abs(controlled_alt_final - config.reference_altitude)

    print("\nALTITUDE CONTROL:")
    print(f"  Reference altitude:           {config.reference_altitude:.2f} m")
    print(f"  Baseline final altitude:      {baseline_alt_final:.2f} m (error: {baseline_alt_error:.2f} m)")
    print(f"  Controlled final altitude:    {controlled_alt_final:.2f} m (error: {controlled_alt_error:.2f} m)")
    print(f"  Improvement:                  {baseline_alt_error - controlled_alt_error:.2f} m")

    # Altitude RMS error
    baseline_alt_rms = np.sqrt(np.mean((np.array(baseline_data['y']) - config.reference_altitude) ** 2))
    controlled_alt_rms = np.sqrt(np.mean((np.array(controlled_data['y']) - config.reference_altitude) ** 2))

    print(f"\n  Baseline RMS altitude error:  {baseline_alt_rms:.2f} m")
    print(f"  Controlled RMS altitude error: {controlled_alt_rms:.2f} m")

    # Velocity statistics
    baseline_v = np.sqrt(np.array(baseline_data['vx']) ** 2 + np.array(baseline_data['vy']) ** 2)
    controlled_v = np.sqrt(np.array(controlled_data['vx']) ** 2 + np.array(controlled_data['vy']) ** 2)
    baseline_v_final = baseline_v[-1]
    controlled_v_final = controlled_v[-1]

    print("\nVELOCITY:")
    print(f"  Reference velocity:          {config.reference_velocity_x:.2f} m/s")
    print(f"  Baseline final velocity:     {baseline_v_final:.2f} m/s")
    print(f"  Controlled final velocity:   {controlled_v_final:.2f} m/s")

    # Distance traveled
    baseline_distance = baseline_data['x'][-1] - baseline_data['x'][0]
    controlled_distance = controlled_data['x'][-1] - controlled_data['x'][0]

    print("\nTRAJECTORY:")
    print(f"  Baseline distance traveled:  {baseline_distance:.2f} m")
    print(f"  Controlled distance traveled: {controlled_distance:.2f} m")

    # Pitch statistics
    baseline_pitch_mean = np.mean(np.array(baseline_data['pitch']) * (180 / math.pi))
    controlled_pitch_mean = np.mean(np.array(controlled_data['pitch']) * (180 / math.pi))

    print("\nATTITUDE:")
    print(f"  Baseline mean pitch:         {baseline_pitch_mean:.2f} degrees")
    print(f"  Controlled mean pitch:       {controlled_pitch_mean:.2f} degrees")

    print("\n" + "=" * 70)