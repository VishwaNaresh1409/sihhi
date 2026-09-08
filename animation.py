"""
Simple 2D animation of vehicle flight.

Creates an animated visualization of the trajectory and vehicle state.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import math
import config


def create_simple_animation(baseline_data, controlled_data, interval_ms=50):
    """
    Create a simple 2D animation showing vehicle trajectories.

    Args:
        baseline_data: dict with time, x, y position data
        controlled_data: dict with time, x, y position data
        interval_ms: milliseconds between frames
    """
    # Determine time points to animate
    num_frames = min(len(baseline_data['time']), len(controlled_data['time']))
    frame_indices = np.linspace(0, num_frames - 1, min(num_frames, 200), dtype=int)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Initialize plot elements
    ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
    ax.set_ylabel('Altitude (m)', fontsize=12)
    ax.set_title('Vehicle Flight Animation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Set axis limits
    x_min = min(min(baseline_data['x']), min(controlled_data['x'])) - 50
    x_max = max(max(baseline_data['x']), max(controlled_data['x'])) + 50
    y_min = 0
    y_max = max(max(baseline_data['y']), max(controlled_data['y'])) + 20

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Reference altitude line
    ax.axhline(y=config.reference_altitude, color='green', linestyle='--',
               linewidth=2, alpha=0.5, label='Reference Altitude')

    # Line objects for trajectories
    line_baseline, = ax.plot([], [], 'b-', linewidth=2, alpha=0.6, label='Baseline')
    line_controlled, = ax.plot([], [], 'r-', linewidth=2, alpha=0.6, label='Controlled')

    # Markers for current positions
    marker_baseline, = ax.plot([], [], 'bo', markersize=10, label='Baseline Position')
    marker_controlled, = ax.plot([], [], 'rs', markersize=10, label='Controlled Position')

    # Legend
    ax.legend(loc='upper left', fontsize=10)

    # Time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle='round',
                                                           facecolor='wheat', alpha=0.5))

    def update_frame(frame_idx):
        """Update animation frame."""
        idx = frame_indices[frame_idx]

        # Update trajectories
        line_baseline.set_data(baseline_data['x'][:idx + 1], baseline_data['y'][:idx + 1])
        line_controlled.set_data(controlled_data['x'][:idx + 1], controlled_data['y'][:idx + 1])

        # Update current position markers
        marker_baseline.set_data([baseline_data['x'][idx]], [baseline_data['y'][idx]])
        marker_controlled.set_data([controlled_data['x'][idx]], [controlled_data['y'][idx]])

        # Update time display
        time_text.set_text(f"Time: {baseline_data['time'][idx]:.2f} s")

        return line_baseline, line_controlled, marker_baseline, marker_controlled, time_text

    # Create animation
    anim = animation.FuncAnimation(fig, update_frame, frames=len(frame_indices),
                                   interval=interval_ms, blit=True, repeat=True)

    return fig, anim