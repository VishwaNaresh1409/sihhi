"""
Environmental disturbances and external effects.

Handles wind gusts, atmospheric turbulence, and other environmental factors.
"""

import math
import config


class EnvironmentalDisturbance:
    """
    Manages environmental disturbances like wind gusts.
    """

    def __init__(self):
        """Initialize disturbance model."""
        self.current_time = 0.0

    def update(self, time_s):
        """
        Update disturbance model.

        Args:
            time_s: current simulation time in seconds
        """
        self.current_time = time_s

    def get_gust_velocity(self):
        """
        Get gust/disturbance velocity components.

        For now, disturbances are handled in atmosphere.py wind model.
        This module is a placeholder for future expansion.

        Returns:
            gust_x, gust_y: gust velocity components in m/s
        """
        # Placeholder - actual gusts handled in wind model
        return 0.0, 0.0

    def get_altitude_correction(self):
        """
        Get any altitude-related corrections.

        Placeholder for future use.

        Returns:
            correction_y: vertical correction factor
        """
        return 1.0