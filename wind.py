"""
Wind and atmospheric disturbance models.

Provides time-varying wind and environmental effects.
"""

import math
import config
from atmosphere import get_wind_vector


class WindModel:
    """
    Manages wind and atmospheric disturbances.
    """

    def __init__(self):
        """Initialize wind model."""
        self.current_time = 0.0

    def update(self, time_s):
        """
        Update wind model to current time.

        Args:
            time_s: current simulation time in seconds
        """
        self.current_time = time_s

    def get_wind(self, altitude_m=0.0):
        """
        Get wind velocity at current time and altitude.

        Args:
            altitude_m: altitude in meters

        Returns:
            wind_x, wind_y: wind velocity in m/s
        """
        wind_x, wind_y = get_wind_vector(self.current_time, altitude_m)
        return wind_x, wind_y