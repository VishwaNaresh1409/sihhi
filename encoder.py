"""
Rotary encoder sensor simulation.

Measures control surface angles with realistic noise.
"""

import numpy as np
import math
import config


class SurfaceAngleEncoder:
    """
    Simulated encoder for measuring control surface angle.
    """

    def __init__(self):
        """Initialize encoder."""
        self.noise_std = config.encoder_angle_noise_std  # degrees
        self.quantization_bits = config.encoder_quantization_bits

        self.random_state = np.random.RandomState(seed=43)

    def measure(self, true_angle_rad):
        """
        Measure surface angle with noise.

        Args:
            true_angle_rad: true surface angle in radians

        Returns:
            measured_angle_rad: noisy measurement in radians
        """
        # Convert to degrees for noise calculation
        true_angle_deg = math.degrees(true_angle_rad)

        # Add Gaussian noise
        noise_deg = self.random_state.normal(0, self.noise_std)
        measured_angle_deg = true_angle_deg + noise_deg

        # Quantization (if enabled)
        if self.quantization_bits > 0:
            # Simple mid-tread quantizer
            quantization_level = 360.0 / (2 ** self.quantization_bits)
            measured_angle_deg = quantization_level * round(measured_angle_deg / quantization_level)

        # Convert back to radians
        measured_angle_rad = math.radians(measured_angle_deg)

        return measured_angle_rad