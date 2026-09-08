"""
Inertial Measurement Unit (IMU) sensor simulation.

Provides accelerometer measurements with realistic noise and bias.
"""

import numpy as np
import config


class IMU:
    """
    Simulated IMU accelerometer.

    Measures acceleration with Gaussian noise.
    Real IMU would also include gyroscope, magnetometer, etc.
    """

    def __init__(self):
        """Initialize IMU."""
        self.noise_std = config.imu_accel_noise_std  # m/s^2
        self.bias = config.imu_accel_bias  # [ax_bias, ay_bias] in m/s^2

        # For realistic simulation, could add bias drift, random walk, etc.
        self.random_state = np.random.RandomState(seed=42)

    def measure(self, true_acceleration_x, true_acceleration_y):
        """
        Measure acceleration with noise.

        Args:
            true_acceleration_x: true X acceleration in m/s^2
            true_acceleration_y: true Y acceleration in m/s^2

        Returns:
            measured_accel_x, measured_accel_y: noisy measurements in m/s^2
        """
        # Add bias
        biased_accel_x = true_acceleration_x + self.bias[0]
        biased_accel_y = true_acceleration_y + self.bias[1]

        # Add Gaussian noise
        noise_x = self.random_state.normal(0, self.noise_std)
        noise_y = self.random_state.normal(0, self.noise_std)

        measured_accel_x = biased_accel_x + noise_x
        measured_accel_y = biased_accel_y + noise_y

        return measured_accel_x, measured_accel_y