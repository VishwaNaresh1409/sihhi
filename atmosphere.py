"""
Atmospheric and environmental property calculations.

Provides functions for air density, speed of sound, and other atmospheric properties
that vary with altitude and conditions.
"""

import math
import config


def get_air_density(altitude_m):
    """
    Calculate air density at given altitude.

    Uses simplified linear model valid for low altitudes (< 10 km).
    At sea level: 1.225 kg/m^3
    Decreases approximately 0.0001184 kg/m^3 per meter

    Args:
        altitude_m: altitude in meters

    Returns:
        air_density: density in kg/m^3
    """
    # Simplified model
    sea_level_density = 1.225  # kg/m^3
    density_gradient = -0.0001184  # kg/m^3 per meter

    rho = sea_level_density + (density_gradient * altitude_m)

    # Clamp to realistic minimum
    rho = max(rho, 0.1)

    return rho


def get_speed_of_sound(altitude_m, temperature_kelvin=288.15):
    """
    Calculate speed of sound at given altitude and temperature.

    Uses simplified standard atmosphere model.

    Args:
        altitude_m: altitude in meters
        temperature_kelvin: temperature in Kelvin (default: sea level standard day)

    Returns:
        speed_of_sound: in m/s
    """
    # Temperature decreases with altitude in troposphere
    # Standard lapse rate: -6.5 K/km
    lapse_rate = -0.0065  # K/m
    temperature = temperature_kelvin + (lapse_rate * altitude_m)

    # Speed of sound: a = sqrt(gamma * R * T)
    # gamma = 1.4 for air, R = 287 J/(kg*K)
    gamma = 1.4
    gas_constant_air = 287.0

    speed_of_sound = math.sqrt(gamma * gas_constant_air * temperature)

    return speed_of_sound


def get_wind_vector(time_s, altitude_m=0.0):
    """
    Get wind velocity at given time and altitude.

    Combines base wind with slowly-varying disturbance.

    Args:
        time_s: current simulation time in seconds
        altitude_m: altitude in meters (for altitude-dependent wind if added)

    Returns:
        wind_x, wind_y: wind velocity components in m/s
    """
    # Base wind
    wind_x = config.base_wind_speed_x
    wind_y = config.base_wind_speed_y

    # Add disturbance based on configured model
    if config.wind_disturbance_type == "sine":
        # Smooth sinusoidal variation
        disturbance = config.wind_disturbance_amplitude * math.sin(
            2.0 * math.pi * config.wind_disturbance_frequency * time_s
        )
        wind_x += disturbance

    elif config.wind_disturbance_type == "step":
        # Step change at specific times
        if time_s > 5.0 and time_s < 15.0:
            wind_x += config.wind_disturbance_amplitude

    elif config.wind_disturbance_type == "noise":
        # Smoothly interpolated random-like variation
        # Use hash of time to get pseudo-random but deterministic values
        seed = int(time_s * 10)  # Quantize time to 0.1s steps
        pseudo_rand = (math.sin(seed * 12.9898) * 43758.5453) % 1.0
        disturbance = (pseudo_rand - 0.5) * config.wind_disturbance_amplitude * 2.0
        wind_x += disturbance

    return wind_x, wind_y