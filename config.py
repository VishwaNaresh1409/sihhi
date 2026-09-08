"""
Configuration and parameter definitions for the aerodynamic vehicle simulation.

All physical parameters, simulation settings, and tuning constants are centralised here.

FIXES vs original:
- Removed duplicate TARGET AND BALLISTIC MODE block
- PID gains tuned for PKG flight time (~17s at 35m/s over 600m)
- terminal_guidance_kp is now a unitless scale factor (not a PID gain)
- Added vehicle_pitch_inertia (was already there but confirm it's reasonable)
"""
"""
Configuration and parameter definitions for the aerodynamic vehicle simulation.
"""

# =========================
# SIMULATION TIME PARAMETERS
# =========================
simulation_time = 15.0       # seconds
time_step = 0.01             # seconds (100 Hz)
sensor_update_period = 0.05  # seconds (20 Hz)
controller_update_period = 0.05  # seconds (20 Hz)

# =========================
# COORDINATE SYSTEM & GRAVITY
# =========================
gravity = 9.81      # m/s^2
air_density = 1.225  # kg/m^3

# =========================
# VEHICLE GEOMETRY & MASS
# =========================
vehicle_mass = 10.0               # kg
vehicle_reference_length = 2.0    # m
vehicle_reference_area = 0.05     # m^2
vehicle_pitch_inertia = 3.0       # kg*m^2

# =========================
# BODY AERODYNAMICS
# =========================
body_drag_coefficient = 0.02
body_lift_coefficient_per_angle = 0.01
body_zero_lift_pitch_angle = 0.0

# =========================
# CONTROL SURFACES
# =========================
num_control_surfaces = 2
control_surface_area = 0.02                           # m^2
control_surface_max_angle = 15.0                      # degrees
control_surface_min_angle = -15.0                     # degrees
control_surface_max_rotation_speed = 60.0             # degrees/second
control_surface_lift_coefficient_per_angle = 0.03     # C_L per degree
control_surface_drag_coefficient_increase = 0.0005

# =========================
# WIND & ENVIRONMENT
# =========================
base_wind_speed_x = 2.0
base_wind_speed_y = 0.0
wind_disturbance_amplitude = 1.0
wind_disturbance_frequency = 0.5
wind_disturbance_type = "sine"

# =========================
# VEHICLE INITIAL CONDITIONS
# =========================
initial_position_x = 0.0
initial_position_y = 100.0
initial_velocity_x = 100.0
initial_velocity_y = 0.0
initial_pitch_angle = 0.0

# =========================
# REFERENCE TRAJECTORY
# =========================
reference_altitude = 100.0
reference_velocity_x = 100.0
reference_pitch_angle = 0.0

# =========================
# SENSOR PROPERTIES
# =========================
imu_accel_noise_std = 0.05
imu_accel_bias = [0.0, 0.0]
encoder_angle_noise_std = 0.5
encoder_quantization_bits = 12

# =========================
# ACTUATOR PROPERTIES
# =========================
actuator_max_speed = 60.0
actuator_response_delay = 0.0
actuator_damping = 0.2

# =========================
# CONTROLLER (PID) TUNING
# =========================
altitude_controller_kp = 1.2
altitude_controller_ki = 0.0
altitude_controller_kd = 0.2
altitude_controller_output_limit = 18.0

pitch_controller_kp = 1.5
pitch_controller_ki = 0.0
pitch_controller_kd = 0.3
pitch_controller_output_limit = 12.0
pid_integral_limit = 0.0

# =========================
# TARGET AND BALLISTIC MODE
# =========================
ballistic_mode_enabled = True
target_position_x = 500.0
target_position_y = 100.0
ballistic_launch_altitude = 100.0
ballistic_launch_velocity = 100.0
cep_radius = 20.0

run_scenario_ballistic = True
run_scenario_guided = True
run_scenario_comparison = True

# =========================
# GUIDANCE MODE
# =========================
guidance_mode = "trajectory"
terminal_guidance_enable = False
terminal_guidance_distance = 100.0
terminal_guidance_kp = 0.5

# =========================
# SCENARIO SETUP
# =========================
run_scenario_baseline = True
run_scenario_controlled = True
run_scenario_disturbance = True

# =========================
# VISUALIZATION & LOGGING
# =========================
plot_show = True
plot_save = False
plot_save_dir = "./plots"
animation_enabled = True
animation_interval = 50
animation_save = False
log_level = "INFO"
print_summary = True

# =========================
# LOGGING & OUTPUT
# =========================

log_level = "INFO"
print_summary = True