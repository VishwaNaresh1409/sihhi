"""
Configuration and parameter definitions for the aerodynamic vehicle simulation.

All physical parameters, simulation settings, and tuning constants are centralized here.
Change values here to experiment with different vehicle, environment, and control configurations.
"""

# =========================
# SIMULATION TIME PARAMETERS
# =========================

simulation_time = 30.0  # seconds - total duration of simulation
time_step = 0.01        # seconds - physics integration timestep (100 Hz)
sensor_update_period = 0.05  # seconds - how often sensors produce new measurements (20 Hz)
controller_update_period = 0.05  # seconds - how often controller runs (20 Hz)


# =========================
# COORDINATE SYSTEM & GRAVITY
# =========================

gravity = 9.81  # m/s^2 - standard Earth gravity, positive downward in physics

# Atmospheric conditions
air_density = 1.225  # kg/m^3 - sea level standard atmosphere


# =========================
# VEHICLE GEOMETRY & MASS
# =========================

vehicle_mass = 10.0  # kg - total vehicle mass
vehicle_reference_length = 2.0  # m - characteristic length (used for some aero models)
vehicle_reference_area = 0.5  # m^2 - reference planform area for body aerodynamics

# Vehicle inertia (simplified - for pitch rotation if modeled)
vehicle_pitch_inertia = 0.5  # kg*m^2 - moment of inertia about pitch axis (simplified)


# =========================
# BODY AERODYNAMICS
# =========================

# Lift and drag coefficients for the main body
# These are very simplified and constant; a real model would vary with angle of attack
body_drag_coefficient = 0.02  # C_D - body drag (baseline)
body_lift_coefficient_per_angle = 0.1  # C_L per degree of pitch - slope of lift curve
body_zero_lift_pitch_angle = 0.0  # degrees - pitch angle at which body produces zero lift

# Angle of attack calculation uses vehicle pitch angle and velocity vector
# This is a simplification; a full model would track angle between velocity and body axis


# =========================
# CONTROL SURFACES
# =========================

# Two controllable surfaces (e.g., left and right elevators or flaps)
num_control_surfaces = 2

# Each surface properties (assumed identical for simplicity)
control_surface_area = 0.2  # m^2 - planform area of control surface
control_surface_max_angle = 25.0  # degrees - maximum deflection magnitude
control_surface_min_angle = -25.0  # degrees - minimum deflection magnitude
control_surface_max_rotation_speed = 45.0  # degrees/second - max rate of deflection change

# Aerodynamic effectiveness of control surface
# Lift coefficient = C_L_surface * surface_angle (linear approximation)
control_surface_lift_coefficient_per_angle = 0.15  # per degree of deflection
control_surface_drag_coefficient_increase = 0.001  # additional drag per degree deflection squared


# =========================
# WIND & ENVIRONMENT
# =========================

# Base wind speed and direction
base_wind_speed_x = 2.0  # m/s - constant wind in X direction
base_wind_speed_y = 0.0  # m/s - constant wind in Y direction (rarely used for 2D)

# Wind disturbance model
wind_disturbance_amplitude = 1.0  # m/s - magnitude of wind variation
wind_disturbance_frequency = 0.5  # Hz - how fast wind varies (sine wave)
wind_disturbance_type = "sine"  # options: "sine", "step", "noise"


# =========================
# VEHICLE INITIAL CONDITIONS
# =========================

initial_position_x = 0.0  # m
initial_position_y = 100.0  # m - start at altitude
initial_velocity_x = 30.0  # m/s - forward speed
initial_velocity_y = 0.0  # m/s - vertical speed (zero = level flight)
initial_pitch_angle = 0.0  # degrees - nose direction


# =========================
# REFERENCE TRAJECTORY (Desired state for control)
# =========================

# The vehicle should try to maintain these conditions
reference_altitude = 100.0  # m - desired height
reference_velocity_x = 30.0  # m/s - desired forward speed (cruise)
reference_pitch_angle = 0.0  # degrees - desired pitch


# =========================
# SENSOR PROPERTIES
# =========================

# IMU (Inertial Measurement Unit) - accelerometer
imu_accel_noise_std = 0.05  # m/s^2 - standard deviation of acceleration measurement noise
imu_accel_bias = [0.0, 0.0]  # m/s^2 - constant bias in accelerometer [x, y]

# Encoder - measures control surface angle
encoder_angle_noise_std = 0.5  # degrees - standard deviation of angle measurement
encoder_quantization_bits = 12  # bits - if using quantization (0 to disable)


# =========================
# ACTUATOR PROPERTIES
# =========================

# Control surface actuators
actuator_max_speed = 45.0  # degrees/second - how fast actuator can move surface
actuator_response_delay = 0.0  # seconds - lag in actuator response (simplified model)
actuator_damping = 0.5  # fraction - damping applied to smooth motion (0=none, 1=critically damped)


# =========================
# CONTROLLER (PID) TUNING
# =========================

# PID gains for altitude control loop
altitude_controller_kp = 0.1  # proportional gain
altitude_controller_ki = 0.01  # integral gain
altitude_controller_kd = 0.05  # derivative gain
altitude_controller_output_limit = 20.0  # degrees - max pitch command

# PID gains for pitch control loop (secondary loop)
pitch_controller_kp = 2.0  # proportional gain
pitch_controller_ki = 0.1  # integral gain
pitch_controller_kd = 0.5  # derivative gain
pitch_controller_output_limit = 25.0  # degrees - max surface deflection

# Integral windup protection
pid_integral_limit = 10.0  # prevent integral term from growing unbounded


# =========================
# VISUALIZATION
# =========================

plot_show = True  # whether to display plots
plot_save = False  # whether to save plot images
plot_save_dir = "./plots"  # directory for saved plots

animation_enabled = True  # whether to generate animation
animation_interval = 50  # ms between animation frames
animation_save = False  # whether to save animation


# =========================
# LOGGING & OUTPUT
# =========================

log_level = "INFO"  # "DEBUG", "INFO", "WARNING"
print_summary = True  # print summary statistics at end

# =========================
# TARGET AND BALLISTIC MODE
# =========================

# Enable ballistic vs guided comparison
ballistic_mode_enabled = True

# Target location
target_position_x = 600.0  # m - horizontal distance to target
target_position_y = 100.0  # m - altitude of target

# Launch parameters for ballistic shot
ballistic_launch_altitude = 100.0  # m - starting altitude for ballistic shot
ballistic_launch_velocity = 35.0  # m/s - launch speed (will be calculated)
ballistic_launch_pitch = 0.0  # degrees - initial pitch (will be calculated)

# CEP (Circular Error Probable) visualization
cep_radius = 20.0  # m - show this error circle around target

# Scenario selection
run_scenario_ballistic = True  # unguided ballistic shot
run_scenario_guided = True  # guided to target
run_scenario_comparison = True  # side-by-side comparison

# =========================
# TARGET AND BALLISTIC MODE
# =========================

ballistic_mode_enabled = True

# Target location
target_position_x = 600.0  # m - horizontal distance to target
target_position_y = 100.0  # m - altitude of target

# Launch parameters
ballistic_launch_altitude = 100.0  # m
ballistic_launch_velocity = 35.0  # m/s

# CEP visualization
cep_radius = 20.0  # m - error circle radius

# Scenarios
run_scenario_ballistic = True
run_scenario_guided = True
# =========================


# GUIDANCE MODE
# =========================

# Control strategy for ballistic + guided scenario
guidance_mode = "trajectory"  # options: "altitude", "trajectory", "terminal_guidance"

# Trajectory guidance: follow a pre-calculated path to target
guidance_trajectory_points = None  # will be calculated from ballistic solution

# Terminal guidance: switch from ballistic to guidance near target
terminal_guidance_enable = True
terminal_guidance_distance = 150.0  # meters - switch to guidance when this close
terminal_guidance_kp = 0.3  # stronger control near target

# =========================
# SCENARIO SETUP
# =========================

# Run multiple scenarios automatically
run_scenario_baseline = True  # uncontrolled flight
run_scenario_controlled = True  # with control system active
run_scenario_disturbance = True  # with environmental disturbance