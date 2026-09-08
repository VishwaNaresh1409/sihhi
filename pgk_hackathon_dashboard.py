"""
dashboard.py — Standalone Fixed-Canard Phase Guidance Dashboard
===============================================================

Fully self-contained.  Drop this file next to the existing project
files (config.py, vehicle.py, dynamics.py, controller.py, forces.py,
atmosphere.py, wind.py, actuator.py, projectile_motion.py) and run:

    python dashboard.py

It re-runs the simulation, captures all telemetry, then produces two
publication-quality figures:

  Figure 1  —  Mission Overview
        Panel A  (large left)    Trajectory: ballistic miss vs guided hit
        Panel B  (top-right)     Altitude error convergence
        Panel C  (mid-right)     Airspeed comparison
        Panel D  (bottom-right)  CEP target view

  Figure 2  —  Fixed-Canard Phase Guidance Telemetry
        Row 1    Required aerodynamic correction (PD output)
        Row 2    Canard clock orientation vs time
        Row 3    Normalised correction demand (−1 to +1)
        Row 4    Command-to-orientation mapping consistency

Debug print statements are emitted at every significant step so you can
follow exactly what the simulator is doing.
"""

import math
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── make sure project root is on the path ────────────────────────────────────
import os
sys.path.insert(0, os.path.dirname(__file__))

# ── project imports ───────────────────────────────────────────────────────────
import config
from vehicle import Vehicle
from dynamics import VehicleDynamics
from controller import VehicleController
from wind import WindModel
from projectile_motion import (
    calculate_ballistic_solution,
    calculate_ballistic_trajectory,
)

print("=" * 70)
print("  FIXED-CANARD PHASE GUIDANCE  —  STANDALONE DASHBOARD")
print("=" * 70)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1  —  BALLISTIC SOLUTION
# ═════════════════════════════════════════════════════════════════════════════
print("\n[STEP 1/5]  Calculating ballistic solution …")

b_sol = calculate_ballistic_solution(
    target_x=config.target_position_x,
    target_y=config.target_position_y,
    launch_x=0.0,
    launch_y=config.ballistic_launch_altitude,
    initial_velocity_magnitude=config.ballistic_launch_velocity,
)

if not b_sol["success"]:
    print(f"  ✗  No ballistic solution: {b_sol['reason']}")
    sys.exit(1)

print(f"  ✓  Launch angle   : {b_sol['launch_angle_deg']:.3f}°")
print(f"  ✓  Launch velocity: {b_sol['launch_velocity']:.2f} m/s")
print(f"  ✓  Flight time est: {b_sol['time_to_target']:.3f} s")
print(f"  ⚠  Drag ignored in ballistic solution — real shot will miss!")

# Predicted (drag-free) trajectory for the overlay
pred_bx, pred_by = calculate_ballistic_trajectory(
    launch_x=0.0,
    launch_y=config.ballistic_launch_altitude,
    launch_velocity=b_sol["launch_velocity"],
    launch_angle_rad=b_sol["launch_angle_rad"],
    num_points=200,
)
print(f"  ✓  Predicted arc computed ({len(pred_bx)} points)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2  —  SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _make_vehicle_at_launch(b):
    """Return a Vehicle initialised to the ballistic launch state."""
    v = Vehicle()
    v.position_x = 0.0
    v.position_y = config.ballistic_launch_altitude
    v.velocity_x = b["launch_velocity"] * math.cos(b["launch_angle_rad"])
    v.velocity_y = b["launch_velocity"] * math.sin(b["launch_angle_rad"])
    v.pitch_angle = b["launch_angle_rad"]
    v.pitch_rate  = 0.0
    return v


def _run_until_target(vehicle, dynamics, controller, wind_model,
                      enable_control, guidance_trajectory,
                      tag="", report_every=500):
    """
    Generic physics loop.  Returns a data dict with all telemetry.
    Prints a progress line every `report_every` steps.
    """
    dt          = config.time_step
    ctrl_dt     = config.controller_update_period
    t           = 0.0
    step        = 0
    next_ctrl   = ctrl_dt

    rec = dict(
        time=[], x=[], y=[], vx=[], vy=[],
        pitch=[], pitch_rate=[], wind_x=[], wind_y=[],
        surface_cmd=[], surface_actual=[],
        altitude_error=[], pitch_command=[],
        # fixed-canard telemetry
        required_correction_deg=[],
        required_correction_fraction=[],
        required_canard_clock_angle_deg=[],
        current_canard_clock_angle_deg=[],
        flight_path_error_deg=[],
        pitch_error_deg=[],
    )

    wind_x, wind_y = 0.0, 0.0

    while t < config.simulation_time:
        if vehicle.position_y < 0:
            print(f"  [{tag}] Ground impact at t={t:.3f}s  x={vehicle.position_x:.1f}m")
            break
        if vehicle.position_x >= config.target_position_x:
            print(f"  [{tag}] Reached target x-range at t={t:.3f}s")
            break

        # wind
        wind_model.update(t)
        wind_x, wind_y = wind_model.get_wind(vehicle.position_y)
        dynamics.set_wind(wind_x, wind_y)

        # controller
        if enable_control and t >= next_ctrl:
            controller.update(
                vehicle_state=vehicle,
                measured_altitude=vehicle.position_y,
                measured_surface_angles=controller.get_surface_angles(),
                dt=ctrl_dt,
                target_position=(config.target_position_x, config.target_position_y),
                guidance_trajectory=guidance_trajectory,
                guidance_mode="trajectory",
            )
            next_ctrl += ctrl_dt

        if enable_control:
            controller.step_actuators(dt)
            surface_angles = controller.get_surface_angles()
        else:
            surface_angles = [0.0] * config.num_control_surfaces

        dynamics.step(dt, surface_angles)

        # record every controller period to keep arrays manageable
        if step % max(1, int(ctrl_dt / dt)) == 0:
            rec["time"].append(t)
            rec["x"].append(vehicle.position_x)
            rec["y"].append(vehicle.position_y)
            rec["vx"].append(vehicle.velocity_x)
            rec["vy"].append(vehicle.velocity_y)
            rec["pitch"].append(vehicle.pitch_angle)
            rec["pitch_rate"].append(vehicle.pitch_rate)
            rec["wind_x"].append(wind_x)
            rec["wind_y"].append(wind_y)
            rec["altitude_error"].append(vehicle.position_y - config.target_position_y)

            if enable_control and controller.is_enabled:
                rec["surface_cmd"].append(
                    math.degrees(controller.commanded_surface_angles[0])
                )
                rec["surface_actual"].append(
                    math.degrees(controller.actuators[0].get_angle())
                )
                rec["pitch_command"].append(controller.commanded_pitch)
                rec["required_correction_deg"].append(controller.required_correction_deg)
                rec["required_correction_fraction"].append(controller.required_correction_fraction)
                rec["required_canard_clock_angle_deg"].append(controller.required_canard_clock_angle_deg)
                rec["current_canard_clock_angle_deg"].append(controller.current_canard_clock_angle_deg)
                rec["flight_path_error_deg"].append(controller.flight_path_error_deg)
                rec["pitch_error_deg"].append(controller.pitch_error_deg)
            else:
                rec["surface_cmd"].append(0.0)
                rec["surface_actual"].append(0.0)
                rec["pitch_command"].append(0.0)
                rec["required_correction_deg"].append(0.0)
                rec["required_correction_fraction"].append(0.0)
                rec["required_canard_clock_angle_deg"].append(90.0)
                rec["current_canard_clock_angle_deg"].append(90.0)
                rec["flight_path_error_deg"].append(0.0)
                rec["pitch_error_deg"].append(0.0)

        # progress report
        if step % report_every == 0 and step > 0:
            print(f"  [{tag}]  t={t:5.2f}s  pos=({vehicle.position_x:6.1f}, "
                  f"{vehicle.position_y:6.1f})  alt_err="
                  f"{vehicle.position_y - config.target_position_y:+6.2f}m")

        t    += dt
        step += 1

    # append final state
    rec["time"].append(t)
    rec["x"].append(vehicle.position_x)
    rec["y"].append(vehicle.position_y)
    rec["vx"].append(vehicle.velocity_x)
    rec["vy"].append(vehicle.velocity_y)
    rec["pitch"].append(vehicle.pitch_angle)
    rec["pitch_rate"].append(vehicle.pitch_rate)
    rec["wind_x"].append(wind_x)
    rec["wind_y"].append(wind_y)
    rec["altitude_error"].append(vehicle.position_y - config.target_position_y)

    # pad fixed-canard arrays so they match length
    pad_len = len(rec["time"]) - len(rec["required_correction_deg"])
    if pad_len > 0:
        for key in ("surface_cmd","surface_actual","pitch_command",
                    "required_correction_deg","required_correction_fraction",
                    "flight_path_error_deg","pitch_error_deg"):
            rec[key].extend([rec[key][-1] if rec[key] else 0.0] * pad_len)
        for key in ("required_canard_clock_angle_deg","current_canard_clock_angle_deg"):
            rec[key].extend([rec[key][-1] if rec[key] else 90.0] * pad_len)

    miss = math.sqrt(
        (vehicle.position_x - config.target_position_x) ** 2
        + (vehicle.position_y - config.target_position_y) ** 2
    )
    rec["miss_distance"] = miss

    print(f"  [{tag}]  DONE  —  final pos ({vehicle.position_x:.2f}, "
          f"{vehicle.position_y:.2f})  miss={miss:.3f}m  steps={step}")
    return rec


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3  —  RUN BALLISTIC (unguided)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[STEP 2/5]  Running BALLISTIC (unguided) scenario …")

veh_b    = _make_vehicle_at_launch(b_sol)
dyn_b    = VehicleDynamics(veh_b)
ctrl_b   = VehicleController()          # controller inactive
wm_b     = WindModel()

ballistic_data = _run_until_target(
    vehicle=veh_b, dynamics=dyn_b, controller=ctrl_b,
    wind_model=wm_b, enable_control=False,
    guidance_trajectory=None, tag="BALLISTIC"
)

print(f"  Ballistic miss distance : {ballistic_data['miss_distance']:.3f} m")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4  —  RUN GUIDED
# ═════════════════════════════════════════════════════════════════════════════
print("\n[STEP 3/5]  Running GUIDED (fixed-canard phase guidance) scenario …")

# Build guidance waypoints: ballistic arc clipped at target
bx_clip, by_clip = pred_bx, pred_by
guidance_traj = [
    (x, y) for x, y in zip(bx_clip, by_clip)
    if x <= config.target_position_x
]
guidance_traj.append((config.target_position_x, config.target_position_y))
print(f"  Guidance trajectory: {len(guidance_traj)} waypoints")

veh_g  = _make_vehicle_at_launch(b_sol)
dyn_g  = VehicleDynamics(veh_g)
ctrl_g = VehicleController()
wm_g   = WindModel()
ctrl_g.enable()

guided_data = _run_until_target(
    vehicle=veh_g, dynamics=dyn_g, controller=ctrl_g,
    wind_model=wm_g, enable_control=True,
    guidance_trajectory=guidance_traj, tag="GUIDED"
)

print(f"  Guided   miss distance  : {guided_data['miss_distance']:.3f} m")
improvement    = ballistic_data["miss_distance"] - guided_data["miss_distance"]
improvement_pc = improvement / ballistic_data["miss_distance"] * 100
print(f"  Accuracy improvement    : {improvement:.2f} m  ({improvement_pc:.1f}%)")

# Quick telemetry sanity-check printout
print("\n  — Telemetry snapshot (first 5 recorded steps) —")
for i in range(min(5, len(guided_data["time"]))):
    print(
        f"    t={guided_data['time'][i]:.3f}s  "
        f"corr={guided_data['required_correction_deg'][i]:+6.2f}°  "
        f"frac={guided_data['required_correction_fraction'][i]:+.3f}  "
        f"clock={guided_data['required_canard_clock_angle_deg'][i]:.1f}°"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5  —  DESIGN TOKENS
# ═════════════════════════════════════════════════════════════════════════════
# Palette: deep-space navy background with cool-blue and amber-gold signal
# colours.  Looks like real avionics / mil-sim software — appropriate for
# the subject.  No warm cream, no terracotta, no acid-green defaults.

BG_DARK   = "#0d1117"   # figure background
BG_PANEL  = "#161b22"   # individual axes background
BG_CARD   = "#21262d"   # annotation boxes
GRID_COL  = "#30363d"   # grid lines

C_BALLISTIC = "#58a6ff"  # cool blue  — ballistic shot
C_GUIDED    = "#ffa657"  # amber gold — guided shot
C_NEUTRAL   = "#8b949e"  # muted grey — neutral / reference
C_POSITIVE  = "#3fb950"  # green      — on-target / positive correction
C_NEGATIVE  = "#f85149"  # red        — off-target / negative correction
C_CLOCK     = "#d2a8ff"  # violet     — canard clock angle
C_MAPPING   = "#f0883e"  # orange-red — mapping consistency check

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold", "color": "#e6edf3"}
FONT_AXIS   = {"fontsize": 9,  "color": "#8b949e"}
FONT_TICK   = {"colors": "#8b949e", "labelsize": 8}
FONT_ANNOT  = {"fontsize": 8.5, "color": "#e6edf3"}

matplotlib.rcParams.update({
    "figure.facecolor":   BG_DARK,
    "axes.facecolor":     BG_PANEL,
    "axes.edgecolor":     GRID_COL,
    "axes.labelcolor":    "#8b949e",
    "xtick.color":        "#8b949e",
    "ytick.color":        "#8b949e",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "grid.color":         GRID_COL,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.6,
    "text.color":         "#e6edf3",
    "legend.facecolor":   BG_CARD,
    "legend.edgecolor":   GRID_COL,
    "legend.labelcolor":  "#e6edf3",
    "legend.fontsize":    8.5,
    "font.family":        "monospace",
})

def _style_ax(ax):
    """Apply shared avionics-style formatting to an axes."""
    ax.set_facecolor(BG_PANEL)
    ax.grid(True, alpha=0.5)
    ax.tick_params(**FONT_TICK)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
        spine.set_linewidth(0.8)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6  —  FIGURE 1 : MISSION OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
print("\n[STEP 4/5]  Rendering Figure 1: Mission Overview …")

fig1 = plt.figure(figsize=(16, 9), facecolor=BG_DARK)
fig1.suptitle(
    "FIXED-CANARD PHASE GUIDANCE  //  MISSION OVERVIEW",
    fontsize=15, fontweight="bold", color="#e6edf3",
    x=0.5, y=0.985
)

gs1 = gridspec.GridSpec(
    3, 2,
    left=0.06, right=0.97,
    top=0.94,  bottom=0.07,
    hspace=0.55, wspace=0.38,
    width_ratios=[1.55, 1],
    height_ratios=[1, 1, 1],
)

ax_traj = fig1.add_subplot(gs1[:, 0])   # full left column
ax_alt  = fig1.add_subplot(gs1[0, 1])
ax_spd  = fig1.add_subplot(gs1[1, 1])
ax_cep  = fig1.add_subplot(gs1[2, 1])

# ── Panel A: Trajectory ───────────────────────────────────────────────────
_style_ax(ax_traj)

# drag-free predicted arc
ax_traj.plot(pred_bx, pred_by,
             color=C_NEUTRAL, linewidth=1.2, linestyle=":",
             alpha=0.6, label="Predicted arc (no drag)")

# actual ballistic
ax_traj.plot(ballistic_data["x"], ballistic_data["y"],
             color=C_BALLISTIC, linewidth=2.2,
             label="Ballistic — unguided")
ax_traj.plot(ballistic_data["x"][-1], ballistic_data["y"][-1],
             marker="s", color=C_BALLISTIC, markersize=9, zorder=5)

# actual guided
ax_traj.plot(guided_data["x"], guided_data["y"],
             color=C_GUIDED, linewidth=2.2,
             label="Guided — fixed-canard phase")
ax_traj.plot(guided_data["x"][-1], guided_data["y"][-1],
             marker="^", color=C_GUIDED, markersize=9, zorder=5)

# target
tx, ty = config.target_position_x, config.target_position_y
ax_traj.plot(tx, ty, marker="*", color=C_POSITIVE, markersize=18, zorder=6,
             label="Target")
cep_circle = mpatches.Circle(
    (tx, ty), config.cep_radius,
    fill=False, edgecolor=C_POSITIVE, linewidth=1.4,
    linestyle="--", alpha=0.6, label=f"CEP {config.cep_radius}m radius"
)
ax_traj.add_patch(cep_circle)

# miss-distance annotations
b_ix, b_iy = ballistic_data["x"][-1], ballistic_data["y"][-1]
g_ix, g_iy = guided_data["x"][-1],   guided_data["y"][-1]

ax_traj.annotate(
    f"  MISS  {ballistic_data['miss_distance']:.1f}m",
    xy=(b_ix, b_iy), xytext=(b_ix - 60, b_iy + 8),
    color=C_BALLISTIC, fontsize=8.5, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_BALLISTIC, lw=1.2),
)
ax_traj.annotate(
    f"  HIT  {guided_data['miss_distance']:.2f}m",
    xy=(g_ix, g_iy), xytext=(g_ix - 60, g_iy - 14),
    color=C_GUIDED, fontsize=8.5, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_GUIDED, lw=1.2),
)

ax_traj.set_xlabel("Horizontal distance (m)", **FONT_AXIS)
ax_traj.set_ylabel("Altitude (m)", **FONT_AXIS)
ax_traj.set_title("TRAJECTORY COMPARISON", **FONT_TITLE)
ax_traj.legend(loc="upper left", framealpha=0.85)

# stat box
stat_text = (
    f"  Target    ({tx:.0f}m, {ty:.0f}m)\n"
    f"  Ballistic  {ballistic_data['miss_distance']:.1f}m miss\n"
    f"  Guided     {guided_data['miss_distance']:.2f}m miss\n"
    f"  Improvement  {improvement_pc:.1f}%"
)
ax_traj.text(
    0.02, 0.03, stat_text,
    transform=ax_traj.transAxes,
    fontsize=8.5, verticalalignment="bottom",
    color="#e6edf3",
    bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_CARD,
              edgecolor=GRID_COL, alpha=0.92),
    family="monospace",
)

# ── Panel B: Altitude error ───────────────────────────────────────────────
_style_ax(ax_alt)
t_b  = np.array(ballistic_data["time"])
t_g  = np.array(guided_data["time"])
ae_b = np.array(ballistic_data["altitude_error"])
ae_g = np.array(guided_data["altitude_error"])

# Common time axis for fill (guided is shorter — interpolate ballistic onto it)
ae_b_on_tg = np.interp(t_g, t_b, ae_b)

ax_alt.plot(t_b, ae_b, color=C_BALLISTIC, linewidth=1.8, label="Ballistic")
ax_alt.plot(t_g, ae_g, color=C_GUIDED,    linewidth=1.8, label="Guided")
ax_alt.axhline(0, color=C_POSITIVE, linewidth=0.9, linestyle="--", alpha=0.7)
# Shade only where guided is better (closer to zero) than ballistic
ax_alt.fill_between(t_g, ae_g, ae_b_on_tg,
                    where=(np.abs(ae_g) < np.abs(ae_b_on_tg)),
                    color=C_GUIDED, alpha=0.18, label="Guided advantage")
ax_alt.set_xlabel("Time (s)", **FONT_AXIS)
ax_alt.set_ylabel("Altitude error (m)", **FONT_AXIS)
ax_alt.set_title("ALTITUDE ERROR CONVERGENCE", **FONT_TITLE)
ax_alt.legend(loc="upper right")

# ── Panel C: Airspeed ─────────────────────────────────────────────────────
_style_ax(ax_spd)
spd_b = np.sqrt(np.array(ballistic_data["vx"])**2 + np.array(ballistic_data["vy"])**2)
spd_g = np.sqrt(np.array(guided_data["vx"])**2   + np.array(guided_data["vy"])**2)

ax_spd.plot(t_b, spd_b, color=C_BALLISTIC, linewidth=1.8, label="Ballistic")
ax_spd.plot(t_g, spd_g, color=C_GUIDED,    linewidth=1.8, label="Guided")
ax_spd.axhline(config.reference_velocity_x,
               color=C_NEUTRAL, linewidth=0.9, linestyle="--", alpha=0.7,
               label=f"Ref {config.reference_velocity_x}m/s")
ax_spd.set_xlabel("Time (s)", **FONT_AXIS)
ax_spd.set_ylabel("Airspeed (m/s)", **FONT_AXIS)
ax_spd.set_title("AIRSPEED", **FONT_TITLE)
ax_spd.legend(loc="lower left")

# ── Panel D: CEP close-up ────────────────────────────────────────────────
_style_ax(ax_cep)
# Zoom so both impacts AND the CEP circle are visible with a small margin
_all_x = [b_ix, g_ix, tx]
_all_y = [b_iy, g_iy, ty]
_spread = max(
    max(_all_x) - min(_all_x),
    max(_all_y) - min(_all_y),
    config.cep_radius * 2,
)
zoom = _spread * 0.75 + 15          # tight but never smaller than CEP + margin
ax_cep.set_xlim(tx - zoom, tx + zoom)
ax_cep.set_ylim(ty - zoom, ty + zoom)
ax_cep.set_aspect("equal")

cep2 = mpatches.Circle(
    (tx, ty), config.cep_radius,
    fill=False, edgecolor=C_POSITIVE, linewidth=1.6,
    linestyle="--", alpha=0.9
)
ax_cep.add_patch(cep2)
ax_cep.plot(tx, ty, "*", color=C_POSITIVE, markersize=22, zorder=6)

ax_cep.plot(b_ix, b_iy, "s", color=C_BALLISTIC, markersize=12, zorder=5,
            label=f"Ballistic  ({b_ix:.0f}, {b_iy:.0f})")
ax_cep.annotate("",
    xy=(b_ix, b_iy), xytext=(tx, ty),
    arrowprops=dict(arrowstyle="-|>", color=C_BALLISTIC,
                    lw=1.5, mutation_scale=14),
)

ax_cep.plot(g_ix, g_iy, "^", color=C_GUIDED, markersize=12, zorder=5,
            label=f"Guided  ({g_ix:.0f}, {g_iy:.0f})")
ax_cep.annotate("",
    xy=(g_ix, g_iy), xytext=(tx, ty),
    arrowprops=dict(arrowstyle="-|>", color=C_GUIDED,
                    lw=1.5, mutation_scale=14),
)

ax_cep.set_xlabel("Horizontal distance (m)", **FONT_AXIS)
ax_cep.set_ylabel("Altitude (m)", **FONT_AXIS)
ax_cep.set_title("CEP TARGET VIEW", **FONT_TITLE)
ax_cep.legend(loc="lower right", fontsize=7.5)

print("  ✓  Figure 1 built")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7  —  FIGURE 2 : FIXED-CANARD PHASE GUIDANCE TELEMETRY
# ═════════════════════════════════════════════════════════════════════════════
print("\n[STEP 5/5]  Rendering Figure 2: Fixed-Canard Phase Guidance Telemetry …")

t_g  = np.array(guided_data["time"])
corr = np.array(guided_data["required_correction_deg"])
frac = np.array(guided_data["required_correction_fraction"])
clk  = np.array(guided_data["required_canard_clock_angle_deg"])
cur  = np.array(guided_data["current_canard_clock_angle_deg"])
fpa  = np.array(guided_data["flight_path_error_deg"])
pe   = np.array(guided_data["pitch_error_deg"])
mapp = np.cos(np.radians(clk))   # must equal frac

print(f"  Correction range   : [{corr.min():.2f}, {corr.max():.2f}] °")
print(f"  Fraction range     : [{frac.min():.3f}, {frac.max():.3f}]")
print(f"  Clock angle range  : [{clk.min():.1f}, {clk.max():.1f}] °")
print(f"  Mapping max error  : {np.max(np.abs(mapp - frac)):.2e}  (should be ~0)")

fig2 = plt.figure(figsize=(14, 14), facecolor=BG_DARK)
fig2.suptitle(
    "FIXED-CANARD PHASE GUIDANCE  //  CANARD ORIENTATION TELEMETRY",
    fontsize=14, fontweight="bold", color="#e6edf3",
    x=0.5, y=0.99
)

gs2 = gridspec.GridSpec(
    4, 1,
    left=0.10, right=0.95,
    top=0.955, bottom=0.06,
    hspace=0.70,
)

ax_corr = fig2.add_subplot(gs2[0])
ax_clk  = fig2.add_subplot(gs2[1], sharex=ax_corr)
ax_norm = fig2.add_subplot(gs2[2], sharex=ax_corr)
ax_map  = fig2.add_subplot(gs2[3], sharex=ax_corr)

for ax in (ax_corr, ax_clk, ax_norm, ax_map):
    _style_ax(ax)

# ── Row 1: Required aerodynamic correction ────────────────────────────────
ax_corr.plot(t_g, corr, color="#e6edf3", linewidth=1.8, zorder=3)
ax_corr.fill_between(t_g, corr, 0,
                     where=(corr >= 0), color=C_POSITIVE, alpha=0.30, zorder=2)
ax_corr.fill_between(t_g, corr, 0,
                     where=(corr < 0),  color=C_NEGATIVE, alpha=0.30, zorder=2)
ax_corr.axhline(0, color=C_NEUTRAL, linewidth=0.8, linestyle="-")

# Draw authority limits — and set y-range to give signal room to breathe
_corr_abs_max = max(abs(corr.max()), abs(corr.min()))
_corr_ylim    = max(_corr_abs_max * 1.6, 0.5)   # never collapse to zero height
ax_corr.set_ylim(-_corr_ylim, _corr_ylim)

# Authority limit lines (dotted at ±12°, outside the autoscale range is fine)
for _lim in (config.pitch_controller_output_limit, -config.pitch_controller_output_limit):
    ax_corr.axhline(_lim, color=C_NEUTRAL, linewidth=0.6, linestyle=":", alpha=0.45)
    ax_corr.text(t_g[-1] * 0.99, _lim,
                 f" ±{abs(_lim):.0f}° limit",
                 va="center", ha="right", color=C_NEUTRAL, fontsize=7,
                 family="monospace")

ax_corr.set_ylabel("Correction (°)", **FONT_AXIS)
ax_corr.set_title(
    f"REQUIRED AERODYNAMIC CORRECTION  —  PD output  "
    f"(authority ±{config.pitch_controller_output_limit:.0f}°, "
    f"peak used ±{_corr_abs_max:.1f}°)",
    **FONT_TITLE
)
ax_corr.text(0.99, 0.97,
    f"max {corr.max():+.2f}°     min {corr.min():+.2f}°",
    transform=ax_corr.transAxes, ha="right", va="top",
    color=C_NEUTRAL, fontsize=8, family="monospace",
)

# ── Row 2: Canard clock orientation ──────────────────────────────────────
# Shade by correction polarity
ax_clk.fill_between(t_g, 90, clk,
                    where=(clk < 90), color=C_POSITIVE, alpha=0.15, zorder=1)
ax_clk.fill_between(t_g, 90, clk,
                    where=(clk > 90), color=C_NEGATIVE, alpha=0.15, zorder=1)

ax_clk.plot(t_g, clk, color=C_CLOCK, linewidth=2.0,
            label="Required orientation", zorder=3)
ax_clk.plot(t_g, cur, color=C_MAPPING, linewidth=1.2, linestyle="--",
            alpha=0.75, label="Current (instantaneous)", zorder=3)
ax_clk.axhline(90,  color=C_POSITIVE, linewidth=1.0, linestyle="--",
               alpha=0.8, label="90° — neutral (zero correction)")
ax_clk.axhline(0,   color=C_NEUTRAL,  linewidth=0.5, linestyle=":", alpha=0.4)
ax_clk.axhline(180, color=C_NEUTRAL,  linewidth=0.5, linestyle=":", alpha=0.4)

# Y-range: tight around actual signal with enough room to see deviation from 90°
_clk_lo = min(clk.min(), 90) - 8
_clk_hi = max(clk.max(), 90) + 8
ax_clk.set_ylim(_clk_lo, _clk_hi)

# Only draw y-ticks that fall within the visible range
_clk_ticks     = [t for t in [0, 45, 90, 135, 180] if _clk_lo <= t <= _clk_hi]
_clk_tick_lbls = {0: "0°\nmax +ve", 45: "45°", 90: "90°\nneutral",
                  135: "135°", 180: "180°\nmax −ve"}
ax_clk.set_yticks(_clk_ticks)
ax_clk.set_yticklabels([_clk_tick_lbls[t] for t in _clk_ticks],
                        fontsize=7.5, color="#8b949e")
ax_clk.set_ylabel("Clock angle (°)", **FONT_AXIS)
ax_clk.set_title(
    "FIXED-CANARD CLOCK ORIENTATION  —  Required vs Current",
    **FONT_TITLE
)
ax_clk.legend(loc="upper right", framealpha=0.85)

# assumption callout — top centre so it never overlaps y-axis labels
ax_clk.text(0.50, 0.97,
    "Assumption: phase alignment is instantaneous  →  required ≡ current",
    transform=ax_clk.transAxes, ha="center", va="top",
    color=C_NEUTRAL, fontsize=7.5, style="italic", family="monospace",
    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_CARD,
              edgecolor=GRID_COL, alpha=0.8),
)

# ── Row 3: Normalised correction demand ──────────────────────────────────
ax_norm.plot(t_g, frac, color="#e6edf3", linewidth=1.8, zorder=3)
ax_norm.fill_between(t_g, frac, 0,
                     where=(frac >= 0), color=C_POSITIVE, alpha=0.25)
ax_norm.fill_between(t_g, frac, 0,
                     where=(frac <  0), color=C_NEGATIVE, alpha=0.25)
ax_norm.axhline( 1.0, color=C_POSITIVE, linewidth=0.7, linestyle=":", alpha=0.6)
ax_norm.axhline(-1.0, color=C_NEGATIVE, linewidth=0.7, linestyle=":", alpha=0.6)
ax_norm.axhline(0,    color=C_NEUTRAL,  linewidth=0.8, linestyle="-")
ax_norm.set_ylim(-1.35, 1.35)
ax_norm.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax_norm.set_ylabel("Fraction (−1…+1)", **FONT_AXIS)
ax_norm.set_title(
    "NORMALISED CORRECTION DEMAND  —  fraction of max aerodynamic authority",
    **FONT_TITLE
)
ax_norm.text(0.98, 0.95,
    f"peak demand  {np.max(np.abs(frac)):.3f}",
    transform=ax_norm.transAxes, ha="right", va="top",
    color=C_NEUTRAL, fontsize=8, family="monospace",
)

# ── Row 4: Mapping consistency ────────────────────────────────────────────
ax_map.plot(t_g, frac, color=C_CLOCK, linewidth=2.2,
            label="Required correction fraction", zorder=3)
ax_map.plot(t_g, mapp, color=C_MAPPING, linewidth=1.4, linestyle="--",
            alpha=0.85, label="cos(required clock angle)", zorder=4)

# residual (should be ~1e-15 machine epsilon)
residual = np.abs(mapp - frac)
ax_map_r = ax_map.twinx()
ax_map_r.plot(t_g, residual, color="#8b949e", linewidth=0.8,
              linestyle=":", alpha=0.5, label="Residual")
ax_map_r.set_ylim(0, max(residual.max() * 5, 1e-10))
ax_map_r.set_ylabel("Residual", fontsize=7.5, color="#8b949e")
ax_map_r.tick_params(axis="y", colors="#8b949e", labelsize=7)
ax_map_r.spines["right"].set_edgecolor(GRID_COL)

ax_map.axhline(0, color=C_NEUTRAL, linewidth=0.8, linestyle="-")
ax_map.set_ylim(-1.35, 1.35)
ax_map.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax_map.set_xlabel("Time (s)", **FONT_AXIS)
ax_map.set_ylabel("Value", **FONT_AXIS)
ax_map.set_title(
    "COMMAND-TO-ORIENTATION MAPPING CONSISTENCY  —  fraction = cos(clock angle)",
    **FONT_TITLE
)
lines1, labels1 = ax_map.get_legend_handles_labels()
lines2, labels2 = ax_map_r.get_legend_handles_labels()
ax_map.legend(lines1 + lines2, labels1 + labels2,
              loc="upper right", framealpha=0.85)

ax_map.text(0.01, 0.05,
    "These curves are mathematically identical by construction.\n"
    "Overlap confirms the acos mapping is applied correctly.",
    transform=ax_map.transAxes, ha="left", va="bottom",
    color=C_NEUTRAL, fontsize=7.5, style="italic", family="monospace",
    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_CARD,
              edgecolor=GRID_COL, alpha=0.8),
)

plt.setp(ax_corr.get_xticklabels(), visible=False)
plt.setp(ax_clk.get_xticklabels(),  visible=False)
plt.setp(ax_norm.get_xticklabels(), visible=False)

print("  ✓  Figure 2 built")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8  —  SAVE & SHOW
# ═════════════════════════════════════════════════════════════════════════════
print("\n  Saving figures …")
fig1.savefig("dashboard_mission_overview.png",
             dpi=150, facecolor=BG_DARK, bbox_inches="tight")
fig2.savefig("dashboard_canard_telemetry.png",
             dpi=150, facecolor=BG_DARK, bbox_inches="tight")
print("  ✓  dashboard_mission_overview.png")
print("  ✓  dashboard_canard_telemetry.png")

print("\n" + "=" * 70)
print(f"  BALLISTIC miss distance :  {ballistic_data['miss_distance']:.2f} m")
print(f"  GUIDED   miss distance  :  {guided_data['miss_distance']:.4f} m")
print(f"  Accuracy improvement    :  {improvement:.2f} m  ({improvement_pc:.1f}%)")
print("=" * 70)

plt.show()