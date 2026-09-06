"""
PID (Proportional-Integral-Derivative) controller implementation.

Implements a standard PID controller with anti-windup protection.
"""

import config


class PIDController:
    """
    A standard PID controller.

    Implements: output = Kp*error + Ki*integral(error) + Kd*d(error)/dt
    """

    def __init__(self, kp, ki, kd, output_limit=None, integral_limit=None):
        """
        Initialize PID controller.

        Args:
            kp: proportional gain
            ki: integral gain
            kd: derivative gain
            output_limit: maximum magnitude of output (optional)
            integral_limit: maximum magnitude of integral term (anti-windup)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.output_limit = output_limit
        self.integral_limit = integral_limit if integral_limit is not None else 1e10

        # State
        self.integral_term = 0.0
        self.previous_error = 0.0
        self.initialized = False

    def reset(self):
        """Reset controller state."""
        self.integral_term = 0.0
        self.previous_error = 0.0
        self.initialized = False

    def update(self, error, dt):
        """
        Calculate PID output based on error.

        Args:
            error: current error (setpoint - measured value)
            dt: timestep in seconds

        Returns:
            output: PID controller output
        """
        if not self.initialized:
            self.previous_error = error
            self.initialized = True

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral_term += self.ki * error * dt
        if self.integral_limit:
            self.integral_term = max(-self.integral_limit, min(self.integral_limit, self.integral_term))
        i_term = self.integral_term

        # Derivative term
        if dt > 0:
            error_derivative = (error - self.previous_error) / dt
        else:
            error_derivative = 0.0
        d_term = self.kd * error_derivative

        # Total output
        output = p_term + i_term + d_term

        # Limit output
        if self.output_limit:
            output = max(-self.output_limit, min(self.output_limit, output))

        # Store for next iteration
        self.previous_error = error

        return output