class LongitudinalPID:
    """PID speed controller with output saturation and anti-windup."""

    def __init__(
        self,
        kp,
        ki,
        kd,
        dt,
        output_limits=(-1.0, 1.0),
        integral_limits=(-2.0, 2.0),
        derivative_filter=0.2,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        if not 0.0 <= derivative_filter <= 1.0:
            raise ValueError(
                "derivative_filter must be between 0 and 1"
            )

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.output_min, self.output_max = output_limits
        self.integral_min, self.integral_max = integral_limits
        self.derivative_filter = derivative_filter

        self.integral = 0.0
        self.previous_error = None
        self.filtered_derivative = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = None
        self.filtered_derivative = 0.0

    def update(self, target_speed, current_speed):
        error = target_speed - current_speed

        if self.previous_error is None:
            raw_derivative = 0.0
        else:
            raw_derivative = (
                error - self.previous_error
            ) / self.dt

        alpha = self.derivative_filter
        self.filtered_derivative = (
            alpha * raw_derivative
            + (1.0 - alpha) * self.filtered_derivative
        )

        candidate_integral = self.integral + error * self.dt
        candidate_integral = max(
            self.integral_min,
            min(candidate_integral, self.integral_max),
        )

        unsaturated_output = (
            self.kp * error
            + self.ki * candidate_integral
            + self.kd * self.filtered_derivative
        )

        output = max(
            self.output_min,
            min(unsaturated_output, self.output_max),
        )

        # Conditional integration anti-windup:
        # reject integration if it would push farther into saturation.
        saturated_high = (
            unsaturated_output > self.output_max and error > 0.0
        )
        saturated_low = (
            unsaturated_output < self.output_min and error < 0.0
        )

        if not saturated_high and not saturated_low:
            self.integral = candidate_integral

        self.previous_error = error

        return output, error