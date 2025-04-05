import casadi as ca
import numpy as np

def initial_guess(reference_track, N, dt, alpha_max, v_max, wheel_radius, wheelbase):
    tau = ca.MX.sym("tau")
    degree = 2
    n_control_points = reference_track.shape[0]
    kk = np.linspace(0, 1, n_control_points - degree + 1)
    knots = [
        [
            float(i)
            for i in np.concatenate(
                [np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]]
            )
        ]
    ]
    spline = ca.bspline(tau, reference_track.T, knots, [degree], 2, {})
    spline_function = ca.Function("spline", [tau], [spline])
    spline_derivative = ca.jacobian(spline, tau)
    spline_derivative_function = ca.Function(
        "spline_derivative", [tau], [spline_derivative]
    )
    Tau0 = np.zeros(N + 1)
    velocity = np.zeros(N + 1)
    acceleration = np.zeros(N)
    x, y = np.zeros(N + 1), np.zeros(N + 1)
    theta = np.zeros(N + 1)
    delta = np.zeros(N + 1)
    ddelta = np.zeros(N)
    zeta = np.zeros(N)
    Tau0[0] = 0.00001
    v0 = 0.0
    a_max = alpha_max * wheel_radius
    velocity[0] = v0
    x[0], y[0] = spline_function(Tau0[0]).full().flatten()
    dx0, dy0 = spline_derivative_function(Tau0[0]).full().flatten()
    theta[0] = ca.arctan2(dy0, dx0)

    for i in range(1, N + 1):
        x0 = spline_function(Tau0[i - 1]).full().flatten()
        tau = Tau0[i - 1]
        x1 = x0
        while np.linalg.norm(x1 - x0) < min(v0 * dt + a_max * dt**2 / 2, v_max * dt):
            tau += 0.0001
            x1 = spline_function(tau).full().flatten()
        Tau0[i] = tau
        v0 = min(v0 + a_max * dt, v_max)
        velocity[i] = v0
        if i < N:
            acceleration[i] = min(a_max, (v_max - v0) / dt)
        x[i], y[i] = x1
        # heading
        der_points = np.array(spline_derivative_function(Tau0[i]).full().flatten())
        prev_heading = theta[i - 1]
        phi = ca.arctan2(der_points[1], der_points[0])
        heading = phi
        diff_with_prev = heading - prev_heading
        diff_with_prev = ca.fmod(diff_with_prev + np.pi, 2 * np.pi) - np.pi
        theta[i] = prev_heading + diff_with_prev
        # steering angle -> choice to take 'i-1', then it is forwards differences to integrate omega.
        delta[i - 1] = ca.atan2(diff_with_prev * wheelbase, dt * velocity[i])
        # zeta
        zeta[i - 1] = (Tau0[i] - Tau0[i - 1]) / dt
    acceleration[0] = acceleration[1]
    delta[-1] = delta[-2]
    for i in range(N):
        ddelta[i] = (delta[i + 1] - delta[i]) / dt
    X0 = np.array([x, y, theta, delta, velocity, Tau0]).T
    U0 = np.array([acceleration / wheel_radius, ddelta, zeta]).T
    return X0, U0
