import casadi as ca
import numpy as np


def create_spline_function(ref_track, with_derivative=False, degree=2):
    degree = 2
    n_control_points = ref_track.shape[0]
    kk = np.linspace(0, 1, n_control_points - degree + 1)
    knots = [
        [
            float(i)
            for i in np.concatenate(
                [np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]]
            )
        ]
    ]
    tau = ca.MX.sym("tau")
    spline = ca.bspline(tau, ref_track[:, :2].T, knots, [degree], 2, {})
    spline_function = ca.Function("spline", [tau], [spline])
    if with_derivative:
        spline_derivative = ca.jacobian(spline, tau)
        spline_derivative_function = ca.Function(
            "spline_derivative", [tau], [spline_derivative]
        )
        return spline_function, spline_derivative_function
    return spline_function


def sample_equidistant_track_points(ref_track, distance):
    spline_function = create_spline_function(np.vstack((ref_track, ref_track[0, :])))
    prev_point = [[ref_track[0, 0], ref_track[0, 1]]]
    ref_track = [spline_function(0.0).full().flatten()]
    tau = 0.0
    dist = 0.0
    while tau < 1.0:
        point = spline_function(tau).full().flatten()
        dist += np.linalg.norm(point - prev_point)
        if dist > distance:
            ref_track.append(spline_function(tau).full().flatten())
            dist = 0.0
        prev_point = point
        tau += 0.00001
    return np.array(ref_track)
