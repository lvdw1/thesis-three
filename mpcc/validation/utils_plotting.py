import casadi as ca
import matplotlib.pyplot as plt
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
    import rospy

    rospy.logerr(f"REF TRACK: {ref_track}")
    spline_function = create_spline_function(np.vstack((ref_track, ref_track[0, :])))
    prev_point = [[ref_track[0, 0], ref_track[0, 1]]]
    ref_track = []
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


# plotting
def plot_result_xy(complete_track, track, x, x_guess, x_vars, N, circle_radius):
    ec, el, spline, dspline = create_error_function(track, with_splines=True)
    fig = plt.figure()
    plt.plot(
        x[:, x_vars.index("x")], x[:, x_vars.index("y")], "o-", label="MPCC prediction"
    )
    plt.plot(
        np.append(complete_track[:, 0], complete_track[0, 0]),
        np.append(complete_track[:, 1], complete_track[0, 1]),
        "--",
        label="Centerline",
    )

    # comment out the following line to unplot/plot the initial guess
    # plt.plot(x_guess[:,0], x_guess[:,1], 'o-', color='orange', label="Initial Guess")

    # plot initial position
    plt.plot(x_guess[0, 0], x_guess[0, 1], "ro", label="Current position")
    # plot boundaries
    plot_track_boundaries(complete_track, 1.5, lwidth=0.5)

    # plot dotted lines connecting the states (X,Y) with the spline points spline(tau)
    for i in range(N + 1):
        # Spline point at current tau
        spline_xy = spline(x[i, x_vars.index("tau")]).full().flatten()

        # Draw line connecting state to spline
        plt.plot(
            [x[i, x_vars.index("x")], spline_xy[0]],
            [x[i, x_vars.index("y")], spline_xy[1]],
            "k--",
            linewidth=0.5,
        )

        # draw constraint circle
        circle = plt.Circle(
            (spline_xy[0], spline_xy[1]), circle_radius, color="r", fill=False
        )
        plt.gca().add_artist(circle)

        # Compute errors
        ec_val = (
            ec(
                x[i, x_vars.index("x")],
                x[i, x_vars.index("y")],
                x[i, x_vars.index("tau")],
            )
            .full()
            .flatten()
        )
        el_val = (
            el(
                x[i, x_vars.index("x")],
                x[i, x_vars.index("y")],
                x[i, x_vars.index("tau")],
            )
            .full()
            .flatten()
        )

        # Compute spline tangent direction
        spline_der = dspline(x[i, x_vars.index("tau")]).full().flatten()
        tangent_dir = spline_der / (
            np.linalg.norm(spline_der) + 1e-6
        )  # Normalize tangent

        # Lateral error vector
        lateral_vector = np.array([-tangent_dir[1], tangent_dir[0]]) * ec_val

        # Longitudinal error vector
        longitudinal_vector = tangent_dir * el_val

        # Add arrows for longitudinal and lateral errors
        plt.arrow(
            x[i, x_vars.index("x")],
            x[i, x_vars.index("y")],
            lateral_vector[0],
            lateral_vector[1],
            color="blue",
            head_width=0.05,
            length_includes_head=True,
            label="Lateral Error" if i == 0 else "",
        )
        plt.arrow(
            x[i, x_vars.index("x")],
            x[i, x_vars.index("y")],
            longitudinal_vector[0],
            longitudinal_vector[1],
            color="green",
            head_width=0.05,
            length_includes_head=True,
            label="Longitudinal Error" if i == 0 else "",
        )

    plt.axis("equal")
    plt.legend()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("XY plot of one MPCC iteration")

    return fig


def create_error_function(ref_track, with_splines=False):
    spline, dspline = create_spline_function(ref_track, with_derivative=True)
    xpos = ca.MX.sym("xpos")
    ypos = ca.MX.sym("ypos")
    tau = ca.MX.sym("tau")
    phi = ca.atan2((dspline(tau)[1] + 1e-6), dspline(tau)[0] + 1e-6)
    ec = ca.sin(phi) * (xpos - spline(tau)[0]) - ca.cos(phi) * (ypos - spline(tau)[1])
    el = -ca.cos(phi) * (xpos - spline(tau)[0]) - ca.sin(phi) * (ypos - spline(tau)[1])
    ec_function = ca.Function("ec", [xpos, ypos, tau], [ec])
    el_function = ca.Function("el", [xpos, ypos, tau], [el])
    if with_splines:
        return ec_function, el_function, spline, dspline
    return ec_function, el_function


def plot_track_boundaries(track, width, closed_track=True, style="r--", lwidth=0.0):
    left_bp_prev, right_bp_prev = compute_boundary_points(track, 0, width)
    for i in range(1, track.shape[0]):
        left_bp, right_bp = compute_boundary_points(track, i, width)
        plt.plot(
            [left_bp_prev[0], left_bp[0]],
            [left_bp_prev[1], left_bp[1]],
            style,
            linewidth=lwidth,
        )
        plt.plot(
            [right_bp_prev[0], right_bp[0]],
            [right_bp_prev[1], right_bp[1]],
            style,
            linewidth=lwidth,
        )
        left_bp_prev, right_bp_prev = left_bp, right_bp
    if closed_track:
        left_bp_0, right_bp_0 = compute_boundary_points(track, 0, width)
        plt.plot(
            [left_bp_prev[0], left_bp_0[0]],
            [left_bp_prev[1], left_bp_0[1]],
            style,
            linewidth=lwidth,
        )
        plt.plot(
            [right_bp_prev[0], right_bp_0[0]],
            [right_bp_prev[1], right_bp_0[1]],
            style,
            linewidth=lwidth,
        )


def compute_boundary_points(track, idx, width):
    # compute normal vector
    normal = track[idx, :2] - track[(idx + 1) % track.shape[0], :2]
    normal = np.array([-normal[1], normal[0]])
    normal = normal / np.linalg.norm(normal) * width
    # compute left and right boundary points
    left_bp, right_bp = track[idx, :2] - normal, track[idx, :2] + normal
    return left_bp, right_bp
