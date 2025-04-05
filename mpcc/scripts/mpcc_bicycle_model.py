
import casadi as ca
import numpy as np
from acados_template import AcadosModel

import yaml

def export_bicycle_model(n_control_points):
    # model name
    model_name = "bicycle_model"

    with open("params.yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    print(params)

    # model variables

    R = params.get("wheel_radius", 0.2032)
    L = params.get("wheelbase", 1.5)
    # R = rospy.get_param("~wheel_radius", 0.2032)
    # L = rospy.get_param("~wheelbase", 1.5)

    # model states
    posx = ca.MX.sym("posx")  # x position
    posy = ca.MX.sym("posy")  # y position
    theta = ca.MX.sym("theta")  # orientation
    delta = ca.MX.sym("delta")  # steering angle
    v = ca.MX.sym("v")  # velocity
    tau = ca.MX.sym("tau")  # arclength progress
    x = ca.vertcat(posx, posy, theta, delta, v, tau)

    # model inputs
    alpha = ca.MX.sym("alpha")  # acceleration
    phi = ca.MX.sym("phi")  # steering angle rate
    zeta = ca.MX.sym("zeta")  # arclength rate
    u = ca.vertcat(alpha, phi, zeta)

    # model parameters
    control_points = ca.MX.sym("control_points", 2 * n_control_points, 1)

    # model dynamics
    dx = v * ca.cos(theta)
    dy = v * ca.sin(theta)
    omega = v * ca.tan(delta) / L
    dv = alpha * R
    dtau = zeta
    f_expl = ca.vertcat(dx, dy, omega, phi, dv, dtau)

    # create model
    model = AcadosModel()
    model.p = control_points
    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.name = model_name

    # define spline functions
    degree = 2
    kk = np.linspace(0, 1, n_control_points - degree + 1)
    knots = [
        [
            float(i)
            for i in np.concatenate(
                [np.ones(degree) * kk[0], kk, np.ones(degree) * kk[-1]]
            )
        ]
    ]
    c_points = ca.MX.sym("c_points", 2, n_control_points)
    spline = ca.bspline(tau, c_points, knots, [degree], 2, {})
    spline_function = ca.Function("spline", [tau, c_points], [spline])
    spline_derivative = ca.jacobian(spline, tau)
    spline_derivative_function = ca.Function(
        "spline_derivative", [tau, c_points], [spline_derivative]
    )

    # cost contributions
    ql = params.get("ql", 1.0)
    qc = params.get("qc", 0.02)
    ra = params.get("ra", 0.00001)
    rs = params.get("rs", 0.000000001)
    rz = params.get("rz", 5.0)

    control_points_spline = ca.reshape(control_points, 2, n_control_points).T
    tau_mod = ca.fmod(tau, 1.0)
    phi = ca.atan2(
        (spline_derivative_function(tau_mod, control_points_spline.T)[1] + 1e-6),
        spline_derivative_function(tau_mod, control_points_spline.T)[0] + 1e-6,
    )
    ec = ca.sin(phi) * (
        x[0] - spline_function(tau_mod, control_points_spline.T)[0]
    ) - ca.cos(phi) * (x[1] - spline_function(tau_mod, control_points_spline.T)[1])
    el = -ca.cos(phi) * (
        x[0] - spline_function(tau_mod, control_points_spline.T)[0]
    ) - ca.sin(phi) * (x[1] - spline_function(tau_mod, control_points_spline.T)[1])

    model.cost_expr_ext_cost = (
        qc * ec**2 + ql * el**2 + ra * alpha**2 + rs * phi**2 - rz * zeta
    )
    model.cost_expr_ext_cost_e = 0.01 * qc * ec**2 + 0.1 * ql * el**2

    # constraints
    # half_track_width = rospy.get_param("~constraint_circle_radius", 0.8)
    half_track_width = params.get("constraint_circle_radius", 0.8)
    model.con_h_expr = ec**2 + el**2 - half_track_width**2
    model.con_h_expr_e = ec**2 + el**2 - half_track_width**2

    return model, spline_function
