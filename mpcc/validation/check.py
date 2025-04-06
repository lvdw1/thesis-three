import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path as FilePath
import os
import re
import casadi as ca

# Data path
data_path = FilePath(__file__).parent / "06-04-2025__12-03"
print(data_path)
# Find the largest iteration number from filenames
n_iter = max((int(re.search(r'iteration_(\d+)', file).group(1)) for file in os.listdir(data_path / "iterations") if re.search(r'iteration_(\d+)', file)), default=None)

reference_track = np.load(data_path / "reference_track.npy")

steering_inputs = np.zeros(n_iter+1)
acceleration_inputs = np.zeros(n_iter+1)
actual_position = np.zeros((n_iter+1, 2))
actual_speed = np.zeros(n_iter+1)
actual_heading = np.zeros(n_iter+1)
ocp_solve_times = np.zeros(n_iter+1)
solve_times = np.zeros(n_iter+1)
solve_time_ratios = np.zeros(n_iter+1)
diff_actual_vel_vs_init_guess = np.zeros(n_iter+1)
complete_closedloop_times = np.zeros(n_iter+1)


actual_x = np.zeros((n_iter+1,6))
actual_u = np.zeros((n_iter+1,3))
for i in range(0,n_iter-5,1):
    # Load data
    data = np.load(data_path / "iterations" / f"iteration_{i}.npz")
    x_guess = data["x_guess"]
    u_guess = data["u_guess"]
    x_sol = data["x_sol"]
    u_sol = data["u_sol"]
    res = data["res"]
    x0 = data["x0"]
    # ocp_solve_time = data["ocp_solve_time"]
    # solve_time = data["solve_time"]
    # complete_closedloop_time = data["complete_closedloop_time"]


    steering_inputs[i] = u_sol[0, 1]
    acceleration_inputs[i] = u_sol[0, 0]
    actual_position[i] = x_sol[0, 0:2]
    actual_heading[i] = x_sol[0, 2]
    # ocp_solve_times[i] = ocp_solve_time
#     solve_times[i] = solve_time
    # solve_time_ratios[i] = ocp_solve_time / solve_time
    actual_speed[i] = x_sol[0, 4]
    # complete_closedloop_times[i] = complete_closedloop_time


#     # Calculate the difference between the actual speed and the initial guess speed
#     diff_actual_vel_vs_init_guess[i] = x_sol[0, 4] - x_guess[0, 4]

#     # store the actual x and u
    actual_x[i, :] = x_sol[0]
    actual_u[i, :] = u_sol[0]

#     # # xy plot
#     # fig = plt.figure()
#     # fig.canvas.manager.full_screen_toggle()
#     # plt.plot(reference_track[:, 0], reference_track[:, 1], label="Reference track")
#     # plt.plot(x_guess[:, 0], x_guess[:, 1], label="Guess")
#     # plt.plot(x_sol[:, 0], x_sol[:, 1], label="Solution")
#     # plt.title(f"Iteration {i} result: {res}")
#     # plt.legend()
#     # plt.axis("equal")
#     # plt.show()

    fig, ax = plt.subplots(2, 6)
    fig.canvas.manager.full_screen_toggle()
    fig.suptitle(f"Iteration {i} result: {res}")
    dt = 0.2
    N = 20
    Tf = N * dt
    t_x = np.linspace(0, Tf, N + 1)
    t_u = t_x[:-1]


    ax[0, 0].plot(t_x, x_guess[:, 0], label="Guess")
    ax[0, 0].plot(t_x, x_sol[:, 0], label="Solution")
    ax[0, 0].plot(0.0, x0[0], "ro", label="Initial")
    ax[0, 0].set_title("x")
    ax[0, 0].legend()

    ax[0, 1].plot(t_x, x_guess[:, 1], label="Guess")
    ax[0, 1].plot(t_x, x_sol[:, 1], label="Solution")
    ax[0, 1].plot(0.0, x0[1], "ro", label="Initial")
    ax[0, 1].set_title("y")
    ax[0, 1].legend()

    ax[0, 2].plot(t_x, x_guess[:, 2] * 180 / np.pi, label="Guess")
    ax[0, 2].plot(t_x, x_sol[:, 2] * 180 / np.pi, label="Solution")
    ax[0, 2].plot(0.0, x0[2] * 180 / np.pi, "ro", label="Initial")
    ax[0, 2].set_title("theta")
    ax[0, 2].legend()

    ax[0, 3].plot(t_x, x_guess[:, 3] * 180 / np.pi, label="Guess")
    ax[0, 3].plot(t_x, x_sol[:, 3] * 180 / np.pi, label="Solution")
    ax[0, 3].plot(0.0, x0[3] * 180 / np.pi, "ro", label="Initial")
    ax[0, 3].set_title("delta")
    ax[0, 3].legend()

    ax[0, 4].plot(t_x, x_guess[:, 4], label="Guess")
    ax[0, 4].plot(t_x, x_sol[:, 4], label="Solution")
    ax[0, 4].plot(0.0, x0[4], "ro", label="Initial")
    ax[0, 4].set_title("v")
    ax[0, 4].legend()

    ax[0, 5].plot(t_x, x_guess[:, 5], label="Guess")
    ax[0, 5].plot(t_x, x_sol[:, 5], label="Solution")
    ax[0, 5].plot(0.0, x0[5], "ro", label="Initial")
    ax[0, 5].set_title("tau")
    ax[0, 5].legend()

    ax[1, 0].plot(t_u, u_guess[:, 0], label="Guess")
    ax[1, 0].plot(t_u, u_sol[:, 0], label="Solution")
    ax[1, 0].set_title("alpha")
    ax[1, 0].legend()

    ax[1, 1].plot(t_u, u_guess[:, 1] * 180 / np.pi, label="Guess")
    ax[1, 1].plot(t_u, u_sol[:, 1] * 180 / np.pi, label="Solution")
    ax[1, 1].set_title("phi")
    ax[1, 1].legend()

    ax[1, 2].plot(t_u, u_guess[:, 2], label="Guess")
    ax[1, 2].plot(t_u, u_sol[:, 2], label="Solution")
    ax[1, 2].set_title("zeta")
    ax[1, 2].legend()

    ax[1, 3].plot(x_guess[:, 0], x_guess[:, 1], label="Guess")
    ax[1, 3].plot(x_sol[:, 0], x_sol[:, 1], label="Solution")
    ax[1, 3].set_title("xy")
    ax[1, 3].set_aspect("equal")
    ax[1, 3].legend()

    plt.show()
