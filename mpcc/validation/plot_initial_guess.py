from pathlib import Path as FilePath

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from utils_plotting import plot_result_xy

loader = np.load(FilePath(__file__).parents[1] / "validation" / "initial_guess.npz")
x_guess = loader["x_guess"]
u_guess = loader["u_guess"]
ref_track = loader["ref_track"]
reference_track = loader["reference_track"]
spline = loader["spline"]

solution = np.load(FilePath(__file__).parents[1] / "validation" / "solution.npz")
x_solution = solution["x_sol"]
u_solution = solution["u_sol"]

fig, ax = plt.subplots(2, 6)
dt = 0.1
N = 20
Tf = N * dt
t_x = np.linspace(0, Tf, N + 1)
t_u = t_x[:-1]


ax[0, 0].plot(t_x, x_guess[:, 0], label="Guess")
ax[0, 0].plot(t_x, x_solution[:, 0], label="Solution")
ax[0, 0].set_title("x")
ax[0, 0].legend()

ax[0, 1].plot(t_x, x_guess[:, 1], label="Guess")
ax[0, 1].plot(t_x, x_solution[:, 1], label="Solution")
ax[0, 1].set_title("y")
ax[0, 1].legend()

ax[0, 2].plot(t_x, x_guess[:, 2] * 180 / np.pi, label="Guess")
ax[0, 2].plot(t_x, x_solution[:, 2] * 180 / np.pi, label="Solution")
ax[0, 2].set_title("theta")
ax[0, 2].legend()

ax[0, 3].plot(t_x, x_guess[:, 3] * 180 / np.pi, label="Guess")
ax[0, 3].plot(t_x, x_solution[:, 3] * 180 / np.pi, label="Solution")
ax[0, 3].set_title("delta")
ax[0, 3].legend()

ax[0, 4].plot(t_x, x_guess[:, 4], label="Guess")
ax[0, 4].plot(t_x, x_solution[:, 4], label="Solution")
ax[0, 4].set_title("v")
ax[0, 4].legend()

ax[0, 5].plot(t_x, x_guess[:, 5], label="Guess")
ax[0, 5].plot(t_x, x_solution[:, 5], label="Solution")
ax[0, 5].set_title("tau")
ax[0, 5].legend()

ax[1, 0].plot(t_u, u_guess[:, 0], label="Guess")
ax[1, 0].plot(t_u, u_solution[:, 0], label="Solution")
ax[1, 0].set_title("alpha")
ax[1, 0].legend()

ax[1, 1].plot(t_u, u_guess[:, 1] * 180 / np.pi, label="Guess")
ax[1, 1].plot(t_u, u_solution[:, 1] * 180 / np.pi, label="Solution")
ax[1, 1].set_title("phi")
ax[1, 1].legend()

ax[1, 2].plot(t_u, u_guess[:, 2], label="Guess")
ax[1, 2].plot(t_u, u_solution[:, 2], label="Solution")
ax[1, 2].set_title("zeta")
ax[1, 2].legend()

ax[1, 3].plot(x_guess[:, 0], x_guess[:, 1], label="Guess")
ax[1, 3].plot(x_solution[:, 0], x_solution[:, 1], label="Solution")
ax[1, 3].set_title("xy")
ax[1, 3].set_aspect("equal")
ax[1, 3].legend()

# plt.show()

fig = plt.figure()
plt.plot(ref_track[:, 0], ref_track[:, 1], "ro")
plt.gca().set_aspect("equal")

fig = plt.figure()
colors = cm.viridis(np.linspace(0, 1, len(reference_track)))
for i in range(len(reference_track)):
    plt.plot(reference_track[i, 0], reference_track[i, 1], "o", color=colors[i])
plt.colorbar(cm.ScalarMappable(cmap=cm.viridis), ax=plt.gca(), label="Color Gradient")
plt.plot(x_guess[:, 0], x_guess[:, 1], "r--")

plt.gca().set_aspect("equal")

fig = plt.figure()
colors = cm.viridis(np.linspace(0, 1, len(spline)))
for i in range(len(spline)):
    plt.plot(spline[i, 0], spline[i, 1], "o", color=colors[i])
plt.colorbar(cm.ScalarMappable(cmap=cm.viridis), ax=plt.gca(), label="Color Gradient")
plt.plot(x_guess[:, 0], x_guess[:, 1], "r--")
plt.gca().set_aspect("equal")
plt.show()

fig = plot_result_xy(
    ref_track,
    reference_track,
    x_solution,
    x_guess,
    ["x", "y", "theta", "delta", "v", "tau"],
    N,
    0.8,
)
plt.show()
