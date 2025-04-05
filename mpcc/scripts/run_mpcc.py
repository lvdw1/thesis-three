#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path as FilePath
import casadi as ca
import numpy as np
# import rospy
# import tf2_ros as tf
# import tf_conversions

# acados imports
from acados_template import AcadosOcp, AcadosOcpSolver
# from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
# from geometry_msgs.msg import PoseStamped
from initial_guess import initial_guess

# extra import specific for mpcc
from mpcc_bicycle_model import export_bicycle_model
# from nav_msgs.msg import  Path
# from node_fixture.managed_node import ManagedNode
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64
# from ugr_msgs.msg import State
from utils import create_spline_function, sample_equidistant_track_points

import yaml
import time
import json
import socket
import subprocess

import os

class UnityEnv:
    """
    Handles communication with Unity.
    """
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"[UnityEnv] Server listening on {self.host}:{self.port}...")
        self.sim_process = subprocess.Popen(["open", "../../unity/Simulator_WithTrackGeneration/sim_fssim_00.app"])
        self.client_socket, self.addr = self.server_socket.accept()
        print(f"[UnityEnv] Connection from {self.addr}")

    def receive_state(self):
        raw_data = self.client_socket.recv(1024).decode('utf-8').strip()
        messages = raw_data.splitlines()
        for msg in reversed(messages):
            msg = msg.strip()
            fields = msg.split(',')
            if len(fields) >= 6:
                state = {
                    "time": time.time(),
                    "x_pos": float(fields[0]),
                    "z_pos": float(fields[1]),
                    "yaw_angle": -float(fields[2]) + 90,
                    "long_vel": float(fields[3]),
                    "lat_vel": float(fields[4]),
                    "yaw_rate": float(fields[5]),
                }
                return state
        raise ValueError("Incomplete state received, not enough fields in any message.")

    def send_command(self, steering, throttle, brake):
        message = f"{steering},{throttle},{brake}\n"
        print(f"Sending {steering}, {throttle}, {brake}")
        self.client_socket.sendall(message.encode())

    def close(self):
        try:
            if self.client_socket:
                self.client_socket.close()
        except:
            pass
        self.server_socket.close()
        print("[UnityEnv] Connection closed.")

class MPCC:
  def __init__(self):
      self.load_config()
      self.prepare_ocp()

  def load_config(self):
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    self.wheel_radius = params.get("wheel_radius", 0.2032)
    self.wheelbase = params.get("wheelbase", 1.5)
    self.constraint_circle_radius = params.get("constraint_circle_radius", 0.8)
    self.N = params.get("N", 20)
    self.dt = params.get("dt", 0.1)
    self.alpha_min = params.get("alpha_min", -15.0)
    self.alpha_max = params.get("alpha_max", 15.0)

    self.phi_min = params.get("phi_min", -1.5)
    self.phi_max = params.get("phi_max", 1.5)
    self.zeta_min = params.get("zeta_min", 0.001)
    self.zeta_max = params.get("zeta_max", 1.2)
    self.delta_min = params.get("delta_min", -45) * np.pi / 180
    self.delta_max = params.get("delta_max", 45) * np.pi / 180
    self.v_min = params.get("v_min", 0)
    self.v_max = params.get("v_max", 12)
    self.tau_min = params.get("tau_min", 0.000000001)
    self.tau_max = params.get("tau_max", 2.0)

    self.first_iteration = True
    self.x_sol = np.zeros((self.N + 1, 6))
    self.u_sol = np.zeros((self.N, 3))
    self.steering_joint_angle = 0.0
    self.actual_speed = 0.0
    self.ocp_solver = None
    self.mpcc_prepared = False
    self.drive_effort_cmd = float(0.0)
    self.steering_vel_cmd = float(0.0)
    self.steering_pos_cmd = float(0.0)
    self.yaw = None
    self.t = time.time()
    self.get_path("../fssim_fsi2.json")

  def get_path(self, file_path):
    with open(file_path, "r") as f:
      data = json.load(f)
      self.path = np.array([data["centerline_x"], data["centerline_y"]]).T

  def create_ocp_solver(self):
    """
    Create acados_ocp solver instance
    """
    # create ocp and add model
    ocp = AcadosOcp()
    n = 109 # 49 for chicane, 109 for fssim_fsi, 173 for fsg24
    # rospy.logerr(f"n: {n}")
    model, self.spline_function = export_bicycle_model(n)
    ocp.model = model
    ocp.dims.np = 2 * n
    ocp.parameter_values = np.zeros((2 * n,))
    # add nonlinear circular constraints to ocp
    ocp.dims.nh = 1
    ocp.constraints.uh = np.array([0.0])
    ocp.constraints.lh = np.array([-self.constraint_circle_radius**2 - 0.1])
    ocp.dims.nh_e = 1
    ocp.constraints.uh_e = np.array([0.0])
    ocp.constraints.lh_e = np.array([-self.constraint_circle_radius - 0.1])
    # add dimensions and horizon to ocp
    # nx = model.x.rows()
    # nu = model.u.rows()
    ocp.solver_options.N_horizon = self.N
    ocp.solver_options.tf = self.N * self.dt
    # add cost type to ocp
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    # add simple input constraints to ocp
    ocp.constraints.idxbu = np.array([0, 1, 2])
    ocp.constraints.lbu = np.array([self.alpha_min, self.phi_min, self.zeta_min])
    ocp.constraints.ubu = np.array(
      [self.alpha_max, self.phi_max, self.zeta_max]
    ) # 10 times higher then Zander's because ACADOS multiplies by dt=0.1
    # add simple state constraints to ocp
    ocp.constraints.idxbx = np.array([3, 4, 5])
    ocp.constraints.lbx = np.array([self.delta_min, self.v_min, self.tau_min])
    ocp.constraints.ubx = np.array([self.delta_max, self.v_max, self.tau_max])
    ocp.constraints.idxbx_e = np.array([3, 4, 5])
    ocp.constraints.lbx_e = np.array([self.delta_min, self.v_min, self.tau_min])
    ocp.constraints.ubx_e = np.array([self.delta_max, self.v_max, self.tau_max])
    # current state TODO! -> give it actual values of the car!!
    start_state = unity.receive_state()
    ocp.constraints.x0 = np.array([start_state["x_pos"], start_state["z_pos"], 0.0, 0.0, 0.0, 0.0])
    # set the ocp solver
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    # ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    # ocp.solver_options.levenberg_marquardt = 1e-5
    ocp.solver_options.print_level = 1
    # set max iter
    ocp.solver_options.nlp_solver_max_iter = 15
    self.ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

  def get_current_state(self):
    """
    Get current state of the car:
    - x position
    - y position
    - yaw angle
    - steering angle
    - speed
    - tau
    """
    state = unity.receive_state()
    x_pos = state["x_pos"]
    y_pos = state["z_pos"]
    self.yaw = state["yaw_angle"]
    self.actual_speed = state["long_vel"]
    # get current tau value (for now just set it to tau of the previous iteration,
    # might be a good assumption as you move very little in an iteration) TODO!
    # tau = self.x_sol[0, 5]
    # calculate tau as the tau value of the point closest to the current position
    spline = create_spline_function(self.reference_track)
    num_points = 1000
    spline_points = spline(ca.linspace(0, 1, num_points).T).full().T
    dist_squared = np.sum((spline_points - np.array([x_pos, y_pos])) ** 2, axis=1)
    tau = np.argmin(dist_squared) / num_points
    return np.array(
      [x_pos, y_pos, self.yaw, self.steering_joint_angle, self.actual_speed, tau]
    )

  def prepare_ocp(self):
    """
    Prepare the ocp solver for the mpcc controller
    and set reference track and initial guess
    Function isn't called in active because it takes too long
    and results in a timeout error
    """
    # create ocp solver
    self.create_ocp_solver()
    # set reference track to the ocp solver
    self.set_reference_track()
    # create and set initial guess to the ocp solver
    self.set_initial_guess()
    # mpcc is prepared
    self.it = 0
    self.mpcc_prepared = True
    # make folder to save everything
    self.result_folder = (
      FilePath(__file__).parents[1]
      / "mpcc_results"
      / datetime.today().strftime("%d-%m-%Y__%H-%M")
      / "iterations"
    )
    self.result_folder.mkdir(parents=True, exist_ok=True)
    np.save(self.result_folder.parent / "reference_track", self.reference_track)

  def set_reference_track(self):
    # sample equidistant points from the track
    self.reference_track = sample_equidistant_track_points(self.path, 2)
    # set reference track to the ocp solver
    for i in range(self.N + 1):
      self.ocp_solver.set(i, "p", self.reference_track[:, :].flatten())

  def set_initial_guess(self):
    # get initial guess
    self.x_guess, self.u_guess = initial_guess(
      self.reference_track,
      self.N,
      self.dt,
      self.alpha_max,
      self.v_max,
      self.wheel_radius,
      self.wheelbase,
    )
    print("The initial guess for x is:", self.x_guess)
    print("The initial guess for u is:", self.u_guess)
    # save initial guess, the reference track and the spline function
    spline = create_spline_function(self.reference_track)
    spline_points = spline(ca.linspace(0, 1, 100).T).full().T
    np.savez(
      FilePath(__file__).parents[1] / "validation" / "initial_guess",
      x_guess=self.x_guess,
      u_guess=self.u_guess,
      ref_track=self.path,
      reference_track=self.reference_track,
      spline=spline_points,
    )
    # set initial guess to the ocp solver
    for i in range(self.N + 1):
      self.ocp_solver.set(i, "x", self.x_guess[i, :])
    for i in range(self.N):
      self.ocp_solver.set(i, "u", self.u_guess[i, :])

  def active(self):
    if self.mpcc_prepared:
      # set current state
      x0 = self.get_current_state()
      print("The received state from Unity is:", x0)
      # if self.first_iteration:
      #   x0[0] = -71
      #   x0[1] = -20
      # self.ocp_solver.set(0, "x", x0)
      self.ocp_solver.set(0, "lbx", x0)
      self.ocp_solver.set(0, "ubx", x0)
      # set initial guess
      if not self.first_iteration:
        for i in range(self.N + 1):
          self.ocp_solver.set(i, "x", self.x_guess[i, :])
        for i in range(self.N):
          self.ocp_solver.set(i, "u", self.u_guess[i, :])
      self.first_iteration = False
      # solve ocp
      res = self.ocp_solver.solve()
      # get x and u solutions
      self.x_sol = np.array(
        [self.ocp_solver.get(i, "x") for i in range(self.N + 1)]
      )
      self.u_sol = np.array([self.ocp_solver.get(i, "u") for i in range(self.N)])
      # save initial guess and solution of this iteration
      np.savez(
        self.result_folder / f"iteration_{self.it}",
        x_guess=self.x_guess,
        u_guess=self.u_guess,
        x_sol=self.x_sol,
        u_sol=self.u_sol,
        res=res,
        x0=x0,
        # ocp_solve_time=ocp_solve_time.to_sec() * 1e3,
        # solve_time=solve_time.to_sec() * 1e3,
      )
      self.it += 1
      # update initial guess for next iteration with the current solution
      # self.x_guess = np.vstack((self.x_sol[1:, :], self.x_sol[-1, :]))
      # self.u_guess = np.vstack((self.u_guess[1:,:], self.u_guess[-1, :]))
      self.x_guess = self.x_sol
      self.u_guess = self.u_sol
      # send commands to controllers
      self.drive_effort_cmd = self.u_sol[0, 0]
      self.steering_vel_cmd = self.u_sol[0, 1]
      self.steering_pos_cmd = self.x_sol[0, 3]

      # I added this, not sure if this will create problems
      self.steering_joint_angle = self.steering_pos_cmd

      # save solution
      filepath = FilePath(__file__).parents[1] / "validation" / "solution"
      np.savez(filepath, x_sol=self.x_sol, u_sol=self.u_sol)
      self.t = time.time()

if __name__ == "__main__": 
    unity = UnityEnv()

    mpcc = MPCC()
    try:
        while True:
            prev_time = time.time()

            current_state = mpcc.get_current_state()

            mpcc.active()

            steering = mpcc.steering_pos_cmd
            throttle = mpcc.drive_effort_cmd
            brake = 0.0

            unity.send_command(steering, throttle, brake)

            curr_time = time.time()
            sol_time = curr_time - prev_time
            print(sol_time)
            prev_time = curr_time
    except KeyboardInterrupt:
        print("Terminating & closing Unity connection")
    finally:
        unity.close()

