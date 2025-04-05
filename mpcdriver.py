import numpy as np
import time
import math as m
import casadi as ca  # Make sure to install casadi

from rldriver import UnityEnv, Processor, shift_position_single

class MPCDriver:
    def __init__(self,
                 wheelbase=1.5,
                 dt=0.02,            # Discretization time for the model
                 horizon=100,        # Number of timesteps in the MPC horizon
                 max_steering_deg=45,
                 max_throttle=1.0,
                 max_brake=1.0):
        """
        Basic MPC driver for a kinematic bicycle model.
        """
        self.wheelbase = wheelbase
        self.dt = dt
        self.N = horizon

        # Convert steering limit to radians
        self.max_steering = np.radians(max_steering_deg)

        self.max_throttle = max_throttle
        self.max_brake = max_brake

        # === Create CasADi structures for the MPC problem ===
        self._setup_mpc()

    def _setup_mpc(self):
        """
        Sets up the MPC optimization problem using a kinematic bicycle model:
        
        States: x, z, yaw, v
        Inputs: delta (steering), a (acceleration)
        
        x_{k+1} = x_k   + v_k*cos(yaw_k)*dt
        z_{k+1} = z_k   + v_k*sin(yaw_k)*dt
        yaw_{k+1} = yaw_k + (v_k / L)*tan(delta_k)*dt
        v_{k+1} = v_k   + a_k*dt
        """
        # State variables
        x = ca.SX.sym('x')
        z = ca.SX.sym('z')
        yaw = ca.SX.sym('yaw')
        v = ca.SX.sym('v')
        
        # Control variables
        delta = ca.SX.sym('delta')  # steering angle
        a = ca.SX.sym('a')          # acceleration (throttle - brake)

        # State vector and control vector
        state = ca.vertcat(x, z, yaw, v)
        control = ca.vertcat(delta, a)

        # System dynamics (kinematic bicycle)
        # (Assuming small slip angles, the bicycle model is fairly straightforward)
        x_next = x + v * ca.cos(yaw) * self.dt
        z_next = z + v * ca.sin(yaw) * self.dt
        yaw_next = yaw + (v / self.wheelbase) * ca.tan(delta) * self.dt
        v_next = v + a * self.dt

        state_next = ca.vertcat(x_next, z_next, yaw_next, v_next)

        # A function for the state propagation
        self.dynamics = ca.Function('f_dynamics', [state, control], [state_next])

        # For the MPC problem we define symbolic variables over the horizon
        self.U = ca.SX.sym('U', 2, self.N)       # 2 control inputs, over N timesteps
        self.X = ca.SX.sym('X', 4, self.N + 1)   # 4 states, over N+1 timesteps
        self.P = ca.SX.sym('P', 4 + 2*(self.N))  
        """
        We will pack the parameter vector P as follows:
        P = [ x0, z0, yaw0, v0, 
              x_ref_0, z_ref_0,
              x_ref_1, z_ref_1,
              ...
              x_ref_{N-1}, z_ref_{N-1} ]
        
        - (x0, z0, yaw0, v0) is the current state.
        - (x_ref_k, z_ref_k) are the reference positions at each timestep k (or predicted reference).
        """

        # Cost function, constraints
        self.obj = 0
        self.g = []

        # Weights (tune to your liking)
        W_pos = 50.0     # Position error weight
        W_yaw = 10.0     # Heading error weight
        W_v = 0.1       # Speed error weight
        W_delta = 20.0 # Steering usage weight (smooth steering)
        W_a = 10.0      # Acceleration usage weight
        W_ddelta = 50.0# Steering rate (smooth changes)
        W_da = 10.0     # Acceleration rate

        # Build the cost by iterating over the horizon
        # The first 4 entries of P are the current state
        x0 = self.P[0]
        z0 = self.P[1]
        yaw0 = self.P[2]
        v0 = self.P[3]

        # "Lift" the initial state X[:,0] == (x0, z0, yaw0, v0)
        self.g.append(self.X[:, 0] - ca.vertcat(x0, z0, yaw0, v0))

        for k in range(self.N):
            # The reference for position at step k
            x_ref = self.P[4 + 2*k]
            z_ref = self.P[4 + 2*k + 1]

            # The current state variables
            st_k = self.X[:, k]
            con_k = self.U[:, k]

            # Next state variables
            st_k_next = self.X[:, k+1]

            # Cost: track position (x,z), heading (yaw) optional, speed optional
            self.obj += W_pos * ((st_k[0] - x_ref)**2 + (st_k[1] - z_ref)**2)
            self.obj += W_yaw * (st_k[2]**2)
            self.obj += W_v * ((st_k[3] - 0.0)**2)  # Example: you might want to track a target speed

            # Cost on controls
            self.obj += W_delta * (con_k[0]**2)  # steering
            self.obj += W_a * (con_k[1]**2)      # acceleration

            # Add constraints for system dynamics
            st_k_next_model = self.dynamics(st_k, con_k)
            self.g.append(st_k_next - st_k_next_model)

            # Smoothness cost for control changes (delta rate, acceleration rate)
            if k < self.N-1:
                con_k_next = self.U[:, k+1]
                self.obj += W_ddelta * ((con_k_next[0] - con_k[0])**2)
                self.obj += W_da * ((con_k_next[1] - con_k[1])**2)

        # Concatenate constraints
        self.g = ca.vertcat(*self.g)

        # Decision variable vector
        # Flatten the X and U into a single vector
        self.vars = ca.vertcat(
            self.X.reshape((-1, 1)),  # N+1 states
            self.U.reshape((-1, 1))   # N controls
        )

        # Number of decision variables
        self.n_vars = (self.N+1)*4 + self.N*2
        self.n_g = (self.N+1)*4

        # Define bounds for the decision variables
        # (We must define them carefully: states can be free or partially bounded,
        #  but controls must be bounded.)

        # For example, let’s create big bounds on X, but realistic bounds on U.
        self.vars_lb = []
        self.vars_ub = []

        # X bounds
        for _ in range(self.N+1):
            # x, z, yaw, v
            self.vars_lb += [-1e5, -1e5, -1e5, 0.0]  # might want v >= 0
            self.vars_ub += [1e5, 1e5,  1e5, 50.0]   # or some max speed

        # U bounds
        for _ in range(self.N):
            # delta, a
            self.vars_lb += [-self.max_steering, -3.0]  # negative a for braking
            self.vars_ub += [ self.max_steering,  3.0]

        self.vars_lb = np.array(self.vars_lb, dtype=float)
        self.vars_ub = np.array(self.vars_ub, dtype=float)

        # For constraints g, we set them all to 0 because they are equality constraints
        self.g_lb = np.zeros(self.n_g)
        self.g_ub = np.zeros(self.n_g)

        # Create the NLP
        nlp = {
            'f': self.obj,
            'x': self.vars,
            'p': self.P,
            'g': self.g
        }

        opts = {
            'ipopt.max_iter': 100,
            'ipopt.print_level': 0,
            'print_time': False,
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # We will store the "warm-start" solution (initial guess) to speed up consecutive solves
        self.vars_init = np.zeros(self.n_vars)

    def _shift_mpc_solution(self, sol):
        """
        If you want to do a 'shift' in time of the solution to warm-start the solver
        in the next iteration, you can implement that here.
        Typically: X(k+1) becomes the new X0, and so forth.
        """
        # This is optional but highly recommended for performance. 
        pass

    def mpc_control(self, current_state, reference_points):
        """
        Solve the MPC optimization problem for the current state and a set of reference points
        on the centerline.
        
        :param current_state: dict with keys {"x_pos", "z_pos", "yaw_angle", "long_vel"} (or similar)
        :param reference_points: list of (x_ref, z_ref) for the horizon
        :return: (steering, throttle, brake) as float
        """
        # current_state
        x0 = current_state["x_pos"]
        z0 = current_state["z_pos"]
        yaw0 = np.radians(current_state["yaw_angle"])
        v0 = current_state.get("long_vel", 0.0)  # or your speed variable

        # The parameter vector P
        # We assume reference_points has length >= N for simplicity
        ref_arr = []
        for i in range(self.N):
            ref_arr.append(reference_points[i][0])  # x_ref
            ref_arr.append(reference_points[i][1])  # z_ref
        p = np.array([x0, z0, yaw0, v0] + ref_arr)

        # Solve
        sol = self.solver(
            x0=self.vars_init,
            lbx=self.vars_lb,
            ubx=self.vars_ub,
            lbg=self.g_lb,
            ubg=self.g_ub,
            p=p
        )
        sol_x = sol['x'].full().flatten()

        # Extract control at time k=0
        # The states (X) occupy the first (N+1)*4 entries
        # The controls (U) occupy the next N*2 entries
        offset_x = (self.N+1)*4
        delta_0 = sol_x[offset_x + 0]
        a_0 = sol_x[offset_x + 1]

        # Shift if desired
        self._shift_mpc_solution(sol_x)

        # Convert 'a_0' into throttle/brake
        # Something naive: if a_0 >= 0 => throttle, else => brake
        # You can do more advanced logic here.
        if a_0 >= 0:
            throttle = np.clip(a_0 / 3.0, 0.0, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            # For example, negative acceleration -1 => 1/3 brake
            brake = np.clip(-a_0 / 3.0, 0.0, self.max_brake)

        # Steering: convert from radians to [-1, 1] for a [-max_steering, max_steering] system
        steering = -delta_0 / self.max_steering
        steering = float(np.clip(steering, -1.0, 1.0))

        # Save the solution as the next warm start
        self.vars_init = sol_x

        return steering, throttle, brake

    def get_reference_points(self, centerline, current_state):
        """
        Example function to pick N reference points from the centerline.
        For simplicity, let's choose them as equally spaced points in front of the car,
        but you could do something more sophisticated.
        """
        x_car = current_state["x_pos"]
        z_car = current_state["z_pos"]
        yaw_car = np.radians(current_state["yaw_angle"])
        heading = np.array([m.cos(yaw_car), m.sin(yaw_car)])
        car_pos = np.array([x_car, z_car])

        # Pick points that are "ahead" in the track. 
        # Then sample up to N points from them.
        ahead_pts = []
        for pt in centerline:
            vec = np.array([pt[0], pt[1]]) - car_pos
            if np.dot(vec, heading) > 0:  # ahead of the car
                ahead_pts.append(pt)

        # Sort them by distance from the car
        ahead_pts.sort(key=lambda p: np.hypot(p[0]-x_car, p[1]-z_car))

        # If we have fewer than N points, just replicate the last or do fallback
        if len(ahead_pts) < self.N:
            if len(ahead_pts) == 0:
                # fallback: pick the closest in all centerline
                distances = [np.hypot(p[0]-x_car, p[1]-z_car) for p in centerline]
                idx = np.argmin(distances)
                return [centerline[idx]]*self.N
            else:
                # replicate the last point
                return ahead_pts + [ahead_pts[-1]] * (self.N - len(ahead_pts))
        else:
            return ahead_pts[:self.N]

    def control(self, state, centerline):
        """
        High-level interface: 
        1. Obtain reference points 
        2. Call the MPC solve
        3. Return (steering, throttle, brake)
        """
        # Possibly shift the position if needed, like in your pure pursuit code
        # ( 1.7 lateral offset example )
        x_pos, z_pos = shift_position_single(
            state["x_pos"],
            state["z_pos"],
            state["yaw_angle"],
            shift_distance=1.7
        )
        # Overwrite the state's x_pos, z_pos for the solver
        state_for_mpc = {
            "x_pos": x_pos,
            "z_pos": z_pos,
            "yaw_angle": state["yaw_angle"],
            "long_vel": state.get("long_vel", 0.0)
        }

        ref_points = self.get_reference_points(centerline, state_for_mpc)
        steering, throttle, brake = self.mpc_control(state_for_mpc, ref_points)
        return steering, throttle, brake


if __name__ == "__main__":
    # === Example main loop using the MPC driver ===
    unity = UnityEnv(host='127.0.0.1', port=65432)
    processor = Processor()
    track_data = processor.build_track_data("sim/tracks/track17.json")
    centerline = track_data["centerline_pts"]

    # Initialize MPC Driver
    agent = MPCDriver(
        wheelbase=1.5,
        dt=0.05,
        horizon=10,
        max_steering_deg=45,
        max_throttle=1.0,
        max_brake=1.0
    )

    run_data = []
    try:
        while True:
            # Receive current state from the simulator
            state = unity.receive_state()

            # Solve for control
            steering, throttle, brake = agent.control(state, centerline)

            # Send control to simulator
            unity.send_command(steering, throttle, brake)

            # Record data
            run_data.append({
                "time": state["time"],
                "x_pos": state["x_pos"],
                "z_pos": state["z_pos"],
                "yaw_angle": state["yaw_angle"],
                "speed": state.get("speed", 0),
                "steering": steering,
                "throttle": throttle,
                "brake": brake
            })

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Terminating MPC agent and recording data...")
    finally:
        unity.close()
