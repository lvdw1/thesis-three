#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path as FilePath

import casadi as ca
import numpy as np
import rospy
import tf2_ros as tf
import tf_conversions

# acados imports
from acados_template import AcadosOcp, AcadosOcpSolver
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from geometry_msgs.msg import PoseStamped
from initial_guess import initial_guess

# extra import specific for mpcc
from mpcc_bicycle_model import export_bicycle_model
from nav_msgs.msg import Odometry, Path
from node_fixture.managed_node import ManagedNode
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from ugr_msgs.msg import State
from utils import create_spline_function, sample_equidistant_track_points


class MPCC(ManagedNode):
    def __init__(self):
        super().__init__("mpcc_control")

        self.spin()

    def doConfigure(self):
        rospy.logerr("Configuring mpcc node")

        # subscribers
        self.odom_sub = super().AddSubscriber(
            "/input/odom", Odometry, self.get_odom_update
        )
        self.joint_state_sub = super().AddSubscriber(
            "/ugr/car/joint_states", JointState, self.get_joint_states
        )
        self.path_sub = super().AddSubscriber("/input/path", Path, self.get_path)
        self.slamstate_sub = super().AddSubscriber(
            "/state/slam", State, self.getSlamState
        )

        # publishers
        self.mpcc_prediction_pub = super().AddPublisher(
            "/mpcc/mpcc_prediction", Path, queue_size=1
        )
        self.drive__effort_pub = super().AddPublisher(
            "/output/drive_effort_controller/command", Float64, queue_size=10
        )
        self.steering_velocity_pub = super().AddPublisher(
            "/output/steering_velocity_controller/command", Float64, queue_size=10
        )
        self.steering_position_pub = super().AddPublisher(
            "/output/steering_position_controller/command", Float64, queue_size=10
        )

        # world frame and base link frame
        self.world_frame = rospy.get_param("~world_frame", "ugr/map")
        self.base_link_frame = rospy.get_param("~base_link_frame", "ugr/car_base_link")

        # tranform buffer and listener
        self.tf_buffer = tf.Buffer()
        self.tf_listener = tf.TransformListener(self.tf_buffer)

        # parameters
        self.reference_track = np.array([])
        self.wheel_radius = rospy.get_param("~wheel_radius", 0.2032)
        self.wheelbase = rospy.get_param("~wheelbase", 1.5)

        self.constraint_circle_radius = rospy.get_param(
            "~constraint_circle_radius", 0.8
        )

        self.N = rospy.get_param("~N", 20)
        self.dt = rospy.get_param("~dt")

        self.alpha_min = rospy.get_param("~alpha_min", -15.0)
        self.alpha_max = rospy.get_param("~alpha_max", 15.0)
        self.phi_min = rospy.get_param("~phi_min", -1.5)
        self.phi_max = rospy.get_param("~phi_max", 1.5)
        self.zeta_min = rospy.get_param("~zeta_min", 0.001)
        self.zeta_max = rospy.get_param("~zeta_max", 1.2)

        # parameters for state constraints
        self.delta_min = rospy.get_param("~delta_min", -45) * np.pi / 180
        self.delta_max = rospy.get_param("~delta_max", 45) * np.pi / 180
        self.v_min = rospy.get_param("~v_min", 0)
        self.v_max = rospy.get_param("~v_max", 12)
        self.tau_min = rospy.get_param("~tau_min", 0.000000001)
        self.tau_max = rospy.get_param("~tau_max", 2.0)

        self.first_iteration = True
        self.x_sol = np.zeros((self.N + 1, 6))
        self.u_sol = np.zeros((self.N, 3))

        self.steering_joint_angle = 0.0
        self.actual_speed = 0.0

        self.cur_slam_state = ""
        self.ocp_solver = None
        self.mpcc_prepared = False

        # controller commands
        self.drive_effort_cmd = Float64(0.0)
        self.steering_vel_cmd = Float64(0.0)
        self.steering_pos_cmd = Float64(0.0)

        self.yaw = None

        self.t = rospy.Time.now()

    def doActivate(self):
        rospy.logerr("Activating mpcc node")

        # Switch to ROS control controllers for mpcc racing
        # steering_position_controller -> steering_velocity_controller
        # drive_velocity_controller -> drive_effort_controller
        rospy.wait_for_service("/ugr/car/controller_manager/switch_controller")
        try:
            switch_controller = rospy.ServiceProxy(
                "/ugr/car/controller_manager/switch_controller", SwitchController
            )

            req = SwitchControllerRequest()
            req.start_controllers = [
                "joint_state_controller",
                "steering_velocity_controller",
                "drive_effort_controller",
            ]
            req.stop_controllers = [
                "steering_position_controller",
                "drive_velocity_controller",
            ]
            req.strictness = SwitchControllerRequest.BEST_EFFORT

            response = switch_controller(req)

            if response.ok:
                rospy.logerr("Controllers for mpcc racing have been started")

            else:
                rospy.logerr("Could not start controllers")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def get_odom_update(self, msg: Odometry):
        """
        Get current speed of the car
        """
        self.actual_speed = msg.twist.twist.linear.x

    def get_joint_states(self, msg: JointState):
        """
        Get current steering angle of the car
        """
        self.steering_joint_angle = msg.position[msg.name.index("axis_steering")]

    def get_path(self, msg: Path):
        "Receive path in global frame (ugr/map)"

        if msg is None:
            return

        # extract x and y coordinates from the path
        self.path = np.array(
            [[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses]
        )

    def create_ocp_solver(self):
        """
        Create acados_ocp solver instance
        """
        # create ocp and add model
        ocp = AcadosOcp()
        n = 109  # 49 for chicane, 109 for fssim_fsi, 173 for fsg24
        rospy.logerr(f"n: {n}")
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
        )  # 10 times higher then Zander's because ACADOS multiplies by dt=0.1

        # add simple state constraints to ocp
        ocp.constraints.idxbx = np.array([3, 4, 5])
        ocp.constraints.lbx = np.array([self.delta_min, self.v_min, self.tau_min])
        ocp.constraints.ubx = np.array([self.delta_max, self.v_max, self.tau_max])
        ocp.constraints.idxbx_e = np.array([3, 4, 5])
        ocp.constraints.lbx_e = np.array([self.delta_min, self.v_min, self.tau_min])
        ocp.constraints.ubx_e = np.array([self.delta_max, self.v_max, self.tau_max])

        # current state TODO! -> give it actual values of the car!!
        ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # set the ocp solver
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "ERK"
        # ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

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
        # get current position
        trans = self.tf_buffer.lookup_transform(
            self.world_frame,
            self.base_link_frame,
            rospy.Time(),
        )
        x_pos = trans.transform.translation.x
        y_pos = trans.transform.translation.y
        # x_pos = self.reference_track[0, 0]
        # y_pos = self.reference_track[0, 1]

        # yaw = np.arctan2(self.reference_track[1, 1] - self.reference_track[0, 1], self.reference_track[1, 0] - self.reference_track[0, 0])
        # Get the quaternion components
        qx = trans.transform.rotation.x
        qy = trans.transform.rotation.y
        qz = trans.transform.rotation.z
        qw = trans.transform.rotation.w

        # Convert quaternion to Euler angles to get yaw angle
        yaw = tf_conversions.transformations.euler_from_quaternion((qx, qy, qz, qw))[2]
        # update self.yaw but make sure no jump occurs with respect to the previous yaw
        if self.yaw is not None:
            while abs(yaw - self.yaw) > np.pi:
                if yaw > self.yaw:
                    yaw -= 2 * np.pi
                else:
                    yaw += 2 * np.pi
        self.yaw = yaw

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

    def getSlamState(self, state: State):
        self.cur_slam_state = state.cur_state
        rospy.logerr("received slamstate")
        if self.cur_slam_state == "racing":
            self.prepare_ocp()

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
            t0 = rospy.Time.now()

            # set current state
            x0 = self.get_current_state()
            t1 = rospy.Time.now()
            current_state_time = t1 - t0
            # if self.first_iteration:
            #     x0[0] = self.reference_track[0, 0]
            #     x0[1] = self.reference_track[0, 1]
            # self.ocp_solver.set(0, "x", x0)
            self.ocp_solver.set(0, "lbx", x0)
            self.ocp_solver.set(0, "ubx", x0)
            t2 = rospy.Time.now()
            set_current_state_time = t2 - t1
            # set initial guess
            if not self.first_iteration:
                for i in range(self.N + 1):
                    self.ocp_solver.set(i, "x", self.x_guess[i, :])
                for i in range(self.N):
                    self.ocp_solver.set(i, "u", self.u_guess[i, :])
            self.first_iteration = False
            t3 = rospy.Time.now()
            set_initial_guess_time = t3 - t2
            # solve ocp
            res = self.ocp_solver.solve()
            t4 = rospy.Time.now()
            ocp_solve_time = t4 - t3

            # get x and u solutions
            self.x_sol = np.array(
                [self.ocp_solver.get(i, "x") for i in range(self.N + 1)]
            )
            self.u_sol = np.array([self.ocp_solver.get(i, "u") for i in range(self.N)])
            t5 = rospy.Time.now()
            get_solution_time = t5 - t4
            solve_time = t5 - t0

            # save initial guess and solution of this iteration
            np.savez(
                self.result_folder / f"iteration_{self.it}",
                x_guess=self.x_guess,
                u_guess=self.u_guess,
                x_sol=self.x_sol,
                u_sol=self.u_sol,
                res=res,
                x0=x0,
                ocp_solve_time=ocp_solve_time.to_sec() * 1e3,
                solve_time=solve_time.to_sec() * 1e3,
            )
            t6 = rospy.Time.now()
            save_sol_time = t6 - t5
            self.it += 1

            # update initial guess for next iteration with the current solutio
            # self.x_guess = np.vstack((self.x_sol[1:, :], self.x_sol[-1, :]))
            # self.u_guess = np.vstack((self.u_guess[1:,:], self.u_guess[-1, :]))
            self.x_guess = self.x_sol
            self.u_guess = self.u_sol

            # send commands to controllers
            # rospy.logerr(
            #     f"sending controller commands: \n motor effort cmd: {self.u_sol[0, 0]} \n steering vel cmd: {self.u_sol[0, 1]}"
            # )

            t7 = rospy.Time.now()
            self.drive_effort_cmd.data = self.u_sol[0, 0]
            self.steering_vel_cmd.data = self.u_sol[0, 1]
            self.steering_pos_cmd.data = self.x_sol[0, 3]
            self.drive__effort_pub.publish(self.drive_effort_cmd)
            self.steering_velocity_pub.publish(self.steering_vel_cmd)
            t8 = rospy.Time.now()
            publish_commands_time = t8 - t7

            # # save solution
            # filepath = FilePath(__file__).parents[1] / "validation" / "solution"
            # np.savez(filepath, x_sol=self.x_sol, u_sol=self.u_sol)

            # publish prediction
            mpcc_path = Path()
            mpcc_path.header.stamp = rospy.Time.now()
            mpcc_path.header.frame_id = self.world_frame
            for i in range(self.N + 1):
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = self.world_frame
                pose.pose.position.x = self.x_sol[i, 0]
                pose.pose.position.y = self.x_sol[i, 1]
                mpcc_path.poses.append(pose)
            self.mpcc_prediction_pub.publish(mpcc_path)
            t9 = rospy.Time.now()
            publish_prediction_time = t9 - t8

            t = rospy.Time.now()
            # print all the solve times
            rospy.logerr(f"current state time: {current_state_time.to_sec()*1e3} ms")
            rospy.logerr(
                f"set current state time: {set_current_state_time.to_sec()*1e3} ms"
            )
            rospy.logerr(
                f"set initial guess time: {set_initial_guess_time.to_sec()*1e3} ms"
            )
            rospy.logerr(f"ocp solve time: {ocp_solve_time.to_sec()*1e3} ms")
            rospy.logerr(f"get solution time: {get_solution_time.to_sec()*1e3} ms")
            rospy.logerr(f"total solve time: {solve_time.to_sec()*1e3} ms")
            rospy.logerr(f"save solution time: {save_sol_time.to_sec()*1e3} ms")
            rospy.logerr(
                f"publish commands time: {publish_commands_time.to_sec()*1e3} ms"
            )
            rospy.logerr(
                f"publish prediction time: {publish_prediction_time.to_sec()*1e3} ms"
            )

            rospy.logerr(
                f"a total mpcc iteration, from publish to publish, took: {(t - self.t).to_sec()*1e3} seconds"
            )

            self.t = rospy.Time.now()


MPCC()
