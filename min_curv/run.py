#!/usr/bin/env python3

import time
from time import perf_counter

import numpy as np
from utils import (
    boundary_check,
    calc_ax_max_motors,
    calc_ggv,
    check_traj,
    iqp_mincurv_handler,
    prep_track_handler,
    result_plots,
    shortest_path_handler,
)

import json
import yaml

class MinimumCurvature():
    def __init__(self):
        self.load_config()

    def load_config(self):
        with open("params.yaml", "r") as file:
            self.params = yaml.safe_load(file)

        self.car_width = self.params.get("car_width", 1.49)
        self.car_length = self.params.get("car_length", 3.21)
        self.car_mass = self.params.get("car_mass", 270.00)
        self.width_margin_left = self.params.get("width_margin_left", 0.15)
        self.width_margin_right = self.params.get("width_margin_right", 0.15)
        self.cone_width = self.params.get("cone_width", 0.24)
        self.min_track_width = self.params.get("min_track_width", 3.00)
        self.car_width_with_margins = (
            self.car_width
            + self.width_margin_left
            + self.width_margin_right
            + 2 * self.cone_width / 2
        )
        self.kappa_max = self.params.get("kappa_max", 0.275)
        self.curv_error_allowed = self.params.get("curv_error_allowed", 0.05)
        self.stepsize_prep_trajectory = self.params.get(
            "stepsize_prep_trajectory", 0.50
        )
        self.stepsize_prep_boundaries = self.params.get(
            "stepsize_prep_boundaries", 0.10
        )
        self.smoothing_factor_prep_trajectory = self.params.get(
            "smoothing_factor_prep_trajectory", 2.0
        )
        self.smoothing_factor_prep_boundaries = self.params.get(
            "smoothing_factor_prep_boundaries", 0.1
        )
        self.stepsize_opt = self.params.get("stepsize_opt", 0.50)

        self.stepsize_post = self.params.get("stepsize_interp", 0.25)
        self.min_iterations_iqp = self.params.get("min_iterations_iqp", 3)
        self.max_iterations_iqp = self.params.get("max_iterations_iqp", 5)
        self.boundary_check = self.params.get("boundary_check", False)
        self.stepsize_boundary_check = self.params.get(
            "stepsize_boundary_check", 0.00001
        )
        self.calc_shortest_path = self.params.get("calc_shortest_path", False)
        self.postprocessing = self.params.get("postprocessing", True)
        self.vel_calc = self.params.get("vel_calc", True)
        self.vel_calc_all = self.params.get("vel_calc_all", True)
        self.vel_calc_general = True
        self.vel_calc_prep = True
        self.vel_calc_shpath = True
        self.vel_calc_iqp = True
        self.vel_calc_general = True
        self.v_max = self.params.get("v_max", 36.00)
        self.T_motor_max = self.params.get("T_motor_max", 140.00)
        self.P_max = self.params.get("P_max", 80000.00)
        self.num_motors = self.params.get("num_motors", 2)
        self.gear_ratio = self.params.get("gear_ratio", 3.405)
        self.wheel_radius = self.params.get("wheel_radius", 0.206)
        self.drag_coeff = self.params.get("drag_coeff", 1.649217)
        self.acc_limit_long = self.params.get("acc_limit_long", 15.69)
        self.acc_limit_lat = self.params.get("acc_limit_lat", 15.69)
        self.save_plots = self.params.get("save_plots", False)
        self.plot = self.params.get("plot", True)
        self.print_debug = self.params.get("print_debug", False)
        self.plot_all = self.params.get("plot_all", False)
        self.print_debug_all = self.params.get("print_debug_all", False)
        if self.plot:
            self.plot_prep = self.params.get("plot_prep", False)
            self.plot_opt_shpath = self.params.get("plot_opt_shpath", False)
            self.plot_opt_iqp = self.params.get("plot_opt_iqp", False)
            self.plot_post_shpath = self.params.get("plot_post_shpath", False)
            self.plot_post_iqp = self.params.get("plot_post_iqp", False)
            self.plot_final = self.params.get("plot_final", True)
            self.plot_data_general = True
        else:
            self.plot_prep = False
            self.plot_opt_shpath = False
            self.plot_opt_iqp = False
            self.plot_post_shpath = False
            self.plot_post_iqp = False
            self.plot_final = False
            self.plot_data_general = False
        if self.print_debug:
            self.print_debug_prep = self.params.get("print_debug_prep", False)
            self.print_debug_opt_shpath = self.params.get(
                "print_debug_opt_shpath", False
            )
            self.print_debug_opt_iqp = self.params.get("print_debug_opt_iqp", False)
            self.print_debug_post_shpath = self.params.get(
                "print_debug_post_shpath", False
            )
            self.print_debug_post_iqp = self.params.get("print_debug_post_iqp", False)
            self.print_debug_final = self.params.get("print_debug_final", False)
            self.print_data_general = True
        else:
            self.print_debug_prep = False
            self.print_debug_opt_shpath = False
            self.print_debug_opt_iqp = False
            self.print_debug_post_shpath = False
            self.print_debug_post_iqp = False
            self.print_debug_final = False
            self.print_data_general = False
        if self.plot_all:
            self.plot_prep = True
            self.plot_opt_shpath = True
            self.plot_opt_iqp = True
            self.plot_post_shpath = True
            self.plot_post_iqp = True
            self.plot_final = True
            self.plot_data_general = True
        if self.print_debug_all:
            self.print_debug_prep = True
            self.print_debug_opt_shpath = True
            self.print_debug_opt_iqp = True
            self.print_debug_post_shpath = True
            self.print_debug_post_iqp = True
            self.print_debug_final = True
            self.print_data_general = True
        # Declare variables
        self.reference_line = np.array([])
        self.cones_left = np.array([])
        self.cones_right = np.array([])
        self.header = None

        self.calculate = False

    def doActivate(self):
        self.calculate = True

    def receive_new_path(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            self.reference_line = np.array([data["centerline_x"], data["centerline_y"]]).T

    def receive_new_boundaries(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            self.cones_left = np.array([data["left_boundary_x"], data["left_boundary_y"]]).T
            self.cones_right = np.array([data["right_boundary_x"], data["right_boundary_y"]]).T

    def active(self):
        if (
            not self.calculate
            or self.reference_line.size == 0
            or self.cones_left.size == 0
            or self.cones_right.size == 0
        ):
            return

        t_start = time.perf_counter()

        # ----------------------------------------------------------------------------------------------------------------------
        # OPTIONAL: VELOCITY DATA CALCULATION ----------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        if self.vel_calc_general:
            t_start_vel_data = time.perf_counter()

            ggv = calc_ggv(
                v_max=self.v_max,
                acc_limit_long=self.acc_limit_long,
                acc_limit_lat=self.acc_limit_lat,
            )

            ax_max_motors = calc_ax_max_motors(
                v_max=self.v_max,
                T_motor_max=self.T_motor_max,
                P_max=self.P_max,
                num_motors=self.num_motors,
                gear_ratio=self.gear_ratio,
                wheel_radius=self.wheel_radius,
                car_mass=self.car_mass,
            )

            car_vel_data = {
                "car_width": self.car_width,
                "car_length": self.car_length,
                "car_mass": self.car_mass,
                "v_max": self.v_max,
                "drag_coeff": self.drag_coeff,
                "ggv": ggv,
                "ax_max_motors": ax_max_motors,
            }

        else:
            car_vel_data = None

        # ----------------------------------------------------------------------------------------------------------------------
        # PREPROCESSING TRACK --------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        t_prep_start = time.perf_counter()

        (
            prepped_track,
            self.bound_left,
            self.bound_right,
            coeffs_x_prepped,
            coeffs_y_prepped,
            M_prepped,
            normvec_normalized_prepped,
            kappa_prepped_track,
            dkappa_prepped_track,
            s_raceline_prepped,
            vx_profile_prepped,
            ax_profile_prepped,
            lap_time_prepped,
            kappa_ref,
            dkappa_ref,
            distances_along_traj_ref,
        ) = prep_track_handler(
            trajectory_=self.reference_line,
            cones_left_=self.cones_left,
            cones_right_=self.cones_right,
            min_track_width=self.min_track_width,
            stepsize_trajectory=self.stepsize_prep_trajectory,
            stepsize_boundaries=self.stepsize_prep_boundaries,
            sf_trajectory=self.smoothing_factor_prep_trajectory,
            sf_boundaries=self.smoothing_factor_prep_boundaries,
            calc_vel=self.vel_calc_prep,
            car_vel_data=car_vel_data,
            print_debug=self.print_debug_prep,
            plot_debug=self.plot_prep,
            plot_data_general=self.plot_data_general,
            print_data_general=self.print_data_general,
        )

        # ----------------------------------------------------------------------------------------------------------------------
        # OPTIONAL: BOUNDARY CALCULATION CHECK ---------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        if self.boundary_check:
            t_start_boundary_check = time.perf_counter()
            boundary_check(
                bound_left_=self.bound_left,
                bound_right_=self.bound_right,
                cones_left_=self.cones_left,
                cones_right_=self.cones_right,
                stepsize=self.stepsize_boundary_check,
            )
        else:
            pass

        # ----------------------------------------------------------------------------------------------------------------------
        # OPTIONAL: SHORTEST PATH OPTIMISATION ---------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        if self.calc_shortest_path:
            t_start_shortest_path_optimization = perf_counter()

            (
                shpath_track,
                coeffs_x_shpath,
                coeffs_y_shpath,
                M_shpath,
                normvec_norm_shpath,
                kappa_shpath,
                dkappa_shpath,
                s_raceline_shpath,
                vx_profile_shpath,
                ax_profile_shpath,
                lap_time_shpath,
            ) = shortest_path_handler(
                width_veh_real=self.car_width,
                width_veh_opt=self.car_width_with_margins,
                prepped_track_=prepped_track,
                normvec_norm_prepped_=normvec_normalized_prepped,
                stepsize_shpath=self.stepsize_opt,
                cones_left_=self.cones_left,
                cones_right_=self.cones_right,
                initial_poses_=self.reference_line,
                kappa_prepped_=kappa_prepped_track,
                dkappa_prepped_=dkappa_prepped_track,
                bound_left_=self.bound_left,
                bound_right_=self.bound_right,
                s_raceline_prepped_=s_raceline_prepped,
                calc_vel=self.vel_calc_shpath,
                car_vel_data=car_vel_data,
                print_debug=self.print_debug_opt_shpath,
                plot_debug=self.plot_opt_shpath,
                plot_data_general=self.plot_data_general,
                print_data_general=self.print_data_general,
                vx_profile_prepped=vx_profile_prepped,
                ax_profile_prepped=ax_profile_prepped,
            )

        # ----------------------------------------------------------------------------------------------------------------------
        # IQP MINIMUM CURVATURE OPTIMISATION -----------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        t_start_iqp_mincurv_optimization = time.perf_counter()

        (
            iqp_track,
            alpha_iqp,
            coeffs_x_iqp,
            coeffs_y_iqp,
            M_iqp,
            normvec_norm_iqp,
            psi_iqp,
            kappa_iqp,
            dkappa_iqp,
            s_raceline_iqp,
            raceline_iqp,
            vx_profile_iqp,
            ax_profile_iqp,
            lap_time_iqp,
        ) = iqp_mincurv_handler(
            width_veh_real=self.car_width,
            width_veh_opt=self.car_width_with_margins,
            prepped_track_=prepped_track,
            normvec_norm_prepped_=normvec_normalized_prepped,
            coeffs_x_prepped_=coeffs_x_prepped,
            coeffs_y_prepped_=coeffs_y_prepped,
            M_prepped_=M_prepped,
            stepsize_iqp=self.stepsize_opt,
            kappa_bound=self.kappa_max,
            iters_min=self.min_iterations_iqp,
            iters_max=self.max_iterations_iqp,
            curv_error_allowed=self.curv_error_allowed,
            cones_left_=self.cones_left,
            cones_right_=self.cones_right,
            initial_poses_=self.reference_line,
            kappa_prepped_=kappa_prepped_track,
            dkappa_prepped_=dkappa_prepped_track,
            bound_left_=self.bound_left,
            bound_right_=self.bound_right,
            s_raceline_prepped_=s_raceline_prepped,
            calc_vel=self.vel_calc_iqp,
            car_vel_data=car_vel_data,
            print_debug=self.print_debug_opt_iqp,
            plot_debug=self.plot_opt_iqp,
            plot_data_general=self.plot_data_general,
            print_data_general=self.print_data_general,
            vx_profile_prepped=vx_profile_prepped,
            ax_profile_prepped=ax_profile_prepped,
        )

        # ----------------------------------------------------------------------------------------------------------------------
        # OPTIONAL: POSTPROCESSING TRACK AND TRAJECTORY CHECK-------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        if self.postprocessing and self.vel_calc_general:
            t_start_postprocessing = time.perf_counter()

            # Arrange data into one trajectory
            trajectory_opt = np.column_stack(
                (
                    s_raceline_iqp,
                    raceline_iqp,
                    psi_iqp,
                    kappa_iqp,
                    vx_profile_iqp,
                    ax_profile_iqp,
                )
            )
            bound1, bound2 = check_traj(
                reftrack=iqp_track,
                reftrack_normvec_normalized=normvec_norm_iqp,
                length_veh=self.car_length,
                width_veh=self.car_width,
                debug=True,
                trajectory=trajectory_opt,
                ggv=ggv,
                ax_max_machines=ax_max_motors,
                v_max=self.v_max,
                curvlim=self.kappa_max,
                mass_veh=self.car_mass,
                dragcoeff=self.drag_coeff,
            )


        if self.plot_final:
            self.bound1_imp = None
            self.bound2_imp = None

            # Loopclosure for plots
            trajectory_opt = np.vstack((trajectory_opt, trajectory_opt[0]))
            self.bound_left = np.vstack((self.bound_left, self.bound_left[0]))
            self.bound_right = np.vstack((self.bound_right, self.bound_right[0]))
            bound1 = np.vstack((bound1, bound1[0]))
            bound2 = np.vstack((bound2, bound2[0]))

            # Plot results
            self.plot_opts = {
                "mincurv_curv_lin": True,  # plot curv. linearization (original and solution based) (mincurv only)
                "raceline": True,  # plot optimized path
                "imported_bounds": False,  # plot imported bounds (analyze difference to interpolated bounds)
                "raceline_curv": True,  # plot curvature profile of optimized path
                "raceline_curv_3d": True,  # plot 3D curvature profile above raceline
                "raceline_curv_3d_stepsize": 1.0,  # [m] vertical lines stepsize in 3D curvature profile plot
                "racetraj_vel": True,  # plot velocity profile
                "racetraj_vel_3d": True,  # plot 3D velocity profile above raceline
                "racetraj_vel_3d_stepsize": 1.0,  # [m] vertical lines stepsize in 3D velocity profile plot
                "spline_normals": True,  # plot spline normals to check for crossings
                "mintime_plots": False,
            }  # plot states, controls, friction coeffs etc. (mintime only)

            result_plots(
                plot_opts=self.plot_opts,
                width_veh_opt=self.car_width_with_margins,
                width_veh_real=self.car_width,
                refline=self.reference_line[:, :2],
                bound1_imp=self.bound1_imp,
                bound2_imp=self.bound2_imp,
                bound1_interp=bound1,
                bound2_interp=bound2,
                trajectory=trajectory_opt,
                cones_left=self.cones_left,
                cones_right=self.cones_right,
                bound_left=self.bound_left,
                bound_right=self.bound_right,
            )

        path_mincurv = raceline_iqp
        path_mincurv_cl = np.vstack((path_mincurv, path_mincurv[0]))

        export_trajectory = path_mincurv_cl.tolist()

        with open("path_mincurv.json", "w") as f:
            json.dump(export_trajectory, f)

        self.calculate = False


if __name__ == "__main__":
    min_curvature = MinimumCurvature()
    min_curvature.doActivate()
    min_curvature.receive_new_path("track10_adjusted.json")
    min_curvature.receive_new_boundaries("track10_adjusted.json")
    min_curvature.active()

