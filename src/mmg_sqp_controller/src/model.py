from casadi import SX, vertcat, diagcat, solve, cos, sin, Function
from acados_template import AcadosOcp, AcadosOcpSolver
from numpy import ndarray
import numpy as np


class HydroParams:
    def __init__(self):
        self.mass = 38.0
        self.I_z = 6.25
        self.B = 0.74
        self.Xu_dot = -1.9
        self.Yv_dot = -29.3171
        self.Nr_dot = -4.2155
        self.Xu = 26.43
        self.Yv = 72.64
        self.Nr = 22.96


class MMG_Model:
    def __init__(self, params, dt: float, substeps: int):
        self.name = "standard_mmg_model"

        # 状态与控制
        x = SX.sym("x")
        y = SX.sym("y")
        psi = SX.sym("psi")
        u = SX.sym("u")
        v = SX.sym("v")
        r = SX.sym("r")
        Tl = SX.sym("Tl")
        Tr = SX.sym("Tr")
        self.states = vertcat(x, y, psi, u, v, r)
        self.controls = vertcat(Tl, Tr)

        # State Derivatives
        x_dot = SX.sym("x_dot")
        y_dot = SX.sym("y_dot")
        psi_dot = SX.sym("psi_dot")
        u_dot = SX.sym("u_dot")
        v_dot = SX.sym("v_dot")
        r_dot = SX.sym("r_dot")
        self.states_der = vertcat(x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot)

        # === 连续时间动力学 ===
        tau = vertcat(Tl + Tr, 0, 0.5 * params.B * (Tl - Tr))
        C_nu = SX.zeros(3, 3)
        C_nu[0, 2] = -(params.mass - params.Yv_dot) * v
        C_nu[1, 2] = (params.mass - params.Xu_dot) * u
        C_nu[2, 0] = (params.mass - params.Yv_dot) * v
        C_nu[2, 1] = -(params.mass - params.Xu_dot) * u
        M = diagcat(params.mass - params.Xu_dot,
                    params.mass - params.Yv_dot,
                    params.I_z - params.Nr_dot)
        D = diagcat(params.Xu, params.Yv, params.Nr)

        nu = vertcat(u, v, r)
        nu_dot = solve(M, tau - C_nu @ nu - D @ nu)

        self.f_expl = vertcat(
            u * cos(psi) - v * sin(psi),
            u * sin(psi) + v * cos(psi),
            r,
            nu_dot
        )


class Acados_Settings:
    def __init__(self, params: HydroParams, thrust_lim: float, x_init: ndarray, horizon: int, dt: float, Q: ndarray, R: ndarray):
        self.model = MMG_Model(params, dt, 5)
        self.acados_ocp = AcadosOcp()
        # 求解器选项
        self.acados_ocp.solver_options.N_horizon = horizon
        self.acados_ocp.solver_options.tf = horizon * dt
        self.acados_ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.acados_ocp.solver_options.nlp_solver_type = "SQP_RTI"
        self.acados_ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.acados_ocp.solver_options.integrator_type = "ERK"
        self.acados_ocp.solver_options.sim_method_num_stages = 4
        self.acados_ocp.solver_options.sim_method_num_steps = 5
        self.acados_ocp.solver_options.hpipm_mode = "BALANCE"
        self.acados_ocp.solver_options.qp_solver_warm_start = 1
        self.acados_ocp.solver_options.qp_tol = 1e-2

        # ✅ 使用离散动力学
        self.acados_ocp.model.x = self.model.states
        self.acados_ocp.model.u = self.model.controls
        self.acados_ocp.model.name = self.model.name
        self.acados_ocp.model.f_expl_expr = self.model.f_expl

        self.acados_ocp.model.xdot = self.model.states_der

        # Cost
        self.acados_ocp.cost.cost_type = "LINEAR_LS"
        self.acados_ocp.cost.cost_type_e = "LINEAR_LS"

        nx = self.model.states.size()[0]
        nu = self.model.controls.size()[0]

        W = diagcat(Q, R).full()
        W_e = Q * 2

        Vx = np.zeros((nx+nu, nx))
        Vx[:nx, :nx] = np.eye(nx)

        Vx_e = np.eye(nx)

        Vu = np.zeros((nx+nu, nu))
        # Vu[nx:, :] = np.eye(nu)  # 控制量对应 nx~nx+nu 行
        Vu[6, 0] = 1.0
        Vu[7, 1] = 1.0

        # Weights
        self.acados_ocp.cost.W = W
        self.acados_ocp.cost.W_e = W_e
        self.acados_ocp.cost.Vx = Vx
        self.acados_ocp.cost.Vu = Vu
        self.acados_ocp.cost.Vx_e = Vx_e

        # ✅ Add this to avoid W_0/yref_0 errors
        self.acados_ocp.cost.yref = np.zeros(nx + nu)
        self.acados_ocp.cost.yref_e = np.zeros(nx)

        # Constraints
        self.acados_ocp.constraints.x0 = x_init
        self.acados_ocp.constraints.idxbu = np.array([0, 1])  # 推力变化率约束
        self.acados_ocp.constraints.lbu = np.array([-thrust_lim, -thrust_lim])
        self.acados_ocp.constraints.ubu = np.array([thrust_lim, thrust_lim])

        self.acados_solver = AcadosOcpSolver(self.acados_ocp)
