from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import vertcat
from model import MMG_Model, HydroParams
from scipy import linalg
from numpy import ndarray
import numpy as np


class Acados_Settings:
    def __init__(self, params: HydroParams, thrust_lim: float, x_init: ndarray, horizon: int, dt: float, Q: ndarray, R: ndarray):
        self.model = MMG_Model(params)
        self.acados_ocp = AcadosOcp()
        # 求解器选项
        self.acados_ocp.solver_options.N_horizon = horizon
        self.acados_ocp.solver_options.tf = horizon * dt
        self.acados_ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        self.acados_ocp.solver_options.nlp_solver_type = "SQP_RTI"
        self.acados_ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.acados_ocp.solver_options.integrator_type = "ERK"
        self.acados_ocp.solver_options.sim_method_num_stages = 4
        self.acados_ocp.solver_options.hpipm_mode = "BALANCE"
        self.acados_ocp.solver_options.qp_solver_warm_start = 1
        self.acados_ocp.solver_options.qp_tol = 1e-2

        # Model
        self.acados_ocp.model.f_expl_expr = self.model.f_expl
        self.acados_ocp.model.f_impl_expr = self.model.f_impl
        self.acados_ocp.model.x = self.model.states
        self.acados_ocp.model.xdot = self.model.states_der

        self.acados_ocp.model.u = self.model.controls
        self.acados_ocp.model.name = self.model.name

        # Cost
        self.acados_ocp.cost.cost_type = "LINEAR_LS"
        self.acados_ocp.cost.cost_type_e = "LINEAR_LS"

        nx = self.model.states.size()[0]
        nu = self.model.controls.size()[0]

        W = linalg.block_diag(Q, R)
        W_e = Q * 2

        Vx = np.zeros((nx+nu, nx))
        Vx[:nx, :nx] = np.eye(nx)

        Vx_e = np.eye(nx)

        Vu = np.zeros((nx+nu, nu))
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
        self.acados_ocp.constraints.idxbu = np.array([0, 1]) # 推力变化率约束
        self.acados_ocp.constraints.lbu = np.array([thrust_lim, -thrust_lim])
        self.acados_ocp.constraints.ubu = np.array([thrust_lim, -thrust_lim])

        self.acados_solver = AcadosOcpSolver(self.acados_ocp)
