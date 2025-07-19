from casadi import types, MX, vertcat, horzcat, diag, cos, sin, solve


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


def MMG_Model(params: HydroParams) -> types.SimpleNamespace:
    # define structs
    model = types.SimpleNamespace()

    # Model Name
    model.name = "standard_mmg_model"

    # State & Control Variables
    x = MX.sym("x")  # type: ignore
    y = MX.sym("y")  # type: ignore
    psi = MX.sym("psi")  # type: ignore
    u = MX.sym("u")  # type: ignore
    v = MX.sym("v")  # type: ignore
    r = MX.sym("r")  # type: ignore
    Tl = MX.sym("Tl")  # type: ignore
    Tr = MX.sym("Tr")  # type: ignore
    model.states = vertcat(x, y, psi, u, v, r)
    model.controls = vertcat(Tl, Tr)

    # State Derivatives
    x_dot = MX.sym("x_dot")  # type: ignore
    y_dot = MX.sym("y_dot")  # type: ignore
    psi_dot = MX.sym("psi_dot")  # type: ignore
    u_dot = MX.sym("u_dot")  # type: ignore
    v_dot = MX.sym("v_dot")  # type: ignore
    r_dot = MX.sym("r_dot")  # type: ignore
    model.states_der = vertcat(x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot)

    tau = vertcat(Tl + Tr, 0, 0.5 * params.B * (Tl - Tr))
    C = vertcat(
        horzcat(0, 0, -(params.mass - params.Yv_dot) * v),
        horzcat(0, 0, (params.mass - params.Xu_dot) * u),
        horzcat((params.mass - params.Yv_dot) * v, -(params.mass - params.Xu_dot) * u, 0)
    )
    D = diag(MX([params.Xu, params.Yv, params.Nr]))
    M = diag(MX([params.mass - params.Xu_dot, params.mass -
             params.Yv_dot, params.I_z - params.Nr_dot]))

    nu = vertcat(u, v, r)
    # 求 inv(M) * (tau - C @ nu - D @ nu)
    nu_dot = solve(M, tau - C @ nu - D @ nu)

    f_expl = vertcat(
        u*cos(psi) - v*sin(psi),
        u*sin(psi) + v*cos(psi),
        r,
        nu_dot)

    model.f_expl = f_expl
    model.f_impl = model.states_der - f_expl
    return model
