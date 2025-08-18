#include "usv_dynamics.cuh"

namespace heron
{
    // 构造函数
    USVDynamics::USVDynamics(cudaStream_t stream)
        : Dynamics<USVDynamics, USVDynamicsParams>(stream)
    {
        this->params_ = USVDynamicsParams();
    }

    void USVDynamics::setDynamicsParams(const HydroDynamicParams &hydroparams, const float &input_limit, const float &substep)
    {
        // 设置水动力参数
        this->hydroparams_ = hydroparams;
        // 设置输入限制
        this->input_limit_ = input_limit;
        // 设置积分器子步长
        this->substep_ = substep;
        const float mass = this->hydroparams_.mass;
        const float Iz = this->hydroparams_.Iz;
        const float Xu_dot = this->hydroparams_.X_u_dot;
        const float Yv_dot = this->hydroparams_.Y_v_dot;
        const float Nr_dot = this->hydroparams_.N_r_dot;
        // 设置惯性矩阵的逆
        Eigen::Matrix3f M_ = (Eigen::Vector3f() << mass - Xu_dot, mass - Yv_dot, Iz - Nr_dot).finished().asDiagonal();
        // ROS_INFO_STREAM("惯性矩阵 M = \n" << M_.format(Eigen::IOFormat(4, 0, ", ", "\n", "[", "]")));
        this->inv_M_ = M_.inverse();
        // 设置阻尼矩阵
        this->D_ = (Eigen::Vector3f() << this->hydroparams_.X_u, this->hydroparams_.Y_v, this->hydroparams_.N_r).finished().asDiagonal();
        if (this->enable_fxtdo_)
            this->fxtdo_ = FxTDO(mass, Iz, Xu_dot, Yv_dot, Nr_dot);
        // ROS_INFO_STREAM("阻尼矩阵 D = \n" << this->D_.format(Eigen::IOFormat(4, 0, ", ", "\n", "[", "]")));
    }

    // 计算动力学方程
    void USVDynamics::computeDynamics(const Eigen::Ref<const state_array> &state,
                                      const Eigen::Ref<const control_array> &control,
                                      Eigen::Ref<state_array> state_der)
    {
        // Extract state variables
        const float psi_ = state(2);    // Heading Angle
        const float u_ = state(3);      // Surge Velocity
        const float v_ = state(4);      // Sway Velocity
        const float r_ = state(5);      // Yaw Rate
        const float cpsi_ = cosf(psi_); // Cosine of Heading Angle
        const float spsi_ = sinf(psi_); // Sine of Heading Angle

        // Extract control inputs
        const float Tl_ = control(0); // Left Thruster Input
        const float Tr_ = control(1); // Right Thruster Input

        // 计算随体速度导数
        const Eigen::Vector3f tau = (Eigen::Vector3f() << Tl_ + Tr_, 0, 0.5 * this->hydroparams_.B * (Tl_ - Tr_)).finished(); // 构造推进力向量
        const Eigen::Matrix3f C_ = (Eigen::Matrix3f() << 0, 0, -(this->hydroparams_.mass - this->hydroparams_.Y_v_dot) * v_,
                                    0, 0, (this->hydroparams_.mass - this->hydroparams_.X_u_dot) * u_,
                                    (this->hydroparams_.mass - this->hydroparams_.Y_v_dot) * v_, -(this->hydroparams_.mass - this->hydroparams_.X_u_dot) * u_, 0)
                                       .finished(); // 构造科氏力矩阵

        const Eigen::Vector3f tau_eff = tau - C_ * state.tail(3) - this->D_ * state.tail(3); // 计算等效力矩
        // Compute the dynamics
        state_der(0) = cpsi_ * u_ - spsi_ * v_;
        state_der(1) = spsi_ * u_ + cpsi_ * v_;
        state_der(2) = r_;
        state_der.tail(3) = this->inv_M_ * tau_eff; // 计算速度导数

        if (this->enable_fxtdo_)
        {
            // 更新 FxTDO
            this->fxtdo_.update(tau_eff.data(), this->substep_);
            Eigen::Vector3f d_hat = this->fxtdo_.getDisturbance();

            // 叠加到状态导数
            state_der.tail(3) += this->inv_M_ * d_hat;
        }
    }

    __device__ void USVDynamics::computeDynamics(float *state,
                                                 float *control,
                                                 float *state_der,
                                                 float *theta_s)
    {
        float psi = state[2];
        float u = state[3];
        float v = state[4];
        float r = state[5];
        float cpsi = cosf(psi);
        float spsi = sinf(psi);

        float Tl = control[0];
        float Tr = control[1];

        // === 提取水动力参数 ===
        float m = this->hydroparams_.mass;
        float Iz = this->hydroparams_.Iz;
        float Xu_dot = this->hydroparams_.X_u_dot;
        float Yv_dot = this->hydroparams_.Y_v_dot;
        float Nr_dot = this->hydroparams_.N_r_dot;
        float Xu = this->hydroparams_.X_u;
        float Yv = this->hydroparams_.Y_v;
        float Nr = this->hydroparams_.N_r;
        float B = this->hydroparams_.B;

        // === 推进力 ===
        float tau0 = Tl + Tr;
        float tau1 = 0.0f;
        float tau2 = 0.5f * B * (Tl - Tr);

        // === 科氏项 ===
        float C0 = -(m - Yv_dot) * v * r;
        float C1 = (m - Xu_dot) * u * r;
        float C2 = 0.0f;

        // === 阻尼项 ===（D 是对角阵）
        float D0 = Xu * u;
        float D1 = Yv * v;
        float D2 = Nr * r;

        // === 等效力 ===
        float rhs[3] = {tau0 - C0 - D0, tau1 - C1 - D1, tau2 - C2 - D2};

        // === 计算惯性矩阵逆 ===（因为 M 是对角阵）
        float inv_M00 = 1.0f / (m - Xu_dot);
        float inv_M11 = 1.0f / (m - Yv_dot);
        float inv_M22 = 1.0f / (Iz - Nr_dot);

        // === nu_dot ===
        float nu_dot_0 = inv_M00 * rhs[0];
        float nu_dot_1 = inv_M11 * rhs[1];
        float nu_dot_2 = inv_M22 * rhs[2];

        // === 状态导数 ===
        state_der[0] = cpsi * u - spsi * v;
        state_der[1] = spsi * u + cpsi * v;
        state_der[2] = r;
        state_der[3] = nu_dot_0;
        state_der[4] = nu_dot_1;
        state_der[5] = nu_dot_2;

        if (this->enable_fxtdo_)
        {
            this->fxtdo_.update(rhs, this->substep_);
            float d_hat[3];
            fxtdo_.getDisturbance(d_hat);

            // 叠加到状态导数
            state_der[3] += inv_M00 * d_hat[0];
            state_der[4] += inv_M11 * d_hat[1];
            state_der[5] += inv_M22 * d_hat[2];
        }
    }

    void USVDynamics::step(Eigen::Ref<state_array> state,
                           Eigen::Ref<state_array> next_state,
                           Eigen::Ref<state_array> state_der,
                           const Eigen::Ref<const control_array> &control,
                           Eigen::Ref<output_array> output,
                           const float t,
                           const float dt)
    {
        const uint8_t M = static_cast<uint8_t>(dt / this->substep_ + 0.5f); // 四舍五入取整

        state_array x = state;
        state_array k1, k2, k3, k4;
        state_array temp;

        for (uint8_t i = 0; i < M; ++i)
        {
            this->computeStateDeriv(x, control, k1);
            temp = x + 0.5f * this->substep_ * k1;
            this->computeStateDeriv(temp, control, k2);
            temp = x + 0.5f * this->substep_ * k2;
            this->computeStateDeriv(temp, control, k3);
            temp = x + this->substep_ * k3;
            this->computeStateDeriv(temp, control, k4);

            x = x + (this->substep_ / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
        }

        next_state = x;
        state_der = k4; // 最后一阶段的导数
        this->stateToOutput(next_state, output);
    }

    __device__ inline void USVDynamics::step(float *state,
                                             float *next_state,
                                             float *state_der,
                                             float *control,
                                             float *output,
                                             float *theta_s,
                                             const float t,
                                             const float dt)
    {
        constexpr int N = this->STATE_DIM;
        const uint8_t M = static_cast<uint8_t>(dt / this->substep_ + 0.5f); // 四舍五入取整

        float x[N];
        for (int i = 0; i < N; ++i)
            x[i] = state[i];

        float k1[N], k2[N], k3[N], k4[N];
        float temp[N];

        for (uint8_t step = 0; step < M; ++step)
        {
            this->computeStateDeriv(x, control, k1, theta_s);
            __syncthreads();
#pragma unroll
            for (uint8_t i = 0; i < N; ++i)
                temp[i] = x[i] + 0.5f * this->substep_ * k1[i];
            this->computeStateDeriv(temp, control, k2, theta_s);
            __syncthreads();
#pragma unroll
            for (uint8_t i = 0; i < N; ++i)
                temp[i] = x[i] + 0.5f * this->substep_ * k2[i];
            this->computeStateDeriv(temp, control, k3, theta_s);
            __syncthreads();
#pragma unroll
            for (uint8_t i = 0; i < N; ++i)
                temp[i] = x[i] + this->substep_ * k3[i];
            this->computeStateDeriv(temp, control, k4, theta_s);
            __syncthreads();

#pragma unroll
            for (uint8_t i = 0; i < N; ++i)
                x[i] += (this->substep_ / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
        }
#pragma unroll
        for (uint8_t i = 0; i < N; ++i)
        {
            next_state[i] = x[i];
            state_der[i] = k4[i];
        }

        __syncthreads();
        this->stateToOutput(next_state, output);
    }

    // 从输入数据映射到状态
    Dynamics<USVDynamics, USVDynamicsParams>::state_array
    USVDynamics::stateFromMap(const std::map<std::string, float> &map)
    {
        state_array s;
        s(0) = map.at("POS_X");
        s(1) = map.at("POS_Y");
        s(2) = map.at("POS_PSI");
        s(3) = map.at("VEL_U");
        s(4) = map.at("VEL_V");
        s(5) = map.at("VEL_R");
        return s;
    }

    void USVDynamics::printState(float *state)
    {
        printf("X position: %.2f; Y position: %.2f; Heading Angle: %.2f \n", state[0], state[1], state[2]);
    }

    void USVDynamics::printState(const float *state)
    {
        printf("X position: %.2f; Y position: %.2f; Heading Angle: %.2f \n", state[0], state[1], state[2]);
    }

    // 施加控制约束（主机）
    __host__ void USVDynamics::enforceConstraints(Eigen::Ref<state_array> state, Eigen::Ref<control_array> control)
    {
        // 声明 state 未使用
        (void)state;
        control = control.cwiseMin(this->input_limit_).cwiseMax(-this->input_limit_); // 限制控制量在最大范围内
    }

    // 施加控制约束（CUDAs）
    __device__ void USVDynamics::enforceConstraints(float *state, float *control)
    {
        // TODO should control_rngs_ be a constant memory parameter
        int i, p_index, step;
        mppi::p1::getParallel1DIndex<mppi::p1::Parallel1Dir::THREAD_Y>(p_index, step);
        // parallelize setting the constraints with y dim
        for (i = p_index; i < CONTROL_DIM; i += step)
        {
            control[i] = fminf(fmaxf(-this->input_limit_, control[i]), this->input_limit_);
        }
    }
}
