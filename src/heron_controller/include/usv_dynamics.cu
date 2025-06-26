#include "usv_dynamics.cuh"

namespace heron
{
    // 构造函数
    USVDynamics::USVDynamics(cudaStream_t stream)
        : Dynamics<USVDynamics, USVDynamicsParams>(stream)
    {
        this->params_ = USVDynamicsParams();
    }

    void USVDynamics::setDynamicsParams(const HydroDynamicParams &hydroparams, const float &input_limit)
    {
        // 设置水动力参数
        this->hydroparams_ = hydroparams;
        // ROS_INFO_STREAM("Mass: " << this->hydroparams_.mass << " Iz: " << this->hydroparams_.Iz);
        // ROS_INFO_STREAM("X_u_dot: " << this->hydroparams_.X_u_dot << " Y_v_dot: " << this->hydroparams_.Y_v_dot << " N_r_dot: " << this->hydroparams_.N_r_dot);
        // ROS_INFO_STREAM("X_u: " << this->hydroparams_.X_u << " Y_v: " << this->hydroparams_.Y_v << " N_r: " << this->hydroparams_.N_r);
        // 设置输入限制
        this->input_limit_ = input_limit;
        // 设置惯性矩阵的逆
        Eigen::Matrix3f M_ = (Eigen::Vector3f() << this->hydroparams_.mass - this->hydroparams_.X_u_dot, this->hydroparams_.mass - this->hydroparams_.Y_v_dot, this->hydroparams_.Iz - this->hydroparams_.N_r_dot).finished().asDiagonal();
        // ROS_INFO_STREAM("惯性矩阵 M = \n" << M_.format(Eigen::IOFormat(4, 0, ", ", "\n", "[", "]")));
        this->inv_M_ = M_.inverse();
        // 设置阻尼矩阵
        this->D_ = (Eigen::Vector3f() << this->hydroparams_.X_u, this->hydroparams_.Y_v, this->hydroparams_.N_r).finished().asDiagonal();
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

        // Compute the dynamics
        state_der(0) = cpsi_ * u_ - spsi_ * v_;
        state_der(1) = spsi_ * u_ + cpsi_ * v_;
        state_der(2) = r_;
        state_der.tail(3) = this->inv_M_ * (tau - C_ * state.tail(3) - this->D_ * state.tail(3)); // 计算随体速度的导数
    }

    // 连续动力学方程（CUDA设备）
    __device__ void USVDynamics::computeDynamics(float *state, float *control, float *state_der,
                                                 float *theta_s)
    {
        // Extract state variables
        float psi = state[2];   // Heading Angle
        float cpsi = cosf(psi); // Cosine of Heading Angle
        float spsi = sinf(psi); // Sine of Heading Angle
        float u_ = state[3];     // Surge Velocity
        float v_ = state[4];     // Sway Velocity
        float r_ = state[5];     // Yaw Rate
        // Extract control inputs
        float Tl_ = control[0]; // Left Thruster Input
        float Tr_ = control[1]; // Right Thruster Input

        // 计算随体速度导数
        const Eigen::Vector3f nu = (Eigen::Vector3f() << u_, v_, r_).finished();
        const Eigen::Vector3f tau = (Eigen::Vector3f() << Tl_ + Tr_, 0, 0.5 * this->hydroparams_.B * (Tl_ - Tr_)).finished(); // 构造推进力向量
        const Eigen::Matrix3f C_ = (Eigen::Matrix3f() << 0, 0, -(this->hydroparams_.mass - this->hydroparams_.Y_v_dot) * v_,
                                    0, 0, (this->hydroparams_.mass - this->hydroparams_.X_u_dot) * u_,
                                    (this->hydroparams_.mass - this->hydroparams_.Y_v_dot) * v_, -(this->hydroparams_.mass - this->hydroparams_.X_u_dot) * u_, 0)
                                       .finished(); // 构造科氏力矩阵
        const Eigen::Vector3f nu_dot = this->inv_M_ * (tau - C_ * nu - this->D_ * nu); // 计算随体速度的导数

        // Compute the dynamics
        state_der[0] = cpsi * u_ - spsi * v_;
        state_der[1] = spsi * u_ + cpsi * v_;
        state_der[2] = r_;
        state_der[3] = nu_dot(0);
        state_der[4] = nu_dot(1);
        state_der[5] = nu_dot(2);
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
