#include "usv_dynamics.cuh"
#include <cmath>
#include <cstdio>

namespace heron
{
    // 构造函数
    USVDynamics::USVDynamics(cudaStream_t stream)
        : Dynamics<USVDynamics, USVDynamicsParams>(stream),
          enable_fxtdo_(false)
    {
        this->params_ = USVDynamicsParams();
    }

    std::string USVDynamics::getDynamicsModelName() const
    {
        return "SmallYellowBoat Dynamics";
    }

    void USVDynamics::setDynamicsParams(const HydroDynamicParams &hydroparams,
                                        const float &input_limit,
                                        const float &substep)
    {
        hydroparams_ = hydroparams;
        input_limit_ = input_limit;
        substep_ = substep;

        Eigen::Matrix3f M = (Eigen::Vector3f() << hydroparams_.mass - hydroparams_.X_u_dot,
                             hydroparams_.mass - hydroparams_.Y_v_dot,
                             hydroparams_.Iz - hydroparams_.N_r_dot)
                                .finished()
                                .asDiagonal();
        inv_M_ = M.inverse();

        D_ = (Eigen::Vector3f() << hydroparams_.X_u, hydroparams_.Y_v, hydroparams_.N_r).finished().asDiagonal();
    }

    // ---------------- computeTauEff ----------------
    __host__ __device__ void USVDynamics::computeTauEff(const float *state, float Tl, float Tr, float *tau_eff) const
    {
        const float u = state[3];
        const float v = state[4];
        const float r = state[5];

        const float m = hydroparams_.mass;
        const float Iz = hydroparams_.Iz;
        const float Xu_dot = hydroparams_.X_u_dot;
        const float Yv_dot = hydroparams_.Y_v_dot;
        const float Nr_dot = hydroparams_.N_r_dot;
        const float Xu = hydroparams_.X_u;
        const float Yv = hydroparams_.Y_v;
        const float Nr = hydroparams_.N_r;
        const float B = hydroparams_.B;

        // 推进力
        const float tau0 = Tl + Tr;
        const float tau1 = 0.0f;
        const float tau2 = 0.5f * B * (Tl - Tr);

        // 科氏力
        const float C0 = -(m - Yv_dot) * v * r;
        const float C1 = (m - Xu_dot) * u * r;
        const float C2 = (Yv_dot - Xu_dot) * u * v;

        // 阻尼
        const float D0 = Xu * u;
        const float D1 = Yv * v;
        const float D2 = Nr * r;

        tau_eff[0] = tau0 - C0 - D0;
        tau_eff[1] = tau1 - C1 - D1;
        tau_eff[2] = tau2 - C2 - D2;
        if (this->enable_fxtdo_)
        {
            tau_eff[0] += this->fxtdo_state.fd_hat[0];
            tau_eff[1] += this->fxtdo_state.fd_hat[1];
            tau_eff[2] += this->fxtdo_state.fd_hat[2];
        }
    }

    // ---------------- computeDynamics (host) ----------------
    void USVDynamics::computeDynamics(const Eigen::Ref<const state_array> &state,
                                      const Eigen::Ref<const control_array> &control,
                                      Eigen::Ref<state_array> state_der,
                                      FxTDOState &fxtdo_state)
    {
        float tau_eff[3];
        computeTauEff(state.data(), control[0], control[1], tau_eff);

        state_der(0) = cosf(state(2)) * state(3) - sinf(state(2)) * state(4);
        state_der(1) = sinf(state(2)) * state(3) + cosf(state(2)) * state(4);
        state_der(2) = state(5);

        Eigen::Vector3f nu_dot;
        nu_dot(0) = inv_M_(0, 0) * tau_eff[0];
        nu_dot(1) = inv_M_(1, 1) * tau_eff[1];
        nu_dot(2) = inv_M_(2, 2) * tau_eff[2];

        state_der.tail(3) = nu_dot;
    }

    // ---------------- computeDynamics (device) ----------------
    __device__ void USVDynamics::computeDynamics(float *state, float *control, float *state_der,
                                                 FxTDOState &fxtdo_state, float *theta)
    {
        float tau_eff[3];
        computeTauEff(state, control[0], control[1], tau_eff);

        float psi = state[2];
        float u = state[3];
        float v = state[4];
        float r = state[5];

        state_der[0] = cosf(psi) * u - sinf(psi) * v;
        state_der[1] = sinf(psi) * u + cosf(psi) * v;
        state_der[2] = r;

        float inv_M00 = 1.0f / (hydroparams_.mass - hydroparams_.X_u_dot);
        float inv_M11 = 1.0f / (hydroparams_.mass - hydroparams_.Y_v_dot);
        float inv_M22 = 1.0f / (hydroparams_.Iz - hydroparams_.N_r_dot);

        state_der[3] = inv_M00 * tau_eff[0];
        state_der[4] = inv_M11 * tau_eff[1];
        state_der[5] = inv_M22 * tau_eff[2];
    }

    // ---------------- step (host) ----------------
    void USVDynamics::step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state,
                           Eigen::Ref<state_array> state_der, const Eigen::Ref<const control_array> &control,
                           Eigen::Ref<output_array> output, const float t, const float dt)
    {
        const uint8_t M = static_cast<uint8_t>(dt / substep_ + 0.5f);

        state_array x = state;
        state_array k1, k2, k3, k4, temp;

        for (uint8_t i = 0; i < M; ++i)
        {
            computeDynamics(x, control, k1, this->fxtdo_state);
            temp = x + 0.5f * substep_ * k1;
            computeDynamics(temp, control, k2, this->fxtdo_state);
            temp = x + 0.5f * substep_ * k2;
            computeDynamics(temp, control, k3, this->fxtdo_state);
            temp = x + substep_ * k3;
            computeDynamics(temp, control, k4, this->fxtdo_state);

            x += (substep_ / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
        }

        next_state = x;
        state_der = k4;
        stateToOutput(next_state, output);
    }

    // ---------------- step (device) ----------------
    __device__ inline void USVDynamics::step(float *state, float *next_state, float *state_der,
                                             float *control, float *output, float *theta_s,
                                             const float t, const float dt)
    {
        constexpr int N = static_cast<int>(USVDynamicsParams::StateIndex::NUM_STATES);
        const uint8_t M = static_cast<uint8_t>(dt / substep_ + 0.5f);

        float x[N], k1[N], k2[N], k3[N], k4[N], temp[N];
#pragma unroll
        for (int i = 0; i < N; ++i)
            x[i] = state[i];

        for (uint8_t step = 0; step < M; ++step)
        {
            computeDynamics(x, control, k1, this->fxtdo_state, theta_s);
#pragma unroll
            for (uint16_t i = 0; i < N; ++i)
                temp[i] = x[i] + 0.5f * substep_ * k1[i];
            computeDynamics(temp, control, k2, this->fxtdo_state, theta_s);
#pragma unroll
            for (uint16_t i = 0; i < N; ++i)
                temp[i] = x[i] + 0.5f * substep_ * k2[i];
            computeDynamics(temp, control, k3, this->fxtdo_state, theta_s);
#pragma unroll
            for (uint16_t i = 0; i < N; ++i)
                temp[i] = x[i] + substep_ * k3[i];
            computeDynamics(temp, control, k4, this->fxtdo_state, theta_s);

#pragma unroll
            for (uint16_t i = 0; i < N; ++i)
                x[i] += (substep_ / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
        }

#pragma unroll
        for (uint16_t i = 0; i < N; ++i)
        {
            next_state[i] = x[i];
            state_der[i] = k4[i];
        }

        stateToOutput(next_state, output);
    }

    // ---------------- enforceConstraints ----------------
    __host__ void USVDynamics::enforceConstraints(Eigen::Ref<state_array> state, Eigen::Ref<control_array> control)
    {
        (void)state;
        control = control.cwiseMin(input_limit_).cwiseMax(-input_limit_);
    }

    __device__ void USVDynamics::enforceConstraints(float *state, float *control)
    {
        (void)state;
        for (uint8_t i = 0; i < static_cast<uint8_t>(USVDynamicsParams::ControlIndex::NUM_CONTROLS); ++i)
            control[i] = fminf(fmaxf(control[i], -input_limit_), input_limit_);
    }

    // ---------------- stateFromMap ----------------
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

    // ---------------- printState ----------------
    void USVDynamics::printState(float *state)
    {
        printf("X: %.2f, Y: %.2f, PSI: %.2f\n", state[0], state[1], state[2]);
    }

    void USVDynamics::printState(const float *state)
    {
        printf("X: %.2f, Y: %.2f, PSI: %.2f\n", state[0], state[1], state[2]);
    }
}
