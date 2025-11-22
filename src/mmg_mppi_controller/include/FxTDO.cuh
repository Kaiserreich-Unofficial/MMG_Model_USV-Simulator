#ifndef _FXTDO_CUH_
#define _FXTDO_CUH_

#include <cmath>
#include <iostream>
namespace heron
{
    struct FxTDOState
    {
        // z1_hat 存储 M * nu_hat（按你定义）
        float z1_hat[3] = {0.0f, 0.0f, 0.0f};
        // fd_hat 存扰动估计
        float fd_hat[3] = {0.0f, 0.0f, 0.0f};
    };

    class FxTDO
    {
    public:
        __host__ __device__ FxTDO()
        {
            // model params
            m = Iz = Xu_dot = Yv_dot = Nr_dot = X_u = Y_v = N_r = 0.0f;
            B = 0.0f;

            // observer gains/defaults
            L1 = L2 = 0.0f;
            k1 = k1p = k1pp = k2 = k2p = k2pp = 0.0f;
            d_inf = 0.0f;
        }

        __host__ __device__ void computeTauEff(const float nu[3], float Tl, float Tr, float tauEff[3]) const
        {
            const float u = nu[0];
            const float v = nu[1];
            const float r = nu[2];

            // 推力/控制项 tau
            const float tau0 = Tl + Tr;              // surge
            const float tau1 = 0.0f;                 // sway
            const float tau2 = 0.5f * B * (Tl - Tr); // yaw moment

            const float C0 = -(m - Yv_dot) * v * r;
            const float C1 = (m - Xu_dot) * u * r;
            const float C2 = (Yv_dot - Xu_dot) * u * v;

            const float D0 = X_u * u;
            const float D1 = Y_v * v;
            const float D2 = N_r * r;

            // tauEff = tau - C - D
            tauEff[0] = tau0 - C0 - D0;
            tauEff[1] = tau1 - C1 - D1;
            tauEff[2] = tau2 - C2 - D2;
        }

        __host__ __device__ void integrate(float dt, FxTDOState &state, const float obs_state[6], float Tl, float Tr)
        {
            float nu_obs[3] = {obs_state[3], obs_state[4], obs_state[5]};
            float z1_meas[3];
            z1_meas[0] = (m - Xu_dot) * obs_state[3];
            z1_meas[1] = (m - Yv_dot) * obs_state[4];
            z1_meas[2] = (Iz - Nr_dot) * obs_state[5];

            float T_eff[3];
            float nu_hat[3] = {state.z1_hat[0] / (m - Xu_dot), state.z1_hat[1] / (m - Yv_dot), state.z1_hat[2] / (Iz - Nr_dot)};
            computeTauEff(nu_hat, Tl, Tr, T_eff);

            auto diff = [&](const float z1_hat_in[3], const float fd_hat_in[3], const float z1_real_in[3], float dz[6])
            {
#pragma unroll
                for (int i = 0; i < 3; ++i)
                {
                    // 误差
                    const float e = z1_real_in[i] - z1_hat_in[i];
                    const float s = (e > 0) - (e < 0); // 符号函数
                    const float abs_e = fabsf(e);

                    // 非线性修正项 phi1, phi2（保持你原来的形式）
                    const float phi1 = k1 * s * powf(abs_e, 0.5f) + k1p * s * abs_e + k1pp * s * powf(abs_e, 1.0f / (1.0f - d_inf));

                    const float phi2 = k2 * s + k2p * s * abs_e + k2pp * s * powf(abs_e, (1.0f + d_inf) / (1.0f - d_inf));

                    // z1_hat_dot = fd_hat + T + L1 * phi1
                    dz[i] = fd_hat_in[i] + T_eff[i] + L1 * phi1;
                    // fd_hat_dot = L2 * phi2
                    dz[i + 3] = L2 * phi2;
                }
            };

            float k1_[6], k2_[6], k3_[6], k4_[6];

            diff(state.z1_hat, state.fd_hat, z1_meas, k1_);

            float z1_hat_temp[3], fd_hat_temp[3];
#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                z1_hat_temp[i] = state.z1_hat[i] + 0.5f * dt * k1_[i];
                fd_hat_temp[i] = state.fd_hat[i] + 0.5f * dt * k1_[i + 3];
            }

            diff(z1_hat_temp, fd_hat_temp, z1_meas, k2_);

#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                z1_hat_temp[i] = state.z1_hat[i] + 0.5f * dt * k2_[i];
                fd_hat_temp[i] = state.fd_hat[i] + 0.5f * dt * k2_[i + 3];
            }

            diff(z1_hat_temp, fd_hat_temp, z1_meas, k3_);

#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                z1_hat_temp[i] = state.z1_hat[i] + dt * k3_[i];
                fd_hat_temp[i] = state.fd_hat[i] + dt * k3_[i + 3];
            }

            diff(z1_hat_temp, fd_hat_temp, z1_meas, k4_);

#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                state.z1_hat[i] += dt / 6.0f * (k1_[i] + 2.0f * k2_[i] + 2.0f * k3_[i] + k4_[i]);
                state.fd_hat[i] += dt / 6.0f * (k1_[i + 3] + 2.0f * k2_[i + 3] + 2.0f * k3_[i + 3] + k4_[i + 3]);
            }
        }

        // ------------------
        // Observer 参数（可以在 host 端设置）
        // ------------------
        float L1, L2;
        float k1, k1p, k1pp;
        float k2, k2p, k2pp;
        float d_inf;

    public:
        // ------------------
        // 动力学参数（模型部分）
        // ------------------
        float m;      // mass
        float Iz;     // Iz (moment of inertia)
        float Xu_dot; // added mass
        float Yv_dot;
        float Nr_dot;
        float X_u;
        float Y_v;
        float N_r;
        float B; // half-track or distance param used in tau2 calc (or prop spacing)
    };
} // namespace heron

#endif // _FXTDO_CUH_
