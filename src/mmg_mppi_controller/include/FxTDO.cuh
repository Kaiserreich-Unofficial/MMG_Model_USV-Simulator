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
             // 在 integrate() 中，先准备 M_vals:
                float M_vals[3] = {
                    (m - Xu_dot),
                    (m - Yv_dot),
                    (Iz - Nr_dot)};

            // ---- Compute measurements only once per dt ----
            float z1_meas[3] = {
                M_vals[0] * obs_state[3],
                M_vals[1] * obs_state[4],
                M_vals[2] * obs_state[5]};

            // ---- 子步数 ----
            int N = max(1, (int)roundf(dt / sub_dt));
            float h = dt / N; // 真实子步长（避免 dt/sub_dt 非整数造成漂移）

            for (int step = 0; step < N; ++step)
            {
                float nu_hat[3] = {
                    state.z1_hat[0] / M_vals[0],
                    state.z1_hat[1] / M_vals[1],
                    state.z1_hat[2] / M_vals[2]};

                float T_eff[3];
                computeTauEff(nu_hat, Tl, Tr, T_eff);



                // 替换你当前 diff 的实现为下面这个：
                auto diff = [&](const float z1_hat_in[3], const float fd_hat_in[3], float dz[6])
                {
#pragma unroll
                    for (int i = 0; i < 3; ++i)
                    {
                        const float M_i = M_vals[i];
                        // 原始误差（z 量纲）
                        float e_z = z1_meas[i] - z1_hat_in[i];
                        // 归一化误差（nu 量纲）
                        float e_nu = e_z / M_i;
                        float s = (e_nu > 0) - (e_nu < 0);
                        float abs_en = fabsf(e_nu);

                        // 计算 phi 在 nu（归一化）域的幂次，但映射回 z 域需要乘以 M 的各次幂：
                        // phi1_z = k1 * s * sqrt(|e_z|) + k1p * s * |e_z| + k1pp * s * |e_z|^{1/(1-d)}
                        // = k1 * M_i^{1/2} * s * sqrt(|e_nu|)
                        //   + k1p * M_i * s * |e_nu|
                        //   + k1pp * M_i^{1/(1-d)} * s * |e_nu|^{1/(1-d)}
                        float M_pow_half = sqrtf(M_i);
                        float M_pow_p1 = powf(M_i, 1.0f / (1.0f - d_inf)); // for k1pp term

                        float phi1_z = k1 * M_pow_half * s * sqrtf(abs_en) + k1p * M_i * s * abs_en + k1pp * M_pow_p1 * s * powf(abs_en, 1.0f / (1.0f - d_inf));

                        // phi2_z similar mapping:
                        // phi2_z = k2 * s + k2p * M_i * s * |e_nu| + k2pp * M_i^{(1+d)/(1-d)} * s * |e_nu|^{(1+d)/(1-d)}
                        float M_pow_p2 = powf(M_i, (1.0f + d_inf) / (1.0f - d_inf));
                        float phi2_z = k2 * s + k2p * M_i * s * abs_en + k2pp * M_pow_p2 * s * powf(abs_en, (1.0f + d_inf) / (1.0f - d_inf));

                        // z1_hat_dot = fd_hat + T_eff + L1 * phi1_z
                        dz[i] = fd_hat_in[i] + T_eff[i] + L1 * phi1_z;
                        // fd_hat_dot = L2 * phi2_z
                        dz[i + 3] = L2 * phi2_z;
                    }
                };

                float k1_[6], k2_[6], k3_[6], k4_[6];
                float zh[3], fh[3];

                // RK4 - k1
                diff(state.z1_hat, state.fd_hat, k1_);

                // k2
#pragma unroll
                for (int i = 0; i < 3; ++i)
                {
                    zh[i] = state.z1_hat[i] + 0.5f * h * k1_[i];
                    fh[i] = state.fd_hat[i] + 0.5f * h * k1_[i + 3];
                }
                diff(zh, fh, k2_);

                // k3
#pragma unroll
                for (int i = 0; i < 3; ++i)
                {
                    zh[i] = state.z1_hat[i] + 0.5f * h * k2_[i];
                    fh[i] = state.fd_hat[i] + 0.5f * h * k2_[i + 3];
                }
                diff(zh, fh, k3_);

                // k4
#pragma unroll
                for (int i = 0; i < 3; ++i)
                {
                    zh[i] = state.z1_hat[i] + h * k3_[i];
                    fh[i] = state.fd_hat[i] + h * k3_[i + 3];
                }
                diff(zh, fh, k4_);

                // update
#pragma unroll
                for (int i = 0; i < 3; ++i)
                {
                    state.z1_hat[i] += h / 6.0f * (k1_[i] + 2 * k2_[i] + 2 * k3_[i] + k4_[i]);
                    state.fd_hat[i] += h / 6.0f * (k1_[i + 3] + 2 * k2_[i + 3] + 2 * k3_[i + 3] + k4_[i + 3]);
                }
            }
        }

        // ------------------
        // Observer 参数（可以在 host 端设置）
        // ------------------
        float L1, L2;
        float k1, k1p, k1pp;
        float k2, k2p, k2pp;
        float d_inf;
        float sub_dt; // observer 的采样时间

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
