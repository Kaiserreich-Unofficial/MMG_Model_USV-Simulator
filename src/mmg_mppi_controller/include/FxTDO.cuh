#ifndef _FXTDO_CUH_
#define _FXTDO_CUH_

#include <Eigen/Dense>
#include <cmath>

namespace heron
{
    struct FxTDOState
    {
        float z1_hat[3] = {0.0f, 0.0f, 0.0f};
        float fd_hat[3] = {0.0f, 0.0f, 0.0f};
    };

    class FxTDO
    {
    public:
        // 构造函数
        __host__ __device__ FxTDO()
        {
            m = Iz = Xu_dot = Yv_dot = Nr_dot = 0.0f;
            k1 = k1p = k1pp = k2 = k2p = k2pp = L1 = L2 = d_inf = 0.0f;
        }

        __host__ __device__ FxTDO(float m_, float Iz_,
                                  float Xu_dot_, float Yv_dot_, float Nr_dot_)
            : FxTDO()
        {
            m = m_;
            Iz = Iz_;
            Xu_dot = Xu_dot_;
            Yv_dot = Yv_dot_;
            Nr_dot = Nr_dot_;
        }

        // 根据新的 tau_eff 和观测状态手动积分
        __host__ __device__ void integrate(const float *tau_eff, float dt, FxTDOState &state, const float obs_state[6])
        {
            // 计算 z1_meas
            float nu[3] = {obs_state[3], obs_state[4], obs_state[5]};
            float mass_diag[3] = {m - Xu_dot, m - Yv_dot, Iz - Nr_dot};
            float z1_meas[3], e1[3], phi1[3], phi2[3];

#pragma unroll
            for (int i = 0; i < 3; ++i)
                z1_meas[i] = mass_diag[i] * nu[i];

#pragma unroll
            for (int i = 0; i < 3; ++i)
                e1[i] = z1_meas[i] - state.z1_hat[i];

            float norm_e1_sq = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2];
            float norm_e1 = sqrtf(norm_e1_sq);

            if (norm_e1 > 1e-6f)
            {
#pragma unroll
                for (int i = 0; i < 3; ++i)
                {
                    phi1[i] = k1 * e1[i] / powf(norm_e1, 0.5f) + k1p * e1[i] + k1pp * e1[i] * powf(norm_e1, d_inf / (1.0f - d_inf));
                    phi2[i] = k2 * e1[i] / norm_e1 + k2p * e1[i] + k2pp * e1[i] * powf(norm_e1, (2.0f * d_inf) / (1.0f - d_inf));
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < 3; ++i)
                    phi1[i] = phi2[i] = 0.0f;
            }

#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                state.z1_hat[i] += dt * (state.fd_hat[i] + tau_eff[i] + L1 * phi1[i]);
                state.fd_hat[i] += dt * (L2 * phi2[i]);
            }
        }

        // 参数
        float L1, L2;
        float k1, k1p, k1pp;
        float k2, k2p, k2pp;
        float d_inf;
        float observed_state[6]; // host 设置观测状态

    private:
        float m, Iz, Xu_dot, Yv_dot, Nr_dot;
    };
} // namespace heron

#endif // _FXTDO_CUH_
