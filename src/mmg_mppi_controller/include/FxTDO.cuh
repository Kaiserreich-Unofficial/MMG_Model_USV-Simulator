#ifndef _FXTDO_CUH_
#define _FXTDO_CUH_

// Fixed-Time Disturbance Observer (FxTDO) for USVDynamics
// This class estimates the 3D disturbance acting on the USV body-frame velocity (u, v, r)
#include <Eigen/Dense>
#include <cmath>

namespace heron
{
    class FxTDO
    {
    public:
        __host__ __device__ FxTDO(float m_, float Iz_) : m(m_), Iz(Iz_)
        {
#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                z1_hat[i] = 0.0f;
                fd_hat[i] = 0.0f;
            }
#pragma unroll
            for (int i = 0; i < 6; ++i)
            {
                observed_state[i] = 0.0f;
            }
        }

        __host__ __device__ FxTDO() : FxTDO(1.0f, 1.0f) {} // 委托构造，防止无法空类初始化

        __host__ __device__ void setObserverGains(float L1_, float L2_,
                                                  float k1_, float k1p_, float k1pp_,
                                                  float k2_, float k2p_, float k2pp_,
                                                  float d_inf_ = 0.3f)
        {
            L1 = L1_;
            L2 = L2_;
            k1 = k1_;
            k1p = k1p_;
            k1pp = k1pp_;
            k2 = k2_;
            k2p = k2p_;
            k2pp = k2pp_;
            d_inf = d_inf_;
        }

        __host__ __device__ void setObservedState(const float *state6)
        {
            for (int i = 0; i < 6; ++i)
                observed_state[i] = state6[i];
        }

        __host__ __device__ void update(const float *tau_eff, float dt)
        {
            float nu[3] = {observed_state[3], observed_state[4], observed_state[5]};
            float mass_diag[3] = {m, m, Iz};
            float z1_meas[3];
            float e1[3], phi1[3], phi2[3];

            for (int i = 0; i < 3; ++i)
                z1_meas[i] = mass_diag[i] * nu[i];
#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                e1[i] = z1_meas[i] - z1_hat[i];
                float ei = e1[i];
                float abs_e = fabsf(ei);
                float sgn_e = (ei >= 0.f ? 1.f : -1.f);

                phi1[i] = k1 * sgn_e * sqrtf(abs_e) + k1p * ei + k1pp * sgn_e * powf(abs_e, 1.0f / (1.0f - d_inf));
                phi2[i] = k2 * sgn_e + k2p * ei + k2pp * sgn_e * powf(abs_e, (1.0f + d_inf) / (1.0f - d_inf));
            }
#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                z1_hat[i] += dt * (fd_hat[i] + tau_eff[i] + L1 * phi1[i]);
                fd_hat[i] += dt * (L2 * phi2[i]);
            }
        }

        // 仅在 host 上可用，返回 Eigen 类型
        __host__ Eigen::Map<const Eigen::Vector3f> getDisturbance() const
        {
            return Eigen::Map<const Eigen::Vector3f>(fd_hat);
        }

        __host__ Eigen::Map<const Eigen::Vector3f> getZ1Hat() const
        {
            return Eigen::Map<const Eigen::Vector3f>(z1_hat);
        }

        __host__ Eigen::Map<const Eigen::VectorXf> getObservedState() const
        {
            return Eigen::Map<const Eigen::VectorXf>(observed_state, 6);
        }

        // device 版本（原来就有的）
        __device__ void getDisturbance(float *out) const
        {
#pragma unroll
            for (int i = 0; i < 3; ++i)
                out[i] = fd_hat[i];
        }

        __device__ void getZ1Hat(float *out) const
        {
#pragma unroll
            for (int i = 0; i < 3; ++i)
                out[i] = z1_hat[i];
        }

        __device__ float *getStatePointer()
        {
            return observed_state;
        }

    private:
        float z1_hat[3];         // Estimated momentum
        float fd_hat[3];         // Estimated disturbance force
        float observed_state[6]; // External observed state (full x, y, psi, u, v, r)

        float m, Iz; // Mass and yaw inertia

        float L1, L2, k1, k1p, k1pp, k2, k2p, k2pp, d_inf; // 固定时间扰动观测器参数
    };
} // namespace heron

#endif // FXTDO_CUH
