#ifndef _FXTDO_CUH_
#define _FXTDO_CUH_

#include <Eigen/Dense>
#include <cmath>

namespace heron
{
    class FxTDO
    {
    public:
        __host__ __device__ FxTDO()
        {
            // 船体参数默认0
            m = Iz = Xu_dot = Yv_dot = Nr_dot = 0.0f;
            fxtdo_alpha = 0.1f;

            // 每方向增益默认0
#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                z1_hat[i] = 0.0f;
                fd_hat[i] = 0.0f;
                L1[i] = L2[i] = k1[i] = k1p[i] = k1pp[i] = k2[i] = k2p[i] = k2pp[i] = d_inf[i] = 0.0f;
            }

#pragma unroll
            for (int i = 0; i < 6; ++i)
                observed_state[i] = 0.0f;
        }

        // 构造函数初始化船体参数
        __host__ __device__ FxTDO(float m_, float Iz_,
                                  float Xu_dot_, float Yv_dot_, float Nr_dot_,
                                  float alpha_ = 0.1f)
            : FxTDO() // 调用默认构造先初始化数组
        {
            m = m_;
            Iz = Iz_;
            Xu_dot = Xu_dot_;
            Yv_dot = Yv_dot_;
            Nr_dot = Nr_dot_;
            fxtdo_alpha = alpha_;
        }

        // 设置单个方向的观测器参数
        __host__ __device__ void setObserverGains(int axis, float L1_, float L2_,
                                                  float k1_, float k1p_, float k1pp_,
                                                  float k2_, float k2p_, float k2pp_,
                                                  float d_inf_ = 0.3f)
        {
            L1[axis] = L1_;
            L2[axis] = L2_;
            k1[axis] = k1_;
            k1p[axis] = k1p_;
            k1pp[axis] = k1pp_;
            k2[axis] = k2_;
            k2p[axis] = k2p_;
            k2pp[axis] = k2pp_;
            d_inf[axis] = d_inf_;
        }

        __host__ __device__ void setObservedState(const float *state6)
        {
            for (int i = 0; i < 6; ++i)
                observed_state[i] = state6[i];
        }

        __host__ __device__ void update(const float *tau_eff, float dt)
        {
            float nu[3] = {observed_state[3], observed_state[4], observed_state[5]};
            float mass_diag[3] = {m - Xu_dot, m - Yv_dot, Iz - Nr_dot};
            float z1_meas[3], e1[3], phi1[3], phi2[3];
            float fd_hat_new[3];
            float fd_hat_integral[3] = {0, 0, 0}; // initialize to zero

            for (int i = 0; i < 3; ++i)
                z1_meas[i] = mass_diag[i] * nu[i];

#pragma unroll
            for (int i = 0; i < 3; ++i)
                e1[i] = z1_meas[i] - z1_hat[i];

            float norm_e1_sq = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2];
            float norm_e1 = sqrtf(norm_e1_sq);

            if (norm_e1 > 1e-6)
            {
#pragma unroll
                for (int i = 0; i < 3; ++i)
                {
                    phi1[i] = k1[i] * e1[i] / powf(norm_e1, 0.5f) + k1p[i] * e1[i] + k1pp[i] * e1[i] * powf(norm_e1, d_inf[i] / (1.0f - d_inf[i]));
                    phi2[i] = k2[i] * e1[i] / norm_e1 + k2p[i] * e1[i] + k2pp[i] * e1[i] * powf(norm_e1, (2.0f * d_inf[i]) / (1.0f - d_inf[i]));
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
                z1_hat[i] += dt * (fd_hat[i] + tau_eff[i] + L1[i] * phi1[i]);
                fd_hat_integral[i] += dt * (z1_meas[i] - z1_hat[i]); // 积分误差
                fd_hat_new[i] = fd_hat[i] + dt * (L2[i] * phi2[i]);
                fd_hat[i] = fxtdo_alpha * fd_hat_new[i] + (1.0f - fxtdo_alpha) * fd_hat[i] + 0.01 * fd_hat_integral[i]; // 一阶DOB滤波
            }
        }

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

        // GPU 端访问
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

        __device__ void getObservedState(float *out6) const
        {
#pragma unroll
            for (int i = 0; i < 6; ++i)
                out6[i] = observed_state[i];
        }

    private:
        float z1_hat[3] = {0.0f, 0.0f, 0.0f};
        float fd_hat[3] = {0.0f, 0.0f, 0.0f};
        float observed_state[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // 船体全局参数（共用）
        float m = 0.0f;
        float Iz = 0.0f;
        float Xu_dot = 0.0f;
        float Yv_dot = 0.0f;
        float Nr_dot = 0.0f;
        float fxtdo_alpha; // FX TDO 阻尼系数

        // 每个方向独立的 FX TDO 增益参数
        float L1[3], L2[3], k1[3], k1p[3], k1pp[3], k2[3], k2p[3], k2pp[3], d_inf[3];
    };
} // namespace heron

#endif // _FXTDO_CUH_
