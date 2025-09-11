#ifndef _EXTWIND_HPP_
#define _EXTWIND_HPP_

#include <Eigen/Dense>
#include <cmath>
#include <random>

class WindForceGenerator
{
public:
    WindForceGenerator() = delete;
    WindForceGenerator(float beta_w_init, float V_w_init,
                       float rho_air, float Laa,
                       float beta_w_sigma = 0.009f,
                       float V_w_sigma = 0.009f,
                       unsigned int seed = ros::Time::now().toSec())
        : beta_w_(beta_w_init), V_w_(V_w_init),
          rho_air_(rho_air), Laa_(Laa),
          beta_w_sigma_(beta_w_sigma), V_w_sigma_(V_w_sigma),
          rng_(seed), norm_dist_(0.0f, 1.0f)
    {
        // 经验参数（可根据需要改成 set 函数）
        this->Afw_ = 0.202f;  // 前向受风面积
        this->Alw_ = 0.3025f; // 横向受风面积
        this->cx_ = 0.7f;     // 推进方向风力系数
        this->cy_ = 0.825f;   // 横荡方向风力系数
        this->cz_ = 0.125f;   // 偏航风矩系数
    }

    ~WindForceGenerator() = default;

    // 每一步更新风状态（使用 Random Walk 模型）
    void update()
    {
        beta_w_ += beta_w_sigma_ * norm_dist_(rng_);
        beta_w_ = fmodf(beta_w_, 2.0f * M_PI); // 防止越界
        V_w_ += V_w_sigma_ * norm_dist_(rng_);

        // 可选限制风速最小值，避免非物理负风速
        if (V_w_ < 0.1f)
            V_w_ = 0.1f;
    }

    Eigen::Vector3f getWindForce(const VectorSf &state) const
    {
        // 状态解包
        const float psi = state(2);
        const float u = state(3);
        const float v = state(4);

        // 船体坐标系下的风速分量
        const float v_wx = V_w_ * cosf(beta_w_ - psi);
        const float v_wy = V_w_ * sinf(beta_w_ - psi);

        // 相对风速（船体 - 风）
        const float v_rwx = u - v_wx;
        const float v_rwy = v - v_wy;

        const float v_rw2 = v_rwx * v_rwx + v_rwy * v_rwy;
        const float gamma_rw = -atan2f(v_rwy, v_rwx); // 相对风角（负号保持一致）

        // 风力计算
        return (Eigen::Vector3f() << -0.5f * rho_air_ * v_rw2 * cx_ * cosf(gamma_rw) * Afw_,
                0.5f * rho_air_ * v_rw2 * cy_ * sinf(gamma_rw) * Alw_,
                0.5f * rho_air_ * v_rw2 * cz_ * sinf(2 * gamma_rw) * Alw_ * Laa_)
            .finished();
    }

    // 访问当前风状态
    float getWindDirection() const { return beta_w_; }
    float getWindSpeed() const { return V_w_; }

private:
    // 风状态
    float beta_w_;  // 当前风向
    float V_w_;     // 当前风速
    float rho_air_; // 空气密度
    float Laa_;     // 偏航臂长

    // 风参数
    float Afw_, Alw_;
    float cx_, cy_, cz_;

    // 随机游走参数
    float beta_w_sigma_;
    float V_w_sigma_;
    std::default_random_engine rng_;
    std::normal_distribution<float> norm_dist_; // 正态分布
};

#endif // EXTWIND_HPP
