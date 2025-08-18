#ifndef _DYNAMICS_H_
#define _DYNAMICS_H_
#define N_STATE 6
#define N_INPUT 2
#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <fstream>

typedef Eigen::Matrix<float, N_STATE, 1> VectorSf; // 状态变量向量类型
typedef Eigen::Matrix<float, N_INPUT, 1> VectorIf; // 控制输入向量类型

namespace Simulator
{
    // 水动力参数结构体
    struct DynamicParams
    {
        float mass; // 质量
        float Iz;   // 转动惯量
        float B;    // 桨距
        float X_u_dot;
        float Y_v_dot;
        float N_r_dot;
        float X_u;
        float Y_v;
        float N_r;
    };

    class Dynamics
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // 构造函数，初始化水动力参数和积分器类型
        explicit Dynamics(const DynamicParams &dynamic_params, const std::string &integrator_type, const float &input_limit);
        // 析构函数
        ~Dynamics() = default;
        // 计算离散动力学
        VectorSf Discrete_Dynamics(const VectorSf &state, const VectorIf &input, const float &dt, const Eigen::Vector3f &extforce);

    private:
        using IntegratorFunc = VectorSf (Dynamics::*)(const VectorSf &, const VectorIf &, const float &, const Eigen::Vector3f &); // 积分器函数指针类型
        uint8_t INPUT_DIM;                                                                                                         // 输入维度
        DynamicParams params_;                                                                                                     // 水动力参数
        float input_limit_;                                                                                                        // 输入限制
        Eigen::Matrix3f M_;                                                                                                        // 惯量矩阵
        Eigen::Matrix3f D_;                                                                                                        // 阻尼矩阵
        IntegratorFunc integrator_func_;                                                                                           // 积分器函数指针
        inline VectorSf Continuous_Dynamics(const VectorSf &state, const VectorIf &input, const Eigen::Vector3f &extforce);
        inline VectorSf rk4_step(const VectorSf &state, const VectorIf &input, const float &dt, const Eigen::Vector3f &extforce);
        inline VectorSf rk3_step(const VectorSf &state, const VectorIf &input, const float &dt, const Eigen::Vector3f &extforce);
        inline VectorSf euler_step(const VectorSf &state, const VectorIf &input, const float &dt, const Eigen::Vector3f &extforce);
    };

    inline Dynamics::Dynamics(const DynamicParams &dynamic_params, const std::string &integrator_type, const float &input_limit)
    {
        // 从文件中加载参数
        this->params_ = dynamic_params;
        // 设置输入限制
        this->input_limit_ = input_limit;
        // 设置惯性矩阵
        this->M_ = (Eigen::Vector3f() << this->params_.mass - this->params_.X_u_dot, this->params_.mass - this->params_.Y_v_dot, this->params_.Iz - this->params_.N_r_dot).finished().asDiagonal();
        // 设置阻尼矩阵
        this->D_ = (Eigen::Vector3f() << this->params_.X_u, this->params_.Y_v, this->params_.N_r).finished().asDiagonal();
        // 根据积分器类型设置积分器函数
        if (integrator_type == "rk4")
        {
            this->integrator_func_ = &Dynamics::rk4_step;
        }
        else if (integrator_type == "rk3")
        {
            this->integrator_func_ = &Dynamics::rk3_step;
        }
        else if (integrator_type == "euler")
        {
            this->integrator_func_ = &Dynamics::euler_step;
        }
        else
        {
            // 如果积分器类型未知，则输出错误信息并退出程序
            ROS_ERROR("未知积分器类型: %s", integrator_type.c_str());
            exit(1);
        }
    }

    // 计算连续动力学
    inline VectorSf Dynamics::Continuous_Dynamics(const VectorSf &state, const VectorIf &input, const Eigen::Vector3f &extforce)
    {
        const float psi_ = state(2);
        const float u_ = state(3);
        const float v_ = state(4);
        const float r_ = state(5);
        // 获取输入变量
        const float Tl_ = input(0);
        const float Tr_ = input(1);
        const float cpsi_ = cosf(psi_);
        const float spsi_ = sinf(psi_);

        const Eigen::Vector3f tau = (Eigen::Vector3f() << Tl_ + Tr_, 0, 0.5 * this->params_.B * (Tl_ - Tr_)).finished() + extforce; // 构造力向量
        const Eigen::Matrix3f C_ = (Eigen::Matrix3f() << 0, 0, -(this->params_.mass - this->params_.Y_v_dot) * v_,
                                    0, 0, (this->params_.mass - this->params_.X_u_dot) * u_,
                                    (this->params_.mass - this->params_.Y_v_dot) * v_, -(this->params_.mass - this->params_.X_u_dot) * u_, 0)
                                       .finished();                                                                // 构造科氏力矩阵
        const Eigen::Vector3f nu_dot = this->M_.inverse() * (tau - C_ * state.tail(3) - this->D_ * state.tail(3)); // 计算随体速度的导数
        // 计算位置和姿态的导数
        const float x_dot = u_ * cpsi_ - v_ * spsi_;
        const float y_dot = u_ * spsi_ + v_ * cpsi_;
        const float psi_dot = r_;
        // 返回状态变量的导数
        return (VectorSf() << x_dot, y_dot, psi_dot, nu_dot(0), nu_dot(1), nu_dot(2)).finished();
    }
    // 四阶龙格-库塔法
    inline VectorSf Dynamics::rk4_step(const VectorSf &state, const VectorIf &input, const float &dt, const Eigen::Vector3f &extforce)
    {
        const VectorSf k1 = this->Continuous_Dynamics(state, input, extforce) * dt;
        const VectorSf k2 = this->Continuous_Dynamics(state + k1 / 2, input, extforce) * dt;
        const VectorSf k3 = this->Continuous_Dynamics(state + k2 / 2, input, extforce) * dt;
        const VectorSf k4 = this->Continuous_Dynamics(state + k3, input, extforce) * dt;
        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    }

    // 三阶龙格-库塔法
    inline VectorSf Dynamics::rk3_step(const VectorSf &state, const VectorIf &input, const float &dt, const Eigen::Vector3f &extforce)
    {
        const VectorSf k1 = this->Continuous_Dynamics(state, input, extforce) * dt;
        const VectorSf k2 = this->Continuous_Dynamics(state + k1 / 2, input, extforce) * dt;
        const VectorSf k3 = this->Continuous_Dynamics(state + k2, input, extforce) * dt;
        return state + (k1 + 4 * k2 + k3) / 6;
    }

    // 前向欧拉法
    inline VectorSf Dynamics::euler_step(const VectorSf &state, const VectorIf &input, const float &dt, const Eigen::Vector3f &extforce)
    {
        return state + this->Continuous_Dynamics(state, input, extforce) * dt;
    }

    // 根据输入的状态、输入和离散时间间隔，计算离散动态
    VectorSf Dynamics::Discrete_Dynamics(const VectorSf &state, const VectorIf &input, const float &dt, const Eigen::Vector3f &extforce)
    {
        if ((input.array() > this->input_limit_).any())
            ROS_WARN_THROTTLE(1.0, "输入中存在超过最大限制的值，已裁剪至 %.3f", this->input_limit_);
        else if ((input.array() < -this->input_limit_).any())
            ROS_WARN_THROTTLE(1.0, "输入中存在低于最小限制的值，已裁剪至 %.3f", -this->input_limit_);

        // 调用积分器函数，计算离散动态
        return (this->*integrator_func_)(state, input.array().min(this->input_limit_).max(-this->input_limit_), dt, extforce);
    }
}

#endif // DYNAMICS_H
