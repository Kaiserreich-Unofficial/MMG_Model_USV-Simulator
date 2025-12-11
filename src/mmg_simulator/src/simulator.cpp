#define ANSI_RED "\033[31m"
#define ANSI_GREEN "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_BLUE "\033[34m"
#define ANSI_PURPLE "\033[35m"
#define ANSI_CYAN "\033[36m"

// ROS
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32MultiArray.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32.h>
#include <tf/transform_datatypes.h>
#include <boost/bind.hpp>
// Project
#include "dynamics.hpp"
#include "utils.hpp"

#include "extwave.hpp"
#include "wavefield.cuh"

#include "extwind.hpp"

float dt;                                                   // 仿真步长
VectorSf state_ = VectorSf::Zero();                         // 状态变量
VectorIf input_ = VectorIf::Zero();                         // 控制输入
std::shared_ptr<Simulator::Dynamics> dynamics_;             // 指向动力学模型的指针
std::shared_ptr<waveforce::WaveForceCalculator> waveforce_; // F-K 波浪力计算器
std::shared_ptr<WindForceGenerator> wind_;                  // 指向风力的指针

ros::Publisher odom_pub;        // ODOMETRY 发布器
ros::Publisher target_odom_pub; // 目标 ODOMETRY 发布器
ros::Publisher extforce_pub;    // 发布外部扰动力的真实值
bool wave_enable, wind_enable;  // 波力和风力是否启用

ros::Time last_input_time;  // 上次收到输入的时间
double input_timeout = 0.5; // 超时时间（单位：秒）

void inputCallback(const std_msgs::Float32::ConstPtr &input_val, bool is_right)
{
    last_input_time = ros::Time::now(); // 更新时间
    // 打印验证
    ROS_INFO_STREAM_THROTTLE(1.0, ANSI_PURPLE << "收到控制指令: " << input_.transpose().format(Eigen::IOFormat(2, 0, ", ", "\n", "[", "]")));
    input_(is_right) = input_val->data;
}

void odomCallback(const ros::TimerEvent &event)
{
    // 心跳检测
    double now = ros::Time::now().toSec();
    if ((now - last_input_time.toSec()) > input_timeout)
    {
        if (input_ != VectorIf::Zero())
        {
            ROS_WARN_STREAM_THROTTLE(1.0, "无输入时间达 "
                                              << (now - last_input_time.toSec()) << " 秒，输入信号已重置！");
            input_ = VectorIf::Zero(); // 重置输入
        }
    }

    static double sim_time = 0.0;
    static Eigen::Vector3f extwave = Eigen::Vector3f::Zero(); // 外部波浪力
    static Eigen::Vector3f extwind = Eigen::Vector3f::Zero(); // 外部风力
    Eigen::Vector3f extforce = extwave + extwind;
    // 发布extforce扰动力
    std_msgs::Float32MultiArray extforce_msg;
    extforce_msg.data.resize(3);
    extforce_msg.data[0] = extforce(0);
    extforce_msg.data[1] = extforce(1);
    extforce_msg.data[2] = extforce(2);
    extforce_pub.publish(extforce_msg);
    // 发布odom信息
    state_ = dynamics_->Discrete_Dynamics(state_, input_, dt, extforce); // 更新状态
    sim_time += dt;
    if (wave_enable)
    {
        auto F = waveforce_->compute_force(sim_time, state_(0), state_(1), state_(2));
        extwave << F.Fx, F.Fy, F.Mz;                                                                                                           // 计算外部波浪力
        ROS_INFO_STREAM_THROTTLE(5.0, ANSI_CYAN << "当前波浪力: " << extwave.transpose().format(Eigen::IOFormat(2, 0, ", ", "\n", "[", "]"))); // 打印验证
    }
    if (wind_enable)
    {
        wind_->update();                       // 更新随机游走
        extwind = wind_->getWindForce(state_); // 计算外部风力
        ROS_INFO_STREAM_THROTTLE(5.0, ANSI_GREEN << "当前风力: " << extwind.transpose().format(Eigen::IOFormat(2, 0, ", ", "\n", "[", "]")));
        ROS_INFO_STREAM_THROTTLE(5.0, ANSI_GREEN << "当前风速: " << wind_->getWindSpeed() << " m/s, 全局风向: " << wind_->getWindDirection() / M_PI * 180 << "°");
        ROS_INFO_STREAM_THROTTLE(2.0, ANSI_YELLOW << "当前模拟时间: " << sim_time); // 打印验证
    }
    odom_pub.publish(Simulator::state2odom(state_));
}

void targetCallback(const std_msgs::Float32MultiArray::ConstPtr &msg)
{
    if (msg->data.size() < 6)
    {
        ROS_ERROR("轨迹长度 < 6, 请检查轨迹长度!");
        return;
    }

    // 直接映射前6个元素
    Eigen::Map<const VectorSf> target_state_(msg->data.data(), 6);
    ROS_INFO_STREAM_THROTTLE(5.0, ANSI_BLUE << "当前目标状态: " << target_state_.transpose().format(Eigen::IOFormat(2, 0, ", ", "\n", "[", "]")));

    target_odom_pub.publish(Simulator::state2odom(target_state_));
}

int main(int argc, char **argv)
{
    setlocale(LC_ALL, ""); // Set locale to UTF-8
    ros::init(argc, argv, "mmg_simulator");
    ros::NodeHandle nh;

    std::string integrator_type;           // 积分器类型
    float input_limit;                     // 控制输入限制
    Simulator::DynamicParams params;       // 水动力参数
    float Length_, Width_, Draft_, Depth_; // 船长、宽、吃水、水深
    // 基本参数
    nh.param<float>("dt", dt, 0.1);
    float speed_scale; // 仿真器速度倍率
    nh.param<float>("speed_scale", speed_scale, 1.f);
    nh.param<std::string>("integrator_type", integrator_type, "rk4");
    nh.param<float>("input_limit", input_limit, 20.0);
    nh.param<float>("length", Length_, 1.3f);
    nh.param<float>("width", Width_, 0.98f);
    nh.param<float>("water_depth", Depth_, 10.f);
    nh.param<float>("draft", Draft_, 0.12f);

    // 水动力导数
    nh.param<float>("hydrodynamics/mass", params.mass, 38);            // 质量
    nh.param<float>("hydrodynamics/Iz", params.Iz, 6.25);              // 转动惯量
    nh.param<float>("hydrodynamics/B", params.B, 0.74);                // 桨距
    nh.param<float>("hydrodynamics/Xu_dot", params.X_u_dot, -1.900);   // Xu_dot
    nh.param<float>("hydrodynamics/Yv_dot", params.Y_v_dot, -29.3171); // Yv_dot
    nh.param<float>("hydrodynamics/Nr_dot", params.N_r_dot, -4.2155);  // Nr_dot
    nh.param<float>("hydrodynamics/Xu", params.X_u, 26.43);            // Xu
    nh.param<float>("hydrodynamics/Yv", params.Y_v, 72.64);            // Yv
    nh.param<float>("hydrodynamics/Nr", params.N_r, 22.96);            // Nr

    // 波浪参数
    int wave_N; // 采样点数
    int n_span, n_vert; // 船体横向和竖向分段数
    float Hs, Tp, wave_gamma, wave_L;
    nh.param<bool>("wave/enable", wave_enable, true);
    nh.param<int>("wave/N", wave_N, 128);
    nh.param<float>("wave/L", wave_L, 100.0f);
    nh.param<float>("wave/Hs", Hs, .5f);
    nh.param<float>("wave/Tp", Tp, 1.0f);
    nh.param<float>("wave/gamma", wave_gamma, 3.3f);
    nh.param<int>("wave/n_span", n_span, 10);
    nh.param<int>("wave/n_vert", n_vert, 10);

    auto wave_calc = std::make_shared<wavefield::WaveFieldCalculator>(wave_N, wave_N, wave_L, wave_L, Hs, Tp, wave_gamma, 42);
    waveforce_ = std::make_shared<waveforce::WaveForceCalculator>(
        *wave_calc, Length_, Width_, Draft_, 1025, n_span, n_vert);

    // 风参数
    float beta_w_;  // 初始风向（degree）
    float V_w_;     // 初始风速（m/s）
    float rho_air_; // 空气密度（kg/m³）
    // float Laa_;     // 风距（力矩臂长）

    nh.param<bool>("wind/enable", wind_enable, true);
    nh.param<float>("wind/direction", beta_w_, 0.0f);
    nh.param<float>("wind/speed", V_w_, 1.0f);
    nh.param<float>("wind/rho", rho_air_, 1.225f);

    wind_ = std::make_shared<WindForceGenerator>(beta_w_ / 180.f * M_PI, V_w_, rho_air_, Length_);

    // 读取话题名称
    std::string odom_topic, tgt_topic, tgt_odom_topic, left_cmd_topic, right_cmd_topic;
    nh.param<std::string>("topics/odom", odom_topic, "/heron/odom");
    nh.param<std::string>("topics/target", tgt_topic, "/heron/goal");
    nh.param<std::string>("topics/target_odom", tgt_odom_topic, "/heron/target_odom");
    nh.param<std::string>("topics/left_thruster_cmd", left_cmd_topic, "/heron/left_thruster_cmd");
    nh.param<std::string>("topics/right_thruster_cmd", right_cmd_topic, "/heron/right_thruster_cmd");

    // Initialize dynamics
    dynamics_ = std::make_shared<Simulator::Dynamics>(params, integrator_type, input_limit);

    // 输入话题订阅
    ros::Subscriber left_thruster_sub = nh.subscribe<std_msgs::Float32>(left_cmd_topic, 10, boost::bind(&inputCallback, _1, false));
    ros::Subscriber right_thruster_sub = nh.subscribe<std_msgs::Float32>(right_cmd_topic, 10, boost::bind(&inputCallback, _1, true));
    ros::Subscriber target_sub = nh.subscribe<std_msgs::Float32MultiArray>(tgt_topic, 10, &targetCallback);

    // 输出话题发布
    odom_pub = nh.advertise<nav_msgs::Odometry>(odom_topic, 10);            // 发布当前位置
    target_odom_pub = nh.advertise<nav_msgs::Odometry>(tgt_odom_topic, 10); // 发布目标位置
    extforce_pub = nh.advertise<std_msgs::Float32MultiArray>("/heron/extforce", 10);
    // 设定定时器固定频率发布odom
    ros::Timer odom_timer = nh.createTimer(ros::Duration(dt / speed_scale), &odomCallback);

    ros::spin();
    return 0;
}
