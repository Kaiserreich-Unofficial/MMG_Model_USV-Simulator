#define ANSI_RED "\033[31m"
#define ANSI_GREEN "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_BLUE "\033[34m"
#define ANSI_PURPLE "\033[35m"
#define ANSI_CYAN "\033[36m"

// ROS
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32MultiArray.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float32.h>
#include <tf/transform_datatypes.h>

// MPPI includes
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/controllers/ColoredMPPI/colored_mppi_controller.cuh>
#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/sampling_distributions/colored_noise/colored_noise.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>

// STD includes
#include <iostream>
#include <vector>
#include <cstring>   // for memcpy
#include <algorithm> // for fill
#include <atomic>    // for atomic_bool
#include <chrono>    // for time

// USV model
#include "usv_dynamics.cuh"
#include "usv_mpc_plant.cuh"

using DYN_T = heron::USVDynamics;
using COST_PARAM_T = QuadraticCostTrajectoryParams<DYN_T, 100>;
using COST_T = QuadraticCostTrajectory<DYN_T, 100>;
using FB_T = DDPFeedback<DYN_T, 100>;
using SAMPLER_T = mppi::sampling_distributions::ColoredNoiseDistribution<DYN_T::DYN_PARAMS_T>;
using CONTROLLER_T = ColoredMPPIController<DYN_T, COST_T, FB_T, 100, 8192, SAMPLER_T>;
using CONTROLLER_PARAMS_T = CONTROLLER_T::TEMPLATED_PARAMS;
using PLANT_T = heron::USVMPCPlant<CONTROLLER_T>;
using state_array = DYN_T::state_array;
using control_array = DYN_T::control_array;

int DYN_BLOCK_X;
constexpr uint8_t DYN_BLOCK_Y = DYN_T::STATE_DIM;
state_array observed_state = state_array::Zero();
std::shared_ptr<float[]> target_state; // 观测状态量和目标状态量

DYN_T dynamics;
COST_T cost;
COST_PARAM_T cost_params = cost.getParams();
CONTROLLER_PARAMS_T controller_params;
std::shared_ptr<CONTROLLER_T> controller; // mpc controller
std::shared_ptr<FB_T> fb_controller;      // feedback controller
std::shared_ptr<PLANT_T> plant;           // mpc plant
std::shared_ptr<SAMPLER_T> sampler;       // mppi sampler

float input_limit;     // 控制指令限制
bool use_fxtdo = true; // 是否使用固定时间扰动观测器

// 心跳保活机制
float heartbeat_duration;
double hbeat_target_time = 0.0;
bool hbeat_received_ = false;
bool continuous_hb_received_ = false; // 用于首次恢复/丢失时打印日志

// ROS控制指令发布者
ros::Publisher left_thruster_pub;
ros::Publisher right_thruster_pub;
ros::Publisher predict_traj_pub;

void publishPredictPath(
    const CONTROLLER_T::state_trajectory &traj,
    const std::string &frame_id = "odom")
{
    nav_msgs::Path path_msg;
    path_msg.header.stamp = ros::Time::now();
    path_msg.header.frame_id = frame_id;

    int n = traj.cols();
    path_msg.poses.resize(n); // 预先分配空间

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        geometry_msgs::PoseStamped pose;
        pose.header = path_msg.header;
        pose.pose.position.x = traj(0, i);
        pose.pose.position.y = traj(1, i);
        pose.pose.position.z = 0.0;
        pose.pose.orientation = tf::createQuaternionMsgFromYaw(traj(2, i));

        path_msg.poses[i] = pose; // 通过索引赋值线程安全
    }

    predict_traj_pub.publish(path_msg);
}

// Observer callback
void observer_cb(const nav_msgs::Odometry::ConstPtr &state)
{
    observed_state[0] = state->pose.pose.position.x;
    observed_state[1] = state->pose.pose.position.y;
    observed_state[2] = tf::getYaw(state->pose.pose.orientation);
    observed_state[3] = state->twist.twist.linear.x;
    observed_state[4] = state->twist.twist.linear.y;
    observed_state[5] = state->twist.twist.angular.z;
}

// Target callback
void target_cb(const std_msgs::Float32MultiArray::ConstPtr &msg)
{
    if (msg->data.size() != DYN_T::STATE_DIM * controller_params.num_timesteps_)
    {
        ROS_WARN("轨迹长度不匹配！");
        return;
    }
    memcpy(target_state.get(), msg->data.data(), sizeof(float) * DYN_T::STATE_DIM * controller_params.num_timesteps_); // 拷贝目标状态

    // 更新心跳信号
    if (!continuous_hb_received_)
    {
        ROS_INFO_STREAM("心跳信号恢复，重新启动控制器...");
        continuous_hb_received_ = true;
    }
    hbeat_received_ = true;
}

void mpc_timer_cb(const ros::TimerEvent &event_)
{
    // 心跳超时检测
    double now = ros::Time::now().toSec();
    if (now > hbeat_target_time)
    {
        if (hbeat_received_)
        {
            // 本周期收到心跳，允许输出推进命令
            hbeat_received_ = false;
            // （推进器正常由后面的 cmd 发布）
        }
        else
        {
            // 丢失心跳，停用推进器
            if (continuous_hb_received_)
            {
                ROS_WARN("心跳信号丢失 — 推进器已禁用！");
                continuous_hb_received_ = false;
            }
            std_msgs::Float32 zero;
            zero.data = 0.0;
            left_thruster_pub.publish(zero);
            right_thruster_pub.publish(zero);
            return; // 跳过下发真实命令
        }
        // 设定下一个心跳检测点
        hbeat_target_time = now + heartbeat_duration;
    }

    // 用Eigen::Map将裸指针target_state映射成矩阵，方便访问
    Eigen::Map<Eigen::Matrix<float, DYN_T::STATE_DIM, Eigen::Dynamic>> target_state_mat(
        target_state.get(),
        DYN_T::STATE_DIM,
        controller_params.num_timesteps_);
    {
        auto yaw_row = target_state_mat.row(2).array();
        Eigen::ArrayXf yaw_error = yaw_row - observed_state(2);

        // 将角度差归一到 [-π, π]
        yaw_error = (yaw_error > M_PI).select(yaw_error - 2 * M_PI, (yaw_error < -M_PI).select(yaw_error + 2 * M_PI, yaw_error));

        // 补偿后的航向角 = 当前观测航向 + 修正误差
        target_state_mat.row(2) = (observed_state(2) + yaw_error).matrix();
    }

    memcpy(cost_params.s_goal, target_state.get(), sizeof(float) * DYN_T::STATE_DIM * controller_params.num_timesteps_); // 更新目标状态
    plant->setCostParams(cost_params);
    plant->updateState(observed_state, ros::Time::now().toSec());
    if (dynamics.enable_fxtdo_)
        dynamics.fxtdo_.setObservedState(observed_state.data());
    static std::atomic<bool> alive(true); // 优化线程的存活标志
    plant->runControlIteration(&alive);
    auto fe_stat = controller->getFreeEnergyStatistics(); // 获取自由能统计信息
    ROS_INFO_STREAM(ANSI_GREEN << "平均优化时间: " << std::fixed << std::setprecision(2) << plant->getAvgOptimizationTime() << " ms, 上次优化时间: " << std::setprecision(1) << plant->getLastOptimizationTime() << " ms");
    ROS_INFO_STREAM(ANSI_CYAN << "Free Energy: " << std::fixed << std::setprecision(3) << fe_stat.real_sys.freeEnergyMean << " +- " << fe_stat.real_sys.freeEnergyVariance); // 打印优化结果
    Eigen::Vector2f ctrl = controller->getControlSeq().col(0);
    CONTROLLER_T::state_trajectory predict_traj = controller->getTargetStateSeq();
    publishPredictPath(predict_traj); // 发布预测轨迹

    std_msgs::Float32 left_cmd, right_cmd;
    left_cmd.data = ctrl(0);
    right_cmd.data = ctrl(1);
    left_thruster_pub.publish(left_cmd);
    right_thruster_pub.publish(right_cmd);
}

int main(int argc, char **argv)
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "mppi_controller");
    ros::NodeHandle nh;
    // MPC 基本参数
    nh.param<int>("horizon", controller_params.num_timesteps_, 100); // 预测域
    nh.param<float>("dt", controller_params.dt_, 0.1);               // 步长
    nh.param<float>("lambda", controller_params.lambda_, 1.0);       // 温度参数
    nh.param<float>("alpha", controller_params.alpha_, 0.0);         // 探索参数
    nh.param<int>("max_iter", controller_params.num_iters_, 1);      // 最大迭代次数
    nh.param<int>("dyn_block_size", DYN_BLOCK_X, 32);
    controller_params.dynamics_rollout_dim_ = dim3(DYN_BLOCK_X, DYN_BLOCK_Y, 1); // 动力学仿真块大小
    controller_params.cost_rollout_dim_ = dim3(DYN_BLOCK_X, DYN_BLOCK_Y, 1);     // 代价函数仿真块大小

    // 水动力参数
    heron::HydroDynamicParams hydro_params;                                  // 水动力参数
    float substep;                                                           // 积分器子步长
    nh.param<float>("input_limit", input_limit, 20.0);                       // 控制器输入限制
    nh.param<float>("substep", substep, 0.01);                               // 积分器子步长
    nh.param<float>("hydrodynamics/mass", hydro_params.mass, 38.0);          // 质量
    nh.param<float>("hydrodynamics/Iz", hydro_params.Iz, 6.25);              // 转动惯量
    nh.param<float>("hydrodynamics/B", hydro_params.B, 0.74);                // 桨距
    nh.param<float>("hydrodynamics/Xu_dot", hydro_params.X_u_dot, -1.900);   // Xu_dot
    nh.param<float>("hydrodynamics/Yv_dot", hydro_params.Y_v_dot, -29.3171); // Yv_dot
    nh.param<float>("hydrodynamics/Nr_dot", hydro_params.N_r_dot, -4.2155);  // Nr_dot
    nh.param<float>("hydrodynamics/Xu", hydro_params.X_u, 26.43);            // Xu
    nh.param<float>("hydrodynamics/Yv", hydro_params.Y_v, 72.64);            // Yv
    nh.param<float>("hydrodynamics/Nr", hydro_params.N_r, 22.96);            // Nr

    dynamics.setDynamicsParams(hydro_params, input_limit, substep); // 设置动力学参数

    // 水动力参数
    float L1, L2, k1, k1p, k1pp, k2, k2p, k2pp, d_inf;            // 固定时间扰动观测器参数
    nh.param<bool>("fxtdo/enable", dynamics.enable_fxtdo_, true); // 是否使用固定时间扰动观测器
    if (dynamics.enable_fxtdo_)
    {
        nh.param<float>("fxtdo/L1", L1, 1.0f);
        nh.param<float>("fxtdo/L2", L2, 1.0f);
        nh.param<float>("fxtdo/k1", k1, 3.0f);
        nh.param<float>("fxtdo/k1p", k1p, 1.0f);
        nh.param<float>("fxtdo/k1pp", k1pp, 0.5f);
        nh.param<float>("fxtdo/k2", k2, 2.0f);
        nh.param<float>("fxtdo/k2p", k2p, 1.0f);
        nh.param<float>("fxtdo/k2pp", k2pp, 0.2f);
        nh.param<float>("fxtdo/d_inf", d_inf, 0.3f);
        dynamics.fxtdo_.setObserverGains(L1, L2, k1, k1p, k1pp, k2, k2p, k2pp, d_inf);
    }

    // 参考轨迹初始化
    target_state = std::shared_ptr<float[]>(new float[DYN_T::STATE_DIM * controller_params.num_timesteps_]);

    // 读取心跳超时参数
    nh.param<float>("heartbeat_duration", heartbeat_duration, 0.5);
    hbeat_target_time = ros::Time::now().toSec() + heartbeat_duration;

    // 读取并设置状态权重参数
    std::vector<float> x_weight = {10, 10, 10, 10, 10, 10};
    nh.getParam("x_weight", x_weight);
    memcpy(cost_params.s_coeffs, x_weight.data(), DYN_T::STATE_DIM * sizeof(float));
    cost.setParams(cost_params); // 设置状态权重

    // 读取并设置采样器参数
    float stddev_, exponents_;
    SAMPLER_T::SAMPLING_PARAMS_T sampler_params;   // sampler需要在sampler_params构造好后再传入sampler中
    nh.param<float>("stddev", stddev_, 20.0);      // 噪声标准差
    nh.param<float>("exponents", exponents_, 1.0); // 输入关联指数
    std::fill(sampler_params.std_dev, sampler_params.std_dev + DYN_T::CONTROL_DIM, stddev_);
    std::fill(sampler_params.exponents, sampler_params.exponents + DYN_T::CONTROL_DIM, exponents_);
    sampler = std::make_shared<SAMPLER_T>(sampler_params); // 采样器实例化

    fb_controller = std::make_shared<FB_T>(&dynamics, controller_params.dt_);                                             // 反馈控制器实例化
    controller = std::make_shared<CONTROLLER_T>(&dynamics, &cost, fb_controller.get(), sampler.get(), controller_params); // MPPI控制器实例化
    plant = std::make_shared<PLANT_T>(controller, 1 / controller_params.dt_, 1);                                          // PLANT实例化

    // 读取话题名称
    std::string obs_topic, tgt_topic, left_cmd_topic, right_cmd_topic, predict_traj_topic;
    nh.param<std::string>("topics/observation", obs_topic, "/heron/odom");
    nh.param<std::string>("topics/target", tgt_topic, "/heron/goal");
    nh.param<std::string>("topics/left_thruster_cmd", left_cmd_topic, "/heron/left_thruster_cmd");
    nh.param<std::string>("topics/right_thruster_cmd", right_cmd_topic, "/heron/right_thruster_cmd");
    nh.param<std::string>("topics/predict_trajectory", predict_traj_topic, "/heron/predict_traj");

    ROS_INFO_STREAM(ANSI_PURPLE << "MPPI控制器初始化完成!");
    ROS_INFO_STREAM(ANSI_PURPLE << "预测域: " << controller_params.num_timesteps_ << ", 步长: " << std::fixed << std::setprecision(3) << controller_params.dt_ << ", 积分器子步长: " << substep << ", 控制标准差: " << stddev_ << ", FXTDO: " << std::boolalpha << dynamics.enable_fxtdo_);

    // 设置订阅
    ros::Subscriber sub_obs = nh.subscribe(obs_topic, 10, observer_cb);
    ros::Subscriber sub_tgt = nh.subscribe(tgt_topic, 10, target_cb);
    left_thruster_pub = nh.advertise<std_msgs::Float32>(left_cmd_topic, 10);
    right_thruster_pub = nh.advertise<std_msgs::Float32>(right_cmd_topic, 10);
    predict_traj_pub = nh.advertise<nav_msgs::Path>(predict_traj_topic, 1);

    // Timer for MPC at rate dt
    ros::Timer mpc_timer = nh.createTimer(ros::Duration(controller_params.dt_), &mpc_timer_cb);

    // Spin to process callbacks
    ros::spin();
    return 0;
}
