// ROS
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32.h>
#include <tf/transform_datatypes.h>
#include <boost/bind.hpp>
// Project
#include "dynamics.hpp"
#include "utils.hpp"

float dt;                                       // 仿真步长
VectorSf state_ = VectorSf::Zero();             // 状态变量
VectorIf input_ = VectorIf::Zero();             // 控制输入
std::shared_ptr<Simulator::Dynamics> dynamics_; // 指向动力学模型的指针

ros::Publisher odom_pub; // ODOMETRY 发布器

void inputCallback(const std_msgs::Float32::ConstPtr &input_val, bool is_right)
{
    // 打印验证
    ROS_INFO_STREAM("Received input: " << input_.transpose().format(Eigen::IOFormat(2, 0, ", ", "\n", "[", "]")));
    input_(is_right) = input_val->data;
}

void odomCallback(const ros::TimerEvent &event)
{
    // 发布里程计信息
    state_ = dynamics_->Discrete_Dynamics(state_, input_, dt); // 更新状态
    odom_pub.publish(Simulator::state2odom(state_));
}

int main(int argc, char **argv)
{
    setlocale(LC_ALL, ""); // Set locale to UTF-8
    ros::init(argc, argv, "heron_simulator");
    ros::NodeHandle nh;

    static std::string integrator_type;     // 积分器类型
    static float input_limit;               // 控制输入限制
    static Simulator::DynamicParams params; // 水动力参数
    // Load parameters
    nh.param<float>("dt", dt, 0.1);
    nh.param<std::string>("integrator_type", integrator_type, "rk4");
    nh.param<float>("input_limit", input_limit, 20.0);
    nh.param<float>("hydrodynamics/mass", params.mass, 38);            // 质量
    nh.param<float>("hydrodynamics/Iz", params.Iz, 6.25);              // 转动惯量
    nh.param<float>("hydrodynamics/B", params.B, 0.74);                // 桨距
    nh.param<float>("hydrodynamics/Xu_dot", params.X_u_dot, -1.900);   // Xu_dot
    nh.param<float>("hydrodynamics/Yv_dot", params.Y_v_dot, -29.3171); // Yv_dot
    nh.param<float>("hydrodynamics/Nr_dot", params.N_r_dot, -4.2155);  // Nr_dot
    nh.param<float>("hydrodynamics/Xu", params.X_u, 26.43);            // Xu
    nh.param<float>("hydrodynamics/Yv", params.Y_v, 72.64);            // Yv
    nh.param<float>("hydrodynamics/Nr", params.N_r, 22.96);            // Nr

    // 读取话题名称
    static std::string odom_topic, left_cmd_topic, right_cmd_topic;
    nh.param<std::string>("topics/odom", odom_topic, "/heron/odom");
    nh.param<std::string>("topics/left_thruster_cmd", left_cmd_topic, "/heron/left_thruster_cmd");
    nh.param<std::string>("topics/right_thruster_cmd", right_cmd_topic, "/heron/right_thruster_cmd");

    // Initialize dynamics
    dynamics_ = std::make_shared<Simulator::Dynamics>(params, integrator_type, input_limit);

    // 输入话题订阅
    ros::Subscriber left_thruster_pub = nh.subscribe<std_msgs::Float32>(left_cmd_topic, 10, boost::bind(&inputCallback, _1, false));
    ros::Subscriber right_thruster_pub = nh.subscribe<std_msgs::Float32>(right_cmd_topic, 10, boost::bind(&inputCallback, _1, true));

    // 输出话题发布
    odom_pub = nh.advertise<nav_msgs::Odometry>(odom_topic, 10);
    // 设定定时器固定频率发布odom
    ros::Timer odom_timer = nh.createTimer(ros::Duration(dt), &odomCallback);

    ros::spin();
    return 0;
}
