// ROS
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32.h>
#include <tf/transform_datatypes.h>

// Math-related
#include <math.h>
#include <stdio.h>
// Containers
#include <string>

// USV model
#include "usv_dynamics.cuh"

// 检测到ctrl + c信号，退出程序
#include <signal.h>

using namespace std;

using DYN_T = heron::USVDynamics;

using state_array = DYN_T::state_array;
using control_array = DYN_T::control_array;
using output_array = DYN_T::output_array;

// Globals
float dt;
float u_left;
float u_right;

state_array observed_state;

ros::Publisher pub_left;
ros::Publisher pub_right;
ros::Publisher pub_target;

shared_ptr<DYN_T> dynamics;

// Observer callback
void observer_cb(const nav_msgs::Odometry &state)
{
    observed_state[0] = state.pose.pose.position.x;
    observed_state[1] = state.pose.pose.position.y;
    observed_state[2] = tf::getYaw(state.pose.pose.orientation);
    observed_state[3] = state.twist.twist.linear.x;
    observed_state[4] = state.twist.twist.linear.y;
    observed_state[5] = state.twist.twist.angular.z;
}

// Timer callback: main MPC loop
void dyn_timer_cb(const ros::TimerEvent &event)
{
    static state_array tgt_state = observed_state; // Initial target state is the observed state
    state_array x_next, x_dot;
    output_array y;

    dynamics->step(tgt_state, x_next, x_dot, (control_array() << u_left, u_right).finished(), y, ros::Time::now().toSec(), dt);
    nav_msgs::Odometry tgt_msgs;

    tgt_msgs.header.stamp = ros::Time::now();
    tgt_msgs.header.frame_id = "odom";      // parent frame
    tgt_msgs.child_frame_id = "base_link"; // child frame

    tgt_msgs.pose.pose.position.x = x_next[0];
    tgt_msgs.pose.pose.position.y = x_next[1];
    tgt_msgs.pose.pose.orientation = tf::createQuaternionMsgFromYaw(x_next[2]);
    tgt_msgs.twist.twist.linear.x = x_next[3];
    tgt_msgs.twist.twist.linear.y = x_next[4];
    tgt_msgs.twist.twist.angular.z = x_next[5];
    pub_target.publish(tgt_msgs);

    tgt_state = x_next;
    ROS_INFO_STREAM("Publish Thrust Command:" << fixed << setprecision(2) << u_left << " " << u_right);
    // ROS_INFO_STREAM("Target State: " << fixed << setprecision(2) << tgt_state.transpose());
    std_msgs::Float32 left_msg, right_msg;
    left_msg.data = u_left;
    right_msg.data = u_right;
    pub_left.publish(left_msg);
    pub_right.publish(right_msg);
}

void mySigintHandler(int sig)
{
    ROS_WARN("程序终止...");
    ros::shutdown(); // 通知 ROS 安全终止
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "dynamics_test", ros::init_options::AnonymousName);
    ros::NodeHandle nh;

    // Load parameters
    nh.param<float>("dt", dt, 0.1);
    nh.param<float>("dynamics_test/left_input", u_left, 20);
    nh.param<float>("dynamics_test/right_input", u_right, 20);

    // 水动力参数
    heron::HydroDynamicParams hydro_params;                                  // 水动力参数
    float input_limit, substep;                                              // 控制器输入限制 & 积分器子步长
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

    dynamics->setDynamicsParams(hydro_params, input_limit, substep); // 设置动力学参数

    // 读取话题名称
    std::string obs_topic, tgt_topic, left_thrust_topic, right_thrust_topic;
    nh.param<std::string>("topics/odom", obs_topic, "/heron/odom");
    nh.param<std::string>("topics/target", tgt_topic, "/heron/goal");
    nh.param<std::string>("topics/left_thruster_cmd", left_thrust_topic, "/heron/left_thruster_cmd");
    nh.param<std::string>("topics/right_thruster_cmd", right_thrust_topic, "/heron/right_thruster_cmd");

    ROS_INFO_STREAM("动力学测试初始化完成!");
    ROS_INFO_STREAM("模型名称: " << dynamics->getDynamicsModelName().c_str() << ", 推力设置: " << fixed << setprecision(2) << u_left << " " << u_right);

    // 设置订阅和发布
    ros::Subscriber sub_obs = nh.subscribe(obs_topic, 1, observer_cb);
    // ros::Subscriber sub_tgt = nh.subscribe(tgt_topic, 1, target_cb);
    pub_target = nh.advertise<nav_msgs::Odometry>(tgt_topic, 100);
    pub_left = nh.advertise<std_msgs::Float32>(left_thrust_topic, 100);
    pub_right = nh.advertise<std_msgs::Float32>(right_thrust_topic, 100);

    // Timer for MPC at rate dt
    ros::Timer dyn_test_timer = nh.createTimer(ros::Duration(dt), &dyn_timer_cb);
    signal(SIGINT, mySigintHandler); // 注册自定义SIGINT处理器

    // Spin to process callbacks
    ros::spin();
    return 0;
}
