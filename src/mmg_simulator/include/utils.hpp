#include "dynamics.hpp"

namespace Simulator
{
    nav_msgs::Odometry state2odom(const VectorSf &state)
    {
        nav_msgs::Odometry odom;
        odom.header.stamp = ros::Time::now();
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";
        odom.pose.pose.position.x = state(0);
        odom.pose.pose.position.y = state(1);
        odom.pose.pose.position.z = 0.0;
        odom.pose.pose.orientation = tf::createQuaternionMsgFromYaw(state(2));
        odom.twist.twist.linear.x = state(3);
        odom.twist.twist.linear.y = state(4);
        odom.twist.twist.linear.z = 0.0;
        odom.twist.twist.angular.x = 0.0;
        odom.twist.twist.angular.y = 0.0;
        odom.twist.twist.angular.z = state(5);
        return odom;
    }
}
