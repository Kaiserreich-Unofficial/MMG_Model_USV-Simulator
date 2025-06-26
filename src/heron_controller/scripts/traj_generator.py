import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler
from functools import partial

class TrajectoryGenerator:
    def __init__(self):
        rospy.init_node("TrajGenerator")
        rospy.loginfo("参考轨迹生成器初始化...")
        sim_time = rospy.get_param("simulation/time_total", 500.0)
        self.__dt= rospy.get_param("dt", 0.1)
        self.__total_steps = int(sim_time / self.__dt) # type: ignore
        traj_type = rospy.get_param("simulation/traj_type", "circle")
        if traj_type == "circle":
            # 圆轨迹参数
            self.__radius = rospy.get_param("simulation/radius", 5.0)
            self.__omega = rospy.get_param("simulation/omega", 0.2)
            rospy.loginfo("生成圆轨迹, 半径: %.1f, 角速度: %.1f", self.__radius, self.__omega)
            self.__generator = partial(self.__gen_circle, self.__total_steps, self.__dt, self.__radius, self.__omega)
        elif traj_type == "eight":
            self.__amplitude_x = rospy.get_param("simulation/amplitude_x", 3.4)
            self.__amplitude_y = rospy.get_param("simulation/amplitude_y", 4.8)
            self.__angular_x = rospy.get_param("simulation/angular_x", 0.5)
            self.__angular_y = rospy.get_param("simulation/angular_y", 0.25)
            rospy.loginfo("生成八字轨迹, X 振幅: %.1f, Y 振幅: %.1f, X 轴角速度: %.1f, Y 轴角速度: %.1f", self.__amplitude_x, self.__amplitude_y, self.__angular_x, self.__angular_y)
            self.__generator = partial(self.__gen_eight, self.__total_steps, self.__dt, self.__amplitude_x, self.__amplitude_y, self.__angular_x, self.__angular_y)
        elif traj_type == "point":
            self.__x_goal = rospy.get_param("simulation/x_goal", 10.0)
            self.__y_goal = rospy.get_param("simulation/y_goal", 10.0)
            self.__psi_goal = rospy.get_param("simulation/psi_goal", np.pi/2)
            rospy.loginfo("生成目标点, 目标位置: (%.1f, %.1f), 目标航向: %.2f", self.__x_goal, self.__y_goal, self.__psi_goal)
            self.__generator = partial(self.__gen_point, self.__total_steps, self.__dt, self.__x_goal, self.__y_goal, self.__psi_goal)
        else:
            rospy.logerr("未知的轨迹类型: %s", traj_type)
            rospy.signal_shutdown("未知的轨迹类型")

        target_topic = rospy.get_param("topics/target", "/heron/target")
        rospy.loginfo("参考轨迹发布到: %s, 总仿真时间: %.1f 秒, 时间间隔: %.1f 秒", target_topic, sim_time, self.__dt)
        self.__pub = rospy.Publisher(target_topic, Odometry, queue_size=10)

    def __gen_eight(self, total_steps, dt, amplitude_x, amplitude_y, angular_x, angular_y):
        t = np.linspace(0, (total_steps - 1) * dt, total_steps)

        # 位置
        x = amplitude_x * np.sin(angular_x * t)
        y = amplitude_y * np.cos(angular_y * t) - amplitude_y

        # 一阶导数
        x_dot = amplitude_x * angular_x * np.cos(angular_x * t)
        y_dot = -amplitude_y * angular_y * np.sin(angular_y * t)
        # 二阶导数
        x_ddot = -amplitude_x * angular_x**2 * np.sin(angular_x * t)
        y_ddot = -amplitude_y * angular_y**2 * np.cos(angular_y * t)

        # 艏向角和角速度
        psi = np.arctan2(y_dot, x_dot)
        psi_dot = (x_dot * y_ddot - y_dot * x_ddot) / (x_dot**2 + y_dot**2)

        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        u =  cos_psi * x_dot + sin_psi * y_dot
        v = -sin_psi * x_dot + cos_psi * y_dot
        r = psi_dot

        # 拼接成轨迹数组
        traj = np.stack((x, y, psi, u, v, r), axis=1)
        return traj

    def __gen_circle(self, total_steps, dt, radius, omega):
        t = np.linspace(0, (total_steps - 1) * dt, total_steps)

        # 位置
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        psi = omega * t

        # 一阶导数（速度）
        x_dot = -radius * omega * np.sin(omega * t)
        y_dot = radius * omega * np.cos(omega * t)

        # 艏向角
        psi = np.arctan2(y_dot, x_dot)

        # 角速度
        psi_dot = np.full_like(t, omega)

        # 旋转速度向量到船体坐标系下
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        u = cos_psi * x_dot + sin_psi * y_dot
        v = -sin_psi * x_dot + cos_psi * y_dot
        r = psi_dot

        # 拼接轨迹
        traj = np.stack((x, y, psi, u, v, r), axis=1)
        return traj

    def __gen_point(self, total_steps, dt, x_goal, y_goal, psi_goal):
        t = np.linspace(0, (total_steps - 1) * dt, total_steps)

        # 位置
        x = np.full_like(t, x_goal)
        y = np.full_like(t, y_goal)
        psi = np.full_like(t, psi_goal)

        # 随体速度
        u = np.full_like(t, 0)
        v = np.full_like(t, 0)
        r = np.full_like(t, 0)

        # 拼接轨迹
        traj = np.stack((x, y, psi, u, v, r), axis=1)
        return traj

    def run(self):
        rate = rospy.Rate(1 / self.__dt) # type: ignore
        traj = self.__generator()
        index = 0
        while not rospy.is_shutdown():
            # 构造 Odometry 消息
            odom = Odometry()
            odom.header.stamp    = rospy.Time.now()
            odom.header.frame_id = 'odom'
            odom.child_frame_id  = 'base_link'

            # 位姿
            odom.pose.pose.position.x = traj[index, 0]
            odom.pose.pose.position.y = traj[index, 1]
            odom.pose.pose.position.z = 0.0
            quat = quaternion_from_euler(0, 0, traj[index, 2])
            odom.pose.pose.orientation = Quaternion(*quat)

            # 速度
            odom.twist.twist.linear.x  = traj[index, 3]
            odom.twist.twist.linear.y  = traj[index, 4]
            odom.twist.twist.linear.z  = 0.0
            odom.twist.twist.angular.z = traj[index, 5]
            rospy.loginfo("x: %.2f, y: %.2f, psi: %.2f, u: %.2f, v: %.2f, r: %.2f", odom.pose.pose.position.x, odom.pose.pose.position.y, traj[index, 2], traj[index, 3], traj[index, 4], traj[index, 5])

            self.__pub.publish(odom)
            index += 1
            rate.sleep()
        rospy.loginfo("Done.")
        return

if __name__ == '__main__':
    try:
        generator = TrajectoryGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        pass
