#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from functools import partial
from time import sleep


class TrajectoryGenerator:
    def __init__(self):
        rospy.init_node("TrajGenerator")
        rospy.loginfo("参考轨迹生成器初始化...")

        sim_time = rospy.get_param("simulation/time_total", 500.0)
        self.__speedup = rospy.get_param("simulation/speed_scale", 1.0)
        self.__dt = rospy.get_param("dt", 0.1)
        self.__total_steps = int(sim_time / self.__dt)

        self.__horizon = rospy.get_param("horizon", 100)

        traj_type = rospy.get_param("simulation/traj_type", "circle")
        if traj_type == "circle":
            self.__radius = rospy.get_param("simulation/radius", 5.0)
            self.__omega = rospy.get_param("simulation/omega", 0.2)
            rospy.loginfo("生成圆轨迹, 半径: %.1f, 角速度: %.1f",
                          self.__radius, self.__omega)
            self.__generator = partial(
                self.__gen_circle, self.__total_steps, self.__dt, self.__radius, self.__omega)
        elif traj_type == "eight":
            self.__amplitude_x = rospy.get_param("simulation/amplitude_x", 3.4)
            self.__amplitude_y = rospy.get_param("simulation/amplitude_y", 4.8)
            self.__angular_x = rospy.get_param("simulation/angular_x", 0.5)
            self.__angular_y = rospy.get_param("simulation/angular_y", 0.25)
            rospy.loginfo("生成八字轨迹")
            self.__generator = partial(
                self.__gen_eight, self.__total_steps, self.__dt,
                self.__amplitude_x, self.__amplitude_y, self.__angular_x, self.__angular_y
            )
        elif traj_type == "point":
            self.__x_goal = rospy.get_param("simulation/x_goal", 10.0)
            self.__y_goal = rospy.get_param("simulation/y_goal", 10.0)
            self.__psi_goal = rospy.get_param("simulation/psi_goal", np.pi / 2)
            rospy.loginfo("生成目标点轨迹")
            self.__generator = partial(
                self.__gen_point, self.__total_steps, self.__dt,
                self.__x_goal, self.__y_goal, self.__psi_goal
            )
        else:
            rospy.logerr("未知的轨迹类型: %s", traj_type)
            rospy.signal_shutdown("未知的轨迹类型")

        target_topic = rospy.get_param("topics/target", "/heron/target")
        rospy.loginfo("参考轨迹发布到: %s", target_topic)
        self.__pub = rospy.Publisher(
            target_topic, Float32MultiArray, queue_size=10)

    def __gen_eight(self, total_steps, dt, amplitude_x, amplitude_y, angular_x, angular_y):
        t = np.linspace(0, (total_steps - 1) * dt, total_steps)
        x = amplitude_x * np.sin(angular_x * t)
        y = amplitude_y * np.cos(angular_y * t) - amplitude_y
        x_dot = amplitude_x * angular_x * np.cos(angular_x * t)
        y_dot = -amplitude_y * angular_y * np.sin(angular_y * t)
        x_ddot = -amplitude_x * angular_x**2 * np.sin(angular_x * t)
        y_ddot = -amplitude_y * angular_y**2 * np.cos(angular_y * t)

        psi = np.arctan2(y_dot, x_dot)
        psi_dot = (x_dot * y_ddot - y_dot * x_ddot) / (x_dot**2 + y_dot**2)

        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        u = cos_psi * x_dot + sin_psi * y_dot
        v = -sin_psi * x_dot + cos_psi * y_dot
        r = psi_dot

        traj = np.stack((x, y, psi, u, v, r), axis=1)
        return traj

    def __gen_circle(self, total_steps, dt, radius, omega):
        t = np.linspace(0, (total_steps - 1) * dt, total_steps)
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        psi = np.arctan2(
            radius * omega * np.cos(omega * t),
            -radius * omega * np.sin(omega * t)
        )

        x_dot = -radius * omega * np.sin(omega * t)
        y_dot = radius * omega * np.cos(omega * t)
        psi_dot = np.full_like(t, omega)

        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        u = cos_psi * x_dot + sin_psi * y_dot
        v = -sin_psi * x_dot + cos_psi * y_dot
        r = psi_dot

        traj = np.stack((x, y, psi, u, v, r), axis=1)
        return traj

    def __gen_point(self, total_steps, dt, x_goal, y_goal, psi_goal):
        t = np.linspace(0, (total_steps - 1) * dt, total_steps)
        x = np.full_like(t, x_goal)
        y = np.full_like(t, y_goal)
        psi = np.full_like(t, psi_goal)
        u = np.zeros_like(t)
        v = np.zeros_like(t)
        r = np.zeros_like(t)
        traj = np.stack((x, y, psi, u, v, r), axis=1)
        return traj

    def run(self):
        rate = rospy.Rate(self.__speedup / self.__dt)
        traj = self.__generator()
        index = 0

        while not rospy.is_shutdown():
            horizon_end = min(index + self.__horizon, self.__total_steps)
            traj_segment = traj[index:horizon_end]

            if traj_segment.shape[0] < self.__horizon:
                # 填充最后一段
                pad = np.repeat(
                    traj_segment[-1][np.newaxis, :], self.__horizon - traj_segment.shape[0], axis=0)
                traj_segment = np.vstack((traj_segment, pad))

            msg = Float32MultiArray()
            msg.data = np.round(traj_segment, 3).flatten().tolist()
            self.__pub.publish(msg)

            # rospy.loginfo("发布参考轨迹 index [%d ~ %d]", index, horizon_end - 1)
            index += 1

            if index >= self.__total_steps:
                rospy.loginfo("参考轨迹发布完毕，自动退出节点。")
                rospy.signal_shutdown("Trajectory finished.")
                break  # 可选，确保退出循环

            rate.sleep()


if __name__ == '__main__':
    sleep(2.0)
    try:
        generator = TrajectoryGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        pass
