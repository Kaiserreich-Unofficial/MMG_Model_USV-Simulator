#!/usr/bin/env python3
import os
import sys

# 将当前文件所在目录加入 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from model import HydroParams, Acados_Settings
from std_msgs.msg import Float32, Float32MultiArray
from nav_msgs.msg import Odometry, Path
import rospy
import numpy as np

class MMGMPCController:
    def __init__(self):
        rospy.init_node('MPCController', anonymous=True)
        rospy.loginfo("Acados 求解器初始化...")
        hydro_params = HydroParams()
        hydro_params.mass = rospy.get_param("/hydrodynamics/mass", 38.0)
        hydro_params.I_z = rospy.get_param("/hydrodynamics/I_z", 6.25)
        hydro_params.B = rospy.get_param("/hydrodynamics/B", 0.74)
        hydro_params.Xu_dot = rospy.get_param("/hydrodynamics/Xu_dot", -1.9)
        hydro_params.Yv_dot = rospy.get_param(
            "/hydrodynamics/Yv_dot", -29.3171)
        hydro_params.Nr_dot = rospy.get_param("/hydrodynamics/Nr_dot", -4.2155)
        hydro_params.Xu = rospy.get_param("/hydrodynamics/Xu", 26.43)
        hydro_params.Yv = rospy.get_param("/hydrodynamics/Yv", 72.64)
        hydro_params.Nr = rospy.get_param("/hydrodynamics/Nr", 22.96)
        input_limit: float = rospy.get_param("input_limit", 20.0)
        self.horizon: int = rospy.get_param("horizon", 100)
        self.dt: float = rospy.get_param("dt", 0.1)
        self.__speedup: float = rospy.get_param("simulation/speed_scale", 1.0)
        Q_list: list[float] = rospy.get_param(
            "x_weight", [1.0, 1.0, 0.5, 0.0, 0.0, 0.0])
        R_list: list[float] = rospy.get_param("u_weight", [0.1, 0.1])
        Q = np.diag(Q_list)
        R = np.diag(R_list)
        self.state: np.ndarray = np.zeros(6)
        self.__settings = Acados_Settings(
            hydro_params, input_limit, self.state, self.horizon, self.dt, Q, R)
        self.__solver = self.__settings.acados_solver
        self.__latest_traj: np.ndarray = None  # 目标状态轨迹
        # === 从 ROS param 获取话题名 ===
        obs_topic = rospy.get_param("topics/observation", "/heron/odom")
        traj_topic = rospy.get_param("topics/target", "/heron/goal")
        left_topic = rospy.get_param(
            "topics/left_thruster_cmd", "/heron/left_thruster_cmd")
        right_topic = rospy.get_param(
            "topics/right_thruster_cmd", "/heron/right_thruster_cmd")
        path_topic = rospy.get_param(
            "topics/predict_trajectory", "/heron/predict_traj")

        # === ROS 通信 ===
        rospy.Subscriber(obs_topic, Odometry, self.__odom_cb)
        rospy.Subscriber(traj_topic, Float32MultiArray, self.__traj_cb)
        self.pub_left = rospy.Publisher(left_topic, Float32, queue_size=10)
        self.pub_right = rospy.Publisher(right_topic, Float32, queue_size=10)
        self.pub_path = rospy.Publisher(path_topic, Path, queue_size=1)

    def __odom_cb(self, msg: Odometry) -> None:
        # 更新观测状态
        self.state[0] = msg.pose.pose.position.x
        self.state[1] = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y,
                      orientation_q.z, orientation_q.w]
        self.state[2] = euler_from_quaternion(quaternion)[2]
        self.state[3] = msg.twist.twist.linear.x
        self.state[4] = msg.twist.twist.linear.y
        self.state[5] = msg.twist.twist.angular.z

    def __traj_cb(self, msg: Float32MultiArray) -> None:
        dim_state = 6
        horizon = self.horizon

        expected_len = dim_state * horizon
        data = msg.data

        if len(data) != expected_len:
            rospy.logerr(
                f"目标状态轨迹长度不匹配: expected {expected_len}, got {len(data)}")
            return
        else:
            self.__latest_traj = np.array(data).reshape(horizon, dim_state)  # shape=(horizon, 6)

    def run(self):
        import time
        rate = rospy.Rate(self.__speedup / self.dt)  # 控制频率（Hz）

        while not rospy.is_shutdown():
            if self.__latest_traj is None:
                rospy.logwarn_throttle(5.0, "等待目标轨迹...")
                rate.sleep()
                continue

            # 设置初始状态约束
            self.__solver.set(0, "lbx", self.state)
            self.__solver.set(0, "ubx", self.state)

            # 设置参考轨迹
            for j in range(self.horizon):
                if j < self.__latest_traj.shape[0]:  # 注意改成shape[0]
                    yref = self.__latest_traj[j, :]
                else:
                    yref = self.__latest_traj[-1, :]
                self.__solver.set(j, "yref", np.concatenate([yref, np.zeros(2)]))

            self.__solver.set(self.horizon, "yref", self.__latest_traj[-1, :])

            # MPC求解器调用
            start_time = time.time()  # 开始计时
            status = self.__solver.solve()
            solve_time_ms = (time.time() - start_time) * 1000  # 单位：毫秒
            if status != 0:
                rospy.logwarn(f"Acados MPC 求解失败, status={status}")
                rate.sleep()
                continue
            # 记录并输出时间
            if not hasattr(self, '_solve_times'):
                self._solve_times = []

            self._solve_times.append(solve_time_ms)
            if len(self._solve_times) > 100:
                self._solve_times.pop(0)  # 只保留最近100次

            avg_time = sum(self._solve_times) / len(self._solve_times)

            # ANSI 颜色定义
            ANSI_GREEN = "\033[92m"
            ANSI_RESET = "\033[0m"

            rospy.loginfo(f"{ANSI_GREEN}平均优化时间: {avg_time:.2f} ms, 上次优化时间: {solve_time_ms:.1f} ms{ANSI_RESET}")

            # 发布控制输入
            u0 = self.__solver.get(0, "u")  # [left, right]
            rospy.loginfo(f"控制输入: {u0}")
            self.pub_left.publish(Float32(u0[0]))
            self.pub_right.publish(Float32(u0[1]))

            # 发布预测路径
            self.__publish_predicted_path()

            rate.sleep()

    def __publish_predicted_path(self) -> None:
        from geometry_msgs.msg import PoseStamped
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "odom"

        for i in range(self.horizon + 1):
            x_pred = self.__solver.get(i, "x")  # x = [x, y, psi, u, v, r]

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x_pred[0]
            pose.pose.position.y = x_pred[1]
            pose.pose.position.z = 0.0

            # 使用 psi 作为航向角
            q = quaternion_from_euler(0, 0, x_pred[2])
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]

            path_msg.poses.append(pose)

        self.pub_path.publish(path_msg)


if __name__ == "__main__":
    controller = MMGMPCController()
    controller.run()
