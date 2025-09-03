#!/usr/bin/env python3
import rospy
import pandas as pd
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from tf.transformations import euler_from_quaternion
import time

class PerformanceTracer:
    def __init__(self):
        rospy.init_node("PerformanceTracer")

        # 参数读取
        self.timeout_sec = 1.0
        self.target_topic = "/heron/target_odom"
        self.real_topic = "/heron/odom"
        self.left_thrust_topic = "/heron/left_thruster_cmd"
        self.right_thrust_topic = "/heron/right_thruster_cmd"

        self.last_target_time = None
        self.target_received_once = False

        self.target_log = []
        self.real_log = []

        self.latest_left_thrust = 0.0
        self.latest_right_thrust = 0.0

        # 订阅 Odom
        rospy.Subscriber(self.real_topic, Odometry, self.__real_cb)
        rospy.Subscriber(self.target_topic, Odometry, self.__target_cb)

        # 订阅推力
        rospy.Subscriber(self.left_thrust_topic, Float32, self.__left_thrust_cb)
        rospy.Subscriber(self.right_thrust_topic, Float32, self.__right_thrust_cb)

        self.timer = rospy.Timer(rospy.Duration(0.1), self.__check_timeout)

        rospy.loginfo("PerformanceTracer started. Waiting for odometry and thrust messages...")

    def __real_cb(self, msg):
        if not self.target_received_once:
            return  # 跳过记录
        state = self.__extract_state(msg)
        # 追加推进器命令
        state.extend([self.latest_left_thrust, self.latest_right_thrust])
        self.real_log.append(state)

    def __target_cb(self, msg):
        self.last_target_time = rospy.Time.now().to_sec()
        self.target_received_once = True
        state = self.__extract_state(msg)
        self.target_log.append(state)

    def __left_thrust_cb(self, msg):
        self.latest_left_thrust = msg.data

    def __right_thrust_cb(self, msg):
        self.latest_right_thrust = msg.data

    def __extract_state(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        linear = msg.twist.twist.linear
        angular = msg.twist.twist.angular

        _, _, psi = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        return [pos.x, pos.y, psi, linear.x, linear.y, angular.z]

    def __check_timeout(self, event):
        if not self.target_received_once:
            return
        elapsed = rospy.Time.now().to_sec() - self.last_target_time

        if elapsed > self.timeout_sec:
            rospy.logwarn(f"超过 {self.timeout_sec} 秒未收到目标 Odom，保存数据并关闭节点")
            self.__save_logs()
            rospy.signal_shutdown("PerformanceTracer finished")
    def __save_logs(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        expected_real_len = 8  # x, y, psi, u, v, r, left_thrust, right_thrust
        expected_target_len = 6  # x, y, psi, u, v, r

        # 筛选长度正确的行
        filtered_real = [row for row in self.real_log if len(row) == expected_real_len]
        filtered_target = [row for row in self.target_log if len(row) == expected_target_len]

        if len(filtered_real) != len(self.real_log):
            rospy.logwarn(f"裁剪掉 {len(self.real_log) - len(filtered_real)} 行 real_log，因为长度不匹配")
        if len(filtered_target) != len(self.target_log):
            rospy.logwarn(f"裁剪掉 {len(self.target_log) - len(filtered_target)} 行 target_log，因为长度不匹配")

        # 转为 DataFrame
        df_real = pd.DataFrame(filtered_real, columns=[
            "x", "y", "psi", "u", "v", "r", "left_thrust", "right_thrust"
        ])
        df_target = pd.DataFrame(filtered_target, columns=[
            "x", "y", "psi", "u", "v", "r"
        ])

        # 对齐行数
        min_rows = min(len(df_real), len(df_target))
        if len(df_real) != len(df_target):
            rospy.logwarn(f"df_real 行数 {len(df_real)} 与 df_target 行数 {len(df_target)} 不匹配，裁剪至 {min_rows} 行")
            df_real = df_real.iloc[:min_rows]
            df_target = df_target.iloc[:min_rows]

        # 保存
        df_real.to_csv(f"real_odom_{timestamp}.csv", index=False)
        df_target.to_csv(f"target_odom_{timestamp}.csv", index=False)
        rospy.loginfo("所有数据已保存至 CSV 文件！")

if __name__ == "__main__":
    try:
        logger = PerformanceTracer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
