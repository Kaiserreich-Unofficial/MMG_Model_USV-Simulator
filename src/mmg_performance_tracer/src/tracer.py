#!/usr/bin/env python3
import rospy
import pandas as pd
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import time

class PerformanceTracer:
    def __init__(self):
        rospy.init_node("PerformanceTracer")

        # 参数读取
        self.timeout_sec = 1.0
        self.target_topic = "/heron/target_odom"
        self.real_topic = "/heron/odom"

        self.last_target_time = None  # 不设初值
        self.target_received_once = False  # 尚未收到目标

        self.target_log = []
        self.real_log = []

        # 订阅两个 Odom 话题
        rospy.Subscriber(self.real_topic, Odometry, self.__real_cb)
        rospy.Subscriber(self.target_topic, Odometry, self.__target_cb)

        self.timer = rospy.Timer(rospy.Duration(0.1), self.__check_timeout)
        rospy.loginfo("PerformanceTracer started. Waiting for odometry messages...")

    def __real_cb(self, msg):
        state = self.__extract_state(msg)
        self.real_log.append(state)

    def __target_cb(self, msg):
        self.last_target_time = rospy.Time.now().to_sec()
        self.target_received_once = True
        state = self.__extract_state(msg)
        self.target_log.append(state)

    def __extract_state(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        linear = msg.twist.twist.linear
        angular = msg.twist.twist.angular

        _, _, psi = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        return [pos.x, pos.y, psi, linear.x, linear.y, angular.z]

    def __check_timeout(self, event):
        if not self.target_received_once:
            # 目标轨迹尚未开始发布，跳过
            return
        elapsed = rospy.Time.now().to_sec() - self.last_target_time

        if elapsed > self.timeout_sec:
            rospy.logwarn(f"超过 {self.timeout_sec} 秒未收到目标 Odom，保存数据并关闭节点")
            self.__save_logs()
            rospy.signal_shutdown("PerformanceTracer finished")

    def __save_logs(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        df_real = pd.DataFrame(self.real_log, columns=["x", "y", "psi", "u", "v", "r"])
        df_target = pd.DataFrame(self.target_log, columns=["x", "y", "psi", "u", "v", "r"])
        df_real.to_csv(f"real_odom_{timestamp}.csv", index=False)
        df_target.to_csv(f"target_odom_{timestamp}.csv", index=False)
        rospy.loginfo("数据已保存至 CSV 文件！")

if __name__ == "__main__":
    try:
        logger = PerformanceTracer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
