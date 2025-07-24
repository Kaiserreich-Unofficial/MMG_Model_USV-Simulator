#!/usr/bin/env python3
import numpy as np
import rospy
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from model import HydroParams, Acados_Settings

class MMGMPCController:
    def __init__(self):
        rospy.init_node('mmg_mpc_controller', anonymous=True)
        rospy.loginfo("Acados 求解器初始化...")
        hydro_params = HydroParams()
        hydro_params.mass = rospy.get_param("/hydrodynamics/mass", 38.0)
        hydro_params.I_z = rospy.get_param("/hydrodynamics/I_z", 6.25)
        hydro_params.B = rospy.get_param("/hydrodynamics/B", 0.74)
        hydro_params.Xu_dot = rospy.get_param("/hydrodynamics/Xu_dot", -1.9)
        hydro_params.Yv_dot = rospy.get_param("/hydrodynamics/Yv_dot", -29.3171)
        hydro_params.Nr_dot = rospy.get_param("/hydrodynamics/Nr_dot", -4.2155)
        hydro_params.Xu = rospy.get_param("/hydrodynamics/Xu", 26.43)
        hydro_params.Yv = rospy.get_param("/hydrodynamics/Yv", 72.64)
        hydro_params.Nr = rospy.get_param("/hydrodynamics/Nr", 22.96)
        input_limit = rospy.get_param("/input_limit", 20.0)
        self.horizon = rospy.get_param("horizon", 100)
        self.dt = rospy.get_param("dt", 0.1)
        Q_list = rospy.get_param("x_weight", [1.0, 1.0, 0.5, 0.0, 0.0, 0.0])
        R_list = rospy.get_param("u_weight", [0.1, 0.1])
        Q = np.diag(Q_list)
        R = np.diag(R_list)
        self.state = np.zeros(6)
        self.t = 0.0
        self.settings = Acados_Settings(hydro_params, input_limit, self.state, self.horizon, self.dt, Q, R)

if __name__ == "__main__":
    controller = MMGMPCController()
