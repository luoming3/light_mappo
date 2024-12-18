import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import numpy as np
import math
import copy
import re
import socket
import threading
import time
from tf.transformations import euler_from_quaternion

import os
import sys

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from light_mappo.agent import Agent
from make_plan import get_path
from status import *
from socket_server import car_center_socket_server

# /cmd_vel topic
ACION_PUBLISHER = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

records = []
angle_tolerance = 5 / 180 * math.pi
turn_threshold = 30 / 180 * math.pi
force_threshold = 100000
running_v = 0.25
running_omega = 0.25
turn_omega = 0.5
# w is half the width of the assembled car
# l is half the length of the assembled car
w = 0.4
l = 0.6

class MappoNode:

    def __init__(self, start, goal, id, host, port, method) -> None:
        self.id = id
        self.host = host
        self.port = port
        self.position = np.array([])
        self.car_center = np.array([])
        self.orientation = np.array([])
        self.euler_ori = np.array([])
        self.velocities = np.array([])
        self.force = np.array([0., 0.])
        self.rotation = 0.0
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.status = STATUS_STOP
        self.master_status = STATUS_STOP
        self.path = get_path(start, goal)
        if len(self.path) == 0:
            raise RuntimeError("can't find a path")
        self.guide_point = self.path[0]
        with open("/app/.init_angle", "r") as f:
            self.init_angle = float(f.read())
        self.master_guide_point = np.array([])
        self.method = method
        self.w = w
        self.l = l
        self.gamma = math.atan(w / l)

        self.amcl_subscriber = rospy.Subscriber("/amcl_pose",
                                                PoseWithCovarianceStamped,
                                                self.process_amcl_pose)
        self.odom_subscriber = rospy.Subscriber("/odom", Odometry,
                                                self.process_odom)
        self.sensor_data_subscriber = rospy.Subscriber("/sensor_data", String,
                                                       self.process_sensor_data)
        # startup socket server
        if id == 1:
            sc_th = threading.Thread(target=car_center_socket_server, daemon=True,
                                     args=(host, port))
            sc_th.start()
        # get car center from remote master
        sc_th = threading.Thread(target=self.car_center_socket_client, daemon=True)
        sc_th.start()
        rospy.loginfo("starting mappo node")
        rospy.loginfo(
            f"start position: {self.start}, goal position: {self.goal}")
        rospy.loginfo(f"path: {self.path}")
        # sleep for 1s to ensure that the socket server obtains the status of all MaxBots
        time.sleep(1)

    def process_amcl_pose(self, message):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        self.position = np.array([position.x, position.y])
        self.orientation = np.array(
            [orientation.x, orientation.y, orientation.z, orientation.w])

    def process_odom(self, odom_data):
        if odom_data and self.orientation.size > 0:
            linear_x = odom_data.twist.twist.linear.x
            self.euler_ori = quaternion_to_euler(self.orientation)
            self.velocities = get_vel_from_linear(linear_x, self.euler_ori)

    def process_sensor_data(self, sensor_data):
        sensor_data = str(sensor_data)
        pattern = '"LoadA:(.+?),LoadB:(.+?),encoder:(.+?)"'
        searcher = re.search(pattern, sensor_data)
        if searcher:
            LoadA = float(searcher.group(1))
            LoadB = float(searcher.group(2))
            self.force = np.array([LoadB, LoadA])
            encoder = float(searcher.group(3))
            rotation = math.radians(self.init_angle - encoder)
            if rotation >= 0:
                self.rotation = rotation
            else:
                self.rotation = 2 * math.pi + rotation
        else:
            raise RuntimeError(f"invalid sensor_data: {sensor_data}")

    def get_obs(self):
        # get obs from different methods
        if self.method == "physics": # physics + mappo
            car_center = self.get_car_position_physics()
            guide_point = self.guide_point
        elif self.method == "socket": # socket + mappo
            car_center = self.car_center
            guide_point = self.master_guide_point
        elif self.method == "hard": # socket + hard
            car_center = self.car_center
            guide_point = self.master_guide_point
        else:
            raise RuntimeError("unknown execute method")

        # clip path for update own guide point
        self.clip_path(car_center)

        rpos_car_dest_norm = normalized(guide_point - car_center)
        maxbot_linear_velocities = self.velocities
        maxbot_orientation = np.array([self.euler_ori[2]])
        force = self.force
        obs = np.concatenate((rpos_car_dest_norm, maxbot_linear_velocities,
                              maxbot_orientation, force),
                             axis=0)
        obs = np.expand_dims(obs, axis=0)
        return obs, car_center, guide_point

    def clip_path(self, car_center):
        path = copy.deepcopy(self.path)
        if len(path) > 1:
            first_point = path[0]
            second_point = path[1]
            angle = np.dot(second_point - first_point,
                           car_center - first_point)

            if angle < 0:
                pass
            else:
                rospy.loginfo(f"the next guide point: {self.path[1]}")
                self.path = self.path[1:]
        self.guide_point = self.path[0]

    def step(self):
        if (self.car_center.size == 0) or (self.velocities.size== 0) or \
            (self.orientation.size == 0) or (self.master_guide_point.size == 0):
            self.status = STATUS_STOP
            rospy.logwarn("observation is None")
            return STATUS_STOP
        if self.master_status == STATUS_STOP:
            self.status = STATUS_STOP
            publish_action(np.array([0, 0]))
            return STATUS_STOP
        if self.master_status == STATUS_SUCCESS:
            self.status = STATUS_SUCCESS
            publish_action(np.array([0, 0]))
            return STATUS_SUCCESS
        if self.master_status == STATUS_FAILURE:
            self.status = STATUS_FAILURE
            publish_action(np.array([0, 0]))
            return STATUS_FAILURE
        obs, car_center, guide_point = self.get_obs()
        # get action from different method
        if self.method == "physics": # physics + mappo
            action = get_action(obs)
        elif self.method == "socket": # socket + mappo
            action = get_action(obs)
        elif self.method == "hard": # socket + hard
            action = self.get_action_hardcode()
        else:
            raise RuntimeError("unknown execute method")
        publish_action(action)
        record = []
        # car_center, guide_point, velocity, orientation, force
        record.append(car_center.copy())
        record.append(guide_point.copy())
        record.append(self.velocities.copy())
        record.append(self.euler_ori[2])
        record.append(self.force.copy())
        records.append(record)

        # arrival
        current_dist_to_goal = np.linalg.norm(self.car_center - self.goal)
        current_dist_to_point = np.linalg.norm(self.car_center -
                                               self.guide_point)
        if current_dist_to_goal < 0.2:
            self.status = STATUS_SUCCESS
        if current_dist_to_point > 5:
            self.status = STATUS_FAILURE
        if current_dist_to_point < 0.1:
            self.path = self.path[1:]

        return self.status

    def car_center_socket_client(self):
        retry_count = 0
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self.host, self.port))

                    while True:
                        if self.position.size > 0:
                            send_str = f'{self.id},{self.position[0]},{self.position[1]},{self.status},\
                                {self.guide_point[0]},{self.guide_point[1]}'
                            s.sendall(bytes(send_str, "utf8"))
                            data_str = s.recv(1024)

                            data_str = data_str.decode("utf8")
                            if len(data_str) > 1:
                                data_split = data_str.split(",")
                                self.car_center = np.array([data_split[1],data_split[2]], dtype=np.float32)
                                #self.car_center = self.position
                                master_status = int(data_split[3])
                                self.master_status = master_status
                                self.master_guide_point = np.array([data_split[4], data_split[5]], dtype=np.float32)

                        time.sleep(1 / 50.)
            except ConnectionRefusedError:
                rospy.logwarn("Connection refused, and then wait 1s")
                time.sleep(1)

                retry_count += 1
                if retry_count == 5:
                    raise RuntimeError("Connection refused")

    def get_action_hardcode(self):
        ori = euler_from_quaternion(self.orientation)[2]
        alpha = self.calculate_angle(self.car_center, self.master_guide_point)

        if abs(ori - alpha) < math.pi:
            abs_diff = abs(ori - alpha)
        else:
            abs_diff = 2 * math.pi - abs(ori - alpha)
        same_direction = True if abs_diff < math.pi / 2 else False

        if abs_diff > turn_threshold:
            self.status = STATUS_TURN
            turn_right_condition = False
            diff_angle = ori - alpha
            if same_direction:
                turn_right_condition = (0 < diff_angle and diff_angle < math.pi / 2) or \
                    (-2 * math.pi < diff_angle and diff_angle < -3 * math.pi / 2)
            else:
                turn_right_condition = True
            if turn_right_condition:
                return np.array([0, -turn_omega])
            else:
                return np.array([0, turn_omega])

        force = max(self.force)
        if abs_diff < angle_tolerance and force < force_threshold:
            if self.master_status == STATUS_TURN:
                self.status = STATUS_STOP
                return np.array([0., 0.])
            else:
                self.status = STATUS_RUNNING
                return np.array([running_v, 0.])
        else:
            self.status = STATUS_RUNNING
            turn_right_condition = False
            diff_angle = ori - alpha
            if same_direction:
                turn_right_condition = (0 < diff_angle and diff_angle < math.pi / 2) or \
                    (-2 * math.pi < diff_angle and diff_angle < -3 * math.pi / 2)
            else:
                turn_right_condition = True
            if turn_right_condition:
                return np.array([running_v, -running_omega])
            else:
                return np.array([running_v, running_omega])

    def calculate_angle(self, car_center, guide_point):
        x1, y1 = car_center[0], car_center[1]
        x2, y2 = guide_point[0], guide_point[1]

        dx = x2 - x1
        dy = y2 - y1
        magnitude =  math.sqrt(dx**2 + dy**2)

        cos_theta = dx / magnitude
        theta = math.acos(cos_theta)
        return theta if y2 > y1 else -theta

    def get_car_position_physics(self):
        alpha = euler_from_quaternion(self.orientation)[2]
        beta = self.rotation # should be between 0 ~ 2pi or -pi ~ pi
        x = self.position[0]
        y = self.position[1]
        if self.id == 1:
            gamma_ = self.gamma
            phi = beta - alpha - gamma_
            l_ = math.sqrt(self.w**2 + self.l**2)
            x0 = x - math.cos(phi) * l_
            y0 = y + math.sin(phi) * l_
        elif self.id == 2:
            gamma_ = math.pi / 2 - self.gamma
            phi = beta - alpha - gamma_
            l_ = math.sqrt(self.w**2 + self.l**2)
            x0 = x + math.sin(phi) * l_
            y0 = y + math.cos(phi) * l_
        elif self.id == 3:
            gamma_ = 0
            phi = beta - alpha - gamma_
            l_ = self.w
            x0 = x - math.sin(phi) * l_
            y0 = y - math.cos(phi) * l_
        elif self.id == 4:
            gamma_ = 0
            phi = beta - alpha - gamma_
            l_ = self.w
            x0 = x + math.sin(phi) * l_
            y0 = y + math.cos(phi) * l_
        elif self.id == 5:
            gamma_ = math.pi / 2 - self.gamma
            phi = beta - alpha - gamma_
            l_ = math.sqrt(self.w**2 + self.l**2)
            x0 = x - math.sin(phi) * l_
            y0 = y - math.cos(phi) * l_
        elif self.id == 6:
            gamma_ = self.gamma
            phi = beta - alpha - gamma_
            l_ = math.sqrt(self.w**2 + self.l**2)
            x0 = x + math.cos(phi) * l_
            y0 = y - math.sin(phi) * l_
        else:
            raise RuntimeError("unknown id")

        car_center = np.array([x0, y0])
        return car_center

def get_action(obs):
    '''
    Get action from algorithm
    '''
    if not np.any(obs):
        return None
    agent = Agent()
    return agent.act(obs)


def publish_action(action):
    if action.size == 0:
        return
    twist = Twist()
    twist.linear.x = action[0]
    twist.angular.z = action[1]
    ACION_PUBLISHER.publish(twist)


def quaternion_to_euler(qua_ori):
    x, y, z, w = qua_ori[0], qua_ori[1], qua_ori[2], qua_ori[3]
    # calculate euler angles
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def get_vel_from_linear(linear_vel, euler_ori):
    z = euler_ori[2]
    vel_x = math.cos(z) * linear_vel
    vel_y = math.sin(z) * linear_vel

    return np.array([vel_x, vel_y])


def normalized(v, axis=0):
    result = v / np.linalg.norm(v, axis=axis)
    return result


def main(*args):
    rospy.init_node("mappo_node")
    mappo_node = MappoNode(*args)

    # pub FPS: 10 Hz
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        status = mappo_node.step()
        if status == STATUS_RUNNING:
            rospy.loginfo("running")
        elif status == STATUS_SUCCESS:
            rospy.loginfo("task success")
            return
        elif status == STATUS_FAILURE:
            rospy.logerr("task failure")
            return
        elif status == STATUS_TURN:
            rospy.loginfo("turn direction")
        elif status == STATUS_STOP:
            rospy.loginfo("waiting")
        else:
            rospy.logerr("unknown error")
            publish_action(np.array([0, 0]))
            raise RuntimeError("unknown error")
        rate.sleep()

    rospy.spin()


if __name__ == "__main__":
    import ast
    args = sys.argv[1:]
    try:
        start = ast.literal_eval(args[0])
        goal = ast.literal_eval(args[1])
        id = int(args[2])
        host = args[3]
        port = int(args[4])
        method = args[5]
    except:
        raise RuntimeError("input args is invalid")
    else:
        main(start, goal, id, host, port, method)
    finally:
        # stop maxbot
        os.system("rostopic pub -1 /cmd_vel geometry_msgs/Twist \
                  '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}'")
        np.save(f"/app/logs/step_record_{id}.npy", records)
