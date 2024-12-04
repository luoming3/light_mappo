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

import os
import sys

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from light_mappo.agent import Agent
from make_plan import get_path

# /cmd_vel topic
ACION_PUBLISHER = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
STATUS_RUNNING = 0
STATUS_SUCCESS = 1
STATUS_FAILURE = 2
STATUS_UNKNOWN = 3

records = []


class MappoNode:

    def __init__(self, start, goal, id, host, port) -> None:
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
        self.status = STATUS_RUNNING
        self.path = get_path(start, goal)
        if len(self.path) == 0:
            raise RuntimeError("can't find a path")
        self.guide_point = self.path[0]

        self.amcl_subscriber = rospy.Subscriber("/amcl_pose",
                                                PoseWithCovarianceStamped,
                                                self.process_amcl_pose)
        self.odom_subscriber = rospy.Subscriber("/odom", Odometry,
                                                self.process_odom)
        self.sensor_data_subscriber = rospy.Subscriber("/sensor_data", String,
                                                       self.process_sensor_data)
        # startup socket server
        if id == 1:
            sc_th = threading.Thread(target=self.car_center_socket_server, daemon=True)
            sc_th.start()
        # get car center from remote master
        sc_th = threading.Thread(target=self.car_center_socket_client, daemon=True)
        sc_th.start()
        rospy.loginfo("starting mappo node")
        rospy.loginfo(
            f"start position: {self.start}, goal position: {self.goal}")
        rospy.loginfo(f"path: {self.path}")

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
            self.rotation = float(searcher.group(3))
        else:
            raise RuntimeError(f"invalid sensor_data: {sensor_data}")

    def get_obs(self):
        rpos_car_dest_norm = normalized(self.guide_point - self.car_center)
        maxbot_linear_velocities = self.velocities
        maxbot_orientation = np.array([self.euler_ori[2]])
        force = self.force
        obs = np.concatenate((rpos_car_dest_norm, maxbot_linear_velocities,
                              maxbot_orientation, force),
                             axis=0)
        obs = np.expand_dims(obs, axis=0)
        return obs

    def clip_path(self):
        path = copy.deepcopy(self.path)
        if len(path) > 1:
            first_point = path[0]
            second_point = path[1]
            angle = np.dot(second_point - first_point,
                           self.car_center - first_point)

            if angle < 0:
                pass
            else:
                rospy.loginfo(f"the next guide point: {self.path[1]}")
                self.path = self.path[1:]
        self.guide_point = self.path[0]

    def step(self):
        if (self.car_center.size == 0) or (self.velocities.size== 0) or \
            (self.orientation.size == 0):
            rospy.logwarn("observation is None")
            return STATUS_RUNNING

        self.clip_path()
        obs = self.get_obs()
        action = get_action(obs)
        publish_action(action)
        record = []
        # car_center, guide_point, velocity, orientation, force
        record.append(self.car_center.copy())
        record.append(self.guide_point.copy())
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
                            send_str = f'{self.id},{self.position[0]},{self.position[1]},{self.status}'
                            s.sendall(bytes(send_str, "utf8"))
                            data_str = s.recv(1024)

                            data_str = data_str.decode("utf8")
                            if len(data_str) > 1:
                                data_split = data_str.split(",")
                                self.car_center = np.array([data_split[1],data_split[2]], dtype=np.float32)
                                status = int(data_split[3])
                                if status != STATUS_RUNNING:
                                    self.status = status

                        time.sleep(1 / 50.)
            except ConnectionRefusedError:
                rospy.logwarn("Connection refused, and then wait 1s")
                time.sleep(1)

                retry_count += 1
                if retry_count == 5:
                    raise RuntimeError("Connection refused")

    def car_center_socket_server(self):

        def get_car_center_str(data, status):
            if status != STATUS_RUNNING:
                self.status = status
            if 1 in data and 6 in data:
                car_center = (data[1] + data[6]) / 2
                data_str = f"1,{car_center[0]},{car_center[1]},{status}"
            elif 2 in data and 5 in data:
                car_center = (data[2] + data[5]) / 2
                data_str = f"1,{car_center[0]},{car_center[1]},{status}"
            else:
                data_str = "0"

            return data_str

        def process_req(conn, addr, data):
            with conn:
                print('Connected by', addr)
                while True:
                    data_recv = conn.recv(1024)
                    if not data_recv:
                        break
                    data_split = data_recv.decode("utf8").split(",")
                    id = int(data_split[0])
                    position = np.array([data_split[1], data_split[2]],
                                        dtype=np.float32)
                    data[id] = position
                    status = int(data_split[3])

                    center_str = get_car_center_str(data, status)
                    conn.sendall(bytes(center_str, "utf8"))
        data = {}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            while True:
                conn, addr = s.accept()
                sub_threading = threading.Thread(target=process_req,
                                                args=(conn, addr, data),
                                                daemon=True)
                sub_threading.start()


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
    start, goal, id, host, port = args
    mappo_node = MappoNode(start, goal, id, host, port)

    # pub FPS: 10 Hz
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        status = mappo_node.step()
        if status == STATUS_RUNNING:
            rate.sleep()
            continue
        elif status == STATUS_SUCCESS:
            rospy.loginfo("task success")
            publish_action(np.array([0, 0]))
            return
        elif status == STATUS_FAILURE:
            rospy.logerr("task failure")
            publish_action(np.array([0, 0]))
            return
        else:
            rospy.logerr("unknown error")
            publish_action(np.array([0, 0]))
            raise RuntimeError("unknown error")
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
    except:
        raise RuntimeError("input args is invalid")
    else:
        main(start, goal, id, host, port)
    finally:
        # stop maxbot
        os.system("rostopic pub -1 /cmd_vel geometry_msgs/Twist \
                  '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}'")
        np.save(f"/app/logs/step_record_{id}.npy", records)
