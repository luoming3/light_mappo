import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import numpy as np
import math

from light_mappo.agent import Agent
from make_plan import get_path

# /cmd_vel topic
ACION_PUBLISHER = rospy.Publisher('/cmd_vel', Twist, queue_size=10)


class MappoNode:

    def __init__(self) -> None:
        self.position = None
        self.orientation = None
        self.euler_ori = None
        self.velocities = None
        self.force = np.array([0., 0.])
        self.path = []

        self.amcl_subscriber = rospy.Subscriber("/amcl_pose",
                                                PoseWithCovarianceStamped,
                                                self.process_amcl_pose)
        self.odom_subscriber = rospy.Subscriber("/odom", Odometry,
                                                self.process_odom)
        self.force_subscriber = rospy.Subscriber("/force", Float32,
                                                 self.process_force)  ## TODO
        self.rotation_subscriber = rospy.Subscriber(
            "/rotation", Float32, self.process_rotation)  # TODO

    def process_amcl_pose(self, message):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        self.position = np.array([position.x, position.y])
        self.orientation = np.array(
            [orientation.x, orientation.y, orientation.z, orientation.w])

    def process_odom(self, odom_data):
        if odom_data and np.any(self.orientation):
            linear_x = odom_data.twist.twist.linear.x
            self.euler_ori = quaternion_to_euler(self.orientation)
            self.velocities = get_vel_from_linear(linear_x, self.euler_ori)

    def process_force(self):
        # TODO
        self.force = np.array([0., 0.])

    def process_rotation(self):
        # TODO
        self.rotation = None

    def get_obs(self):
        ## TODO: calculate the rpos of car and dest
        if (not np.any(self.position)) or (not np.any(
                self.velocities)) or (not np.any(self.orientation)):
            return None
        rpos_car_dest_norm = normalized(np.array([0, 0]) - self.position)
        maxbot_linear_velocities = self.velocities
        maxbot_orientation = np.array([self.euler_ori[2]])
        force = self.force
        obs = np.concatenate((rpos_car_dest_norm, maxbot_linear_velocities,
                              maxbot_orientation, force),
                             axis=0)
        obs = np.expand_dims(obs, axis=0)
        return obs

    def get_car_position(self):
        # TODO
        pass

    def get_path(self, start, goal):
        self.path = get_path(start, goal)

    def get_guide_point(self):
        # TODO
        pass


def get_action(obs):
    '''
    Get action from algorithm
    '''
    if not np.any(obs):
        return None
    agent = Agent()
    return agent.act(obs)


def public_action(action):
    if not np.any(action):
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


def main():
    rospy.init_node("mappo_node")
    mappo_node = MappoNode()

    # pub FPS: 10 Hz
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        obs = mappo_node.get_obs()
        action = get_action(obs)
        public_action(action)
        rate.sleep()

    rospy.spin()


if __name__ == "__main__":
    main()
