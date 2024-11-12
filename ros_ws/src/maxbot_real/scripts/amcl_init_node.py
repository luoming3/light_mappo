import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np

import os
import sys

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

CONVERGENCE = False
epsilon = 1e-5
# /cmd_vel topic
ACION_PUBLISHER = rospy.Publisher('/cmd_vel', Twist, queue_size=1)


def publish_action(action):
    if action.size == 0:
        return
    twist = Twist()
    twist.linear.x = action[0]
    twist.angular.z = action[1]
    ACION_PUBLISHER.publish(twist)


def process_amcl_covariance(message):
    global CONVERGENCE
    covariance = message.pose.covariance
    x_var = covariance[0]
    y_var = covariance[7]
    yaw_var = covariance[-1]

    if x_var < epsilon and y_var < epsilon and yaw_var < epsilon:
        CONVERGENCE = True


def topic_exists(topic_name):
    topic_list = rospy.get_published_topics()
    for topic, _ in topic_list:
        if topic == topic_name:
            return True
    return False


def init_amcl():
    rospy.init_node("amcl_init_node")
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped,
                     process_amcl_covariance)
    amcl_topic = "/amcl_pose"
    # pub FPS: 10 Hz
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if not topic_exists(amcl_topic):
            raise RuntimeError("The topic /amcl_pose doesn't exist.")
        if not CONVERGENCE:
            publish_action(np.array([0, 1]))
            rate.sleep()
        else:
            publish_action(np.array([0, 0]))
            return CONVERGENCE
    rospy.spin()


if __name__ == "__main__":
    try:
        if_convergence = init_amcl()
        print("initialize amcl successfully")
    finally:
        # stop maxbot
        os.system("rostopic pub -1 /cmd_vel geometry_msgs/Twist \
            '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}'")
