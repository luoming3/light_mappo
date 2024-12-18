import rospy
import re
import numpy as np
import math
import time

from geometry_msgs.msg import Twist
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped

ACION_PUBLISHER = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
ENCODER = float("inf")
euler_angle = float("inf")


def publish_action(action):
    if action.size == 0:
        return
    twist = Twist()
    twist.linear.x = action[0]
    twist.angular.z = action[1]
    ACION_PUBLISHER.publish(twist)


def process_sensor_data(sensor_data):
    global ENCODER
    sensor_data = str(sensor_data)
    pattern = "encoder:(.+?)\""
    searcher = re.search(pattern, sensor_data)
    if searcher:
        ENCODER = float(searcher.group(1))
    else:
        raise RuntimeError(f"invalid sensor_data: {sensor_data}")


def process_amcl_pose(message):
    global euler_angle
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    position = np.array([position.x, position.y])
    orientation = np.array(
        [orientation.x, orientation.y, orientation.z, orientation.w])
    euler_angle = euler_from_quaternion(orientation)[2]


def main():
    rospy.init_node("init_ratory_encoder")
    rospy.Subscriber("/sensor_data", String, process_sensor_data)
    time.sleep(1) # wait to get sensor_data
    rospy.Subscriber("/amcl_pose",PoseWithCovarianceStamped, process_amcl_pose)
    # pub FPS: 10 Hz
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
<<<<<<< HEAD
        if -math.radians(5) < euler_angle < math.radians(5):
            publish_action(np.array([0., 0.]))
            time.sleep(0.5) # sleep to get the latest value
            with open("/app/.init_angle", "w") as f:
                f.write(f"{ENCODER}")
            rospy.loginfo(f"get init rotary encoder successfully.")
            return
        if euler_angle > 0:
            publish_action(np.array([0., -0.5]))
        else:
            publish_action(np.array([0., 0.5]))
=======
        if -5 < ENCODER < 5:
            rospy.loginfo(f"init rotary encoder successfully.")
            return
        publish_action(np.array([0., 0.5]))
>>>>>>> isaac-sim-maxbot-car-center-socket-hardcode
        rate.sleep()
    rospy.spin()


if __name__ == "__main__":
    main()
