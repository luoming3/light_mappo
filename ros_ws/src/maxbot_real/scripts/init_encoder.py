import rospy
import re
import numpy as np

from geometry_msgs.msg import Twist
from std_msgs.msg import String

ACION_PUBLISHER = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
ENCODER = float("inf")


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
    pattern = "encoder:(-?[0-9]*\.?[0-9]*)"
    searcher = re.search(pattern, sensor_data)
    if searcher:
        ENCODER = float(searcher.group())
    else:
        raise RuntimeError(f"invalid sensor_data: {sensor_data}")


def main():
    rospy.init_node("init_ratory_encoder")
    rospy.Subscriber("/sensor_data", String, process_sensor_data)
    # pub FPS: 10 Hz
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if -5 < ENCODER < 5:
            rospy.loginfo(f"init rotary encoder successfully.")
            return
        publish_action(np.array([0., 0.5]))
        rate.sleep()
    rospy.spin()


if __name__ == "__main__":
    main()
