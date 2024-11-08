import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler

pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)


def publish_initial_pose(x, y, theta):
    rospy.init_node('initial_pose_node', anonymous=True)

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "map"
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = 0.0
    # set orientation, use euler angle
    quaternion = quaternion_from_euler(0, 0, theta)
    msg.pose.pose.orientation.x = quaternion[0]
    msg.pose.pose.orientation.y = quaternion[1]
    msg.pose.pose.orientation.z = quaternion[2]
    msg.pose.pose.orientation.w = quaternion[3]
    # set covariance
    msg.pose.covariance[0] = 0.25
    msg.pose.covariance[7] = 0.25
    msg.pose.covariance[-1] = 0.25

    # only pub msg once
    while pub.get_num_connections() < 1:
        # wait for a subscriber
        pass
    pub.publish(msg)


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    try:
        x = float(args[0])
        y = float(args[1])
        theta = float(args[2])
        publish_initial_pose(x, y, theta)
        rospy.loginfo("initial pose successfully")
    except rospy.ROSInterruptException:
        raise RuntimeError("initial pose error")
