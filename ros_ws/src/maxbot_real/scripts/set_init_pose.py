import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler

pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)


def publish_initial_pose(x, y, ori_z, ori_w):
    rospy.init_node('initial_pose_node', anonymous=True)

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "map"
    # set position
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = 0.0
    # set orientation
    msg.pose.pose.orientation.x = 0.0
    msg.pose.pose.orientation.y = 0.0
    msg.pose.pose.orientation.z = ori_z
    msg.pose.pose.orientation.w = ori_w
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
        ori_z = float(args[2])
        ori_w = float(args[3])

        publish_initial_pose(x, y, ori_z, ori_w)
        rospy.loginfo("initial pose successfully")
        rospy.loginfo(f"initial pose: {x}, {y}, {ori_z}, {ori_w}\n")
    except rospy.ROSInterruptException:
        raise RuntimeError("initial pose error")
