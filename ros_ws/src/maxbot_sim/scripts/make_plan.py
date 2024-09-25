import rospy
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped
import numpy as np

from light_mappo.utils.util import timethis


def get_path(start, goal, tolerance=0.5, distance=1.0):
    start_pose = create_pose(start)
    goal_pose = create_pose(goal)

    plan = make_plan(start_pose, goal_pose, tolerance)
    path = simplify_path(plan, distance)

    return path


def make_plan(start, goal, tolerance):
    rospy.wait_for_service('/move_base/make_plan')
    try:
        get_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
        req = GetPlan()
        req.start = start
        req.goal = goal
        req.tolerance = tolerance
        resp = get_plan(req.start, req.goal, req.tolerance)
        return resp.plan
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return None


def create_pose(pose, frame_id="map"):
    x, y = pose[0], pose[1]
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.w = 1.0
    return pose


def simplify_path(path, distance=1.0):
    if not path.poses:
        rospy.logwarn("Received an empty path")
        return []

    path = np.array([[pose.pose.position.x, pose.pose.position.y]
                     for pose in path.poses])
    simplified_path = [path[-1]]
    for i in range(len(path) - 2, -1, -1):
        last_point = simplified_path[-1]
        current_point = path[i]
        dist = cal_dist(current_point, last_point)
        if dist >= distance:
            simplified_path.append(current_point)

    simplified_path.reverse()

    if len(simplified_path) == 1:
        return simplified_path

    # remove the first point if it is too close to the start point
    if cal_dist(simplified_path[0], path[0]) < distance:
        simplified_path = simplified_path[1:]

    return simplified_path


def cal_dist(point1, point2):
    return np.linalg.norm(point1 - point2)


@timethis
def main():
    start = (-5.0, 3.0)
    goal = (5.0, -3.0)

    guide_point = get_path(start, goal)
    if guide_point:
        [print(point) for point in guide_point]
    else:
        rospy.logwarn("No plan received")


if __name__ == "__main__":
    rospy.init_node('make_plan_client')

    main()
