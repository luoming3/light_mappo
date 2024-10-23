#!/bin/bash

set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/app/ros_ws/devel/setup.bash" || echo ""

if rosnode list | grep -o "mappo_node"
then
    rosnode kill mappo_node
    rostopic pub -1 /cmd_vel geometry_msgs/Twist '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}'
fi
if rosnode list | grep -o "amcl_init_node"
then
    rosnode kill amcl_init_node
    rostopic pub -1 /cmd_vel geometry_msgs/Twist '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}'
fi
echo "kill mappo_node and stop maxbot"

echo "y" | rosnode cleanup
