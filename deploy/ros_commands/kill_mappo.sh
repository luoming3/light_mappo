#!/bin/bash

set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/app/ros_ws/devel/setup.bash" || echo ""

if rosnode list | grep -o ${1} ; then \
    rosnode kill ${1} ; \
fi
rostopic pub -1 /cmd_vel geometry_msgs/Twist '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}'
echo "kill mappo_node and stop maxbot"
