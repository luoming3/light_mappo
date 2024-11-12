#!/bin/bash

set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/app/ros_ws/devel/setup.bash" || echo ""

nohup roslaunch maxbot_real maxbot_real_navigation.launch >> \
    /app/logs/navigation.log 2>&1 &
