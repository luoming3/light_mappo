#!/bin/bash

set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/app/ros_ws/devel/setup.bash" || echo ""

stdbuf -oL roslaunch maxbot_sim maxbot_sim_navigation.launch
