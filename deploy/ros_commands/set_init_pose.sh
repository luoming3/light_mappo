#!/bin/bash

set -e

SCRIPTS_PATH="/app/ros_ws/src/maxbot_real/scripts"

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/app/ros_ws/devel/setup.bash" || echo ""

# init pose
python3 -u ${SCRIPTS_PATH}/set_init_pose.py "${1}" "${2}" "${3}" "${4}"
