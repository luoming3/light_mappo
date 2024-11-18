#!/bin/bash

set -e

SCRIPTS_PATH="/app/ros_ws/src/maxbot_real/scripts"

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/app/ros_ws/devel/setup.bash" || echo ""

# init amcl
python3 -u ${SCRIPTS_PATH}/init_encoder.py
