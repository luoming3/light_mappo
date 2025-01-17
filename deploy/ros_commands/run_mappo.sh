#!/bin/bash

set -e

SCRIPTS_PATH="/app/ros_ws/src/maxbot_real/scripts"

# setup ros environment
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "/app/ros_ws/devel/setup.bash" || echo ""

# clear obstacles in the costmaps before make plan
rosservice call /move_base/clear_costmaps "{}"

python3 -u ${SCRIPTS_PATH}/mappo_node.py "(${1})" "(${2})"
