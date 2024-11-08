#!/bin/bash

set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/app/ros_ws/devel/setup.bash" || echo ""

# init amcl
cd /app && \
python3 -u ./ros_ws/src/maxbot_real/scripts/amcl_init_node.py
