#!/bin/bash

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/app/ros_ws/devel/setup.bash" || echo ""

rosnode kill amcl 2>&1
rosnode kill move_base 2>&1
rosnode kill map_server 2>&1
