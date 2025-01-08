#!/bin/bash

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/app/ros_ws/devel/setup.bash" || echo ""

# kill ros node
rosnode kill mappo_node 2>&1
rosnode kill amcl_init_node 2>&1

echo "kill mappo_node and stop maxbot"
echo "y" | rosnode cleanup
