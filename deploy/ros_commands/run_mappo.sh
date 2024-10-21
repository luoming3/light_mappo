#!/bin/bash

set -e

# setup ros environment
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "/app/ros_ws/devel/setup.bash" || echo ""

# 定义处理函数
function handle_ctrlc() {
    echo "catch CTRL+C, stop maxbot and exit"
    rostopic pub -1 /cmd_vel geometry_msgs/Twist '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}'
    exit 0
}
# 捕获 SIGINT 信号并调用处理函数
trap handle_ctrlc SIGINT

python3 ./ros_ws/src/maxbot_real/scripts/mappo_node.py "(${1})" "(${2})"
