#!/usr/bin/env bash
set -eu

[[ "$(lsb_release -sc)" == "focal" ]] || exit 1
ROS_DISTRO=noetic

## Setup your sources.list
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

## Set up your keys
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

## Installation
apt update
apt-get install -y ros-${ROS_DISTRO}-desktop-full \
    ros-${ROS_DISTRO}-gmapping \
    ros-${ROS_DISTRO}-map-server \
    ros-${ROS_DISTRO}-navigation

## Environment setup
echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

## Dependencies for building packages
apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

## Initialize rosdep
rosdep init 
rosdep update

grep -F "ROS_IP" ~/.bashrc || echo "export ROS_IP=127.0.0.1" >> ~/.bashrc

grep -F "ROS_MASTER_URI" ~/.bashrc || echo "export ROS_MASTER_URI=http://\$ROS_IP:11311" >> ~/.bashrc

echo ""
echo "Success installing ROS ${ROS_DISTRO}"
echo "Run 'source ~/.bashrc'"
echo ""
echo "If any error occurs, please refer to the following URL."
echo "https://github.com/Tiryoh/ros_setup_scripts_ubuntu/"
echo ""
