FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND noninteractive

# Install pacakge
RUN apt-get update && \
    apt-get install -y vim && \
    apt-get install -y python3-pip && \
    apt-get install -y ros-${ROS_DISTRO}-gmapping && \
    apt-get install -y ros-${ROS_DISTRO}-map-server && \
    apt-get install -y ros-${ROS_DISTRO}-amcl && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the working directory in the container
WORKDIR /app

COPY requirements_deploy.txt /app/requirements_deploy.txt
RUN pip3 install --no-cache-dir -r requirements_deploy.txt

COPY light_mappo /app/light_mappo
COPY test /app/test
COPY ros_ws /app/ros_ws
COPY Makefile /app/Makefile

# Build maxbot_real package
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd ros_ws && \
    catkin_make -DCATKIN_BLACKLIST_PACKAGES='maxbot_sim'"

# Setup ros environment
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source /app/ros_ws/devel/setup.bash" >> ~/.bashrc
