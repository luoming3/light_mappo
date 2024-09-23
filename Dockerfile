FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND noninteractive

# Install pacakge
RUN apt-get update && \
    apt-get install -y vim && \
    apt-get install -y python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the working directory in the container
WORKDIR /app

COPY requirements_deploy.txt /app/requirements_deploy.txt
RUN pip3 install --no-cache-dir -r requirements_deploy.txt

COPY light_mappo /app/light_mappo
COPY test /app/test