help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=mappo

MAXBOT_IP_1=
MAXBOT_IP_2=
MAXBOT_IP_3=
MAXBOT_IP_4=
MAXBOT_IP_5=
MAXBOT_IP_6=

REMOTE_USER=ubuntu
MAXBOT_IPS=${MAXBOT_IP_1} ${MAXBOT_IP_2} ${MAXBOT_IP_3} ${MAXBOT_IP_4} \
	${MAXBOT_IP_5} ${MAXBOT_IP_6}

ROS_IP=$(shell hostname -I | awk '{print $$1}')

setup-ssh:
	@for REMOTE_HOST in ${MAXBOT_IPS}; do \
		bash scripts/setup_ssh.sh ${REMOTE_USER} $${REMOTE_HOST}; \
	done

run-container:
	@docker run --init -it -d --name ${CONTAINER_NAME} --network=host \
	-v $(shell pwd)/models:/app/deploy/models \
	-v $(shell pwd)/ros_ws/src/maxbot_real/map:/app/ros_ws/src/maxbot_real/map \
	-e ROS_IP="${ROS_IP}" \
	-e ROS_MASTER_URI="http://${ROS_IP}" \
	-e ROS_HOSTNAME="${ROS_IP}" \
	${IMAGE_NAME}:${IMAGE_TAG}

run-vnc-container:
	@docker run --init -it -d --name ${CONTAINER_NAME}-vnc --network=host \
	-v $(shell pwd)/models:/app/deploy/models \
	-v $(shell pwd)/ros_ws/src/maxbot_real/map:/app/ros_ws/src/maxbot_real/map \
	-e ROS_IP="${ROS_IP}" \
	-e ROS_MASTER_URI="http://${ROS_IP}" \
	-e ROS_HOSTNAME="${ROS_IP}" \
	${IMAGE_NAME}:${IMAGE_TAG}-vnc

remove-container:
	@docker rm -f ${CONTAINER_NAME}

exec-container:
	@docker exec -it ${CONTAINER_NAME} bash

restart-container:
	@docker restart ${CONTAINER_NAME}

load-image:
	@for i in $(shell ls image/*.gz); do \
		docker load < $$i ; \
	done
	@echo "load image successfully"

init-all: # TODO
	@echo "init all maxbot"

start-all: # TODO
	@echo "start all maxbot"

stop-all: # TODO
	@echo "stop all maxbot"

rotate-all: # TODO
	@echo "restore all maxbot oritations to their inital state"
