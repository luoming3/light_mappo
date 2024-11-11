help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=mappo

DEPLOY_TAG=release-20241108

MAXBOT_IP_1=
MAXBOT_IP_2=
MAXBOT_IP_3=
MAXBOT_IP_4=
MAXBOT_IP_5=
MAXBOT_IP_6=

REMOTE_USER=ubuntu
MAXBOT_IPS=${MAXBOT_IP_1} ${MAXBOT_IP_2} ${MAXBOT_IP_3} ${MAXBOT_IP_4} \
	${MAXBOT_IP_5} ${MAXBOT_IP_6}

# ROS_IP=$(shell hostname -I | awk '{print $$1}')
ROS_IP=127.0.0.1

update-tag:
	@last_release=$(shell grep -P -o "release-\d*" ../README.md) ; \
	sed -i "s/$${last_release}/$(DEPLOY_TAG)/g" ../README.md

setup-ssh:
	@for REMOTE_HOST in ${MAXBOT_IPS}; do \
		bash scripts/setup_ssh.sh ${REMOTE_USER} $${REMOTE_HOST}; \
	done

run-container:
	@docker run --init -it -d --name ${CONTAINER_NAME} --network=host \
	--restart=always \
	-v $(shell pwd)/models:/app/models \
	-v $(shell pwd)/ros_ws/src/maxbot_real/map:/app/ros_ws/src/maxbot_real/map \
	-v $(shell pwd)/ros_commands:/app/ros_commands \
	-v $(shell pwd)/logs:/app/logs \
	-e ROS_IP="${ROS_IP}" \
	-e ROS_MASTER_URI="http://${ROS_IP}:13131" \
	-e ROS_HOSTNAME="${ROS_IP}" \
	${IMAGE_NAME}:${IMAGE_TAG}

run-vnc-container:
	@docker run --init -it -d --name ${CONTAINER_NAME}-vnc --network=host \
	--restart=always \
	-v $(shell pwd)/models:/app/models \
	-v $(shell pwd)/ros_ws/src/maxbot_real/map:/app/ros_ws/src/maxbot_real/map \
	-v $(shell pwd)/ros_commands:/app/ros_commands \
	-v $(shell pwd)/logs:/app/logs \
	-e ROS_IP="${ROS_IP}" \
	-e ROS_MASTER_URI="http://${ROS_IP}:13131" \
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

# ros command
launch-gmapping:
	@docker exec -it ${CONTAINER_NAME} /bin/bash -c \
	'bash ros_commands/launch_gmapping.sh'

launch-save-map:
	@docker exec -it ${CONTAINER_NAME} /bin/bash -c \
	'bash ros_commands/launch_save_map.sh'

launch-amcl:
	@docker exec -it ${CONTAINER_NAME} /bin/bash -c \
	'bash ros_commands/launch_amcl.sh'

launch-navigation:
	@docker exec -it ${CONTAINER_NAME} /bin/bash -c \
	'bash ros_commands/launch_navigation.sh'

init:
	@docker exec ${CONTAINER_NAME} /bin/bash -c \
	'make -C ros_commands/ init'

run-mappo: # e.g. make run-mappo start=1,1 goal=0,0
	@docker exec ${CONTAINER_NAME} /bin/bash -c \
	'make -C ros_commands/ run-mappo -e start=${start} -e goal=${goal}'

clean-node:
	@docker exec ${CONTAINER_NAME} /bin/bash -c \
	'make -C ros_commands/ clean-node'

rotate: # TODO
	@echo "rotate maxbot to the inital state"

set-init-pose: # e.g. make set-init-pose x=0 y=0 ori_z=0 ori_w=1
	@docker exec ${CONTAINER_NAME} /bin/bash -c \
	'make -C ros_commands/ set-init-pose \
	-e x=${x} -e y=${y} -e ori_z=${ori_z} -e ori_w=${ori_w}'

# remote command
init-all:
	@for REMOTE_HOST in ${MAXBOT_IPS}; do \
		ssh ${REMOTE_USER}@$${REMOTE_HOST} 'cd ${DEPLOY_TAG}/deploy; \
			make init'; \
	done

start-all: # e.g. make start-all start=1,1 goal=0,0
	@for REMOTE_HOST in ${MAXBOT_IPS}; do \
		ssh ${REMOTE_USER}@$${REMOTE_HOST} 'cd ${DEPLOY_TAG}/deploy; \
			make run-mappo start=${start} goal=${goal}'; \
	done

stop-all:
	@for REMOTE_HOST in ${MAXBOT_IPS}; do \
		ssh ${REMOTE_USER}@$${REMOTE_HOST} 'cd ${DEPLOY_TAG}/deploy; \
			make clean-node'; \
	done

rotate-all: # TODO
	@echo "restore all maxbot oritations to their inital state"
