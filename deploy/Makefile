help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=mappo
MAPPO_NODE_NAME=mappo_node
AMCL_INIT_NODE_NAME=amcl_init_node

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
	-v $(shell pwd)/models:/app/models \
	-v $(shell pwd)/ros_ws/src/maxbot_real/map:/app/ros_ws/src/maxbot_real/map \
	-v $(shell pwd)/ros_commands:/app/ros_commands \
	-e ROS_IP="${ROS_IP}" \
	-e ROS_MASTER_URI="http://${ROS_IP}" \
	-e ROS_HOSTNAME="${ROS_IP}" \
	${IMAGE_NAME}:${IMAGE_TAG}

run-vnc-container:
	@docker run --init -it -d --name ${CONTAINER_NAME}-vnc --network=host \
	-v $(shell pwd)/models:/app/models \
	-v $(shell pwd)/ros_ws/src/maxbot_real/map:/app/ros_ws/src/maxbot_real/map \
	-v $(shell pwd)/ros_commands:/app/ros_commands \
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
	@docker exec -it ${CONTAINER_NAME} /bin/bash -c \
	'bash ros_commands/init.sh'

run-mappo: # e.g. make run-mappo start=1,1 goal=0,0
	@docker exec -it ${CONTAINER_NAME} /bin/bash -c \
	'bash ros_commands/run_mappo.sh ${start} ${goal}'

kill-mappo:
	@docker exec -it ${CONTAINER_NAME} /bin/bash -c \
	'bash ros_commands/kill_mappo.sh ${MAPPO_NODE_NAME}'

kill-init-amcl:
	@docker exec -it ${CONTAINER_NAME} /bin/bash -c \
	'bash ros_commands/kill_init_amcl.sh ${AMCL_INIT_NODE_NAME}'

clean-node: kill-init-amcl kill-mappo
	@docker exec -it ${CONTAINER_NAME} /bin/bash -c \
	'bash ros_commands/clean_node.sh'

rotate: # TODO
	@echo "rotate maxbot to the inital state"

# remote command
init-all: # TODO
	@echo "init all maxbot"

start-all: # TODO
	@echo "start all maxbot"

stop-all: # TODO
	@echo "stop all maxbot"

rotate-all: # TODO
	@echo "restore all maxbot oritations to their inital state"
