help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=mappo

DEPLOY_TAG=release-20241118

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
	@docker exec ${CONTAINER_NAME} /bin/bash -c \
		'nohup bash ros_commands/launch_navigation.sh >> \
			/app/logs/navigation.log 2>&1 &'

kill-navigation:
	@docker exec ${CONTAINER_NAME} /bin/bash -c \
		'nohup bash ros_commands/kill_navigation.sh >> \
			/app/logs/navigation.log 2>&1 &'

init:
	@docker exec ${CONTAINER_NAME} /bin/bash -c \
		'nohup bash ros_commands/init.sh >> \
			/app/logs/init_amcl.log 2>&1 &'

run-mappo: # e.g. make run-mappo start=1,1 goal=0,0
	@docker exec ${CONTAINER_NAME} /bin/bash -c \
		'nohup bash ros_commands/run_mappo.sh ${start} ${goal} >> \
			/app/logs/mappo_node.log 2>&1 &'

clean-node:
	@docker exec ${CONTAINER_NAME} /bin/bash -c \
		'nohup bash ros_commands/clean_node.sh >> \
			/app/logs/clean_node.log 2>&1 &'

rotate:
	@docker exec ${CONTAINER_NAME} /bin/bash -c \
		'nohup bash ros_commands/init_encoder.sh >> \
			/app/logs/init_encoder.log 2>&1 &'

set-init-pose: # e.g. make set-init-pose x=0 y=0 ori_z=0 ori_w=1
	@docker exec ${CONTAINER_NAME} /bin/bash -c \
		'nohup bash ros_commands/set_init_pose.sh ${x} ${y} ${ori_z} ${ori_w} >> \
			/app/logs/set_init_pose.log 2>&1 &'

# remote command
init-all: set-init-pose-all
	@for REMOTE_HOST in ${MAXBOT_IPS}; do \
		echo "init on $${REMOTE_HOST}"; \
		ssh ${REMOTE_USER}@$${REMOTE_HOST} 'cd ${DEPLOY_TAG}/deploy; \
			make init'; \
	done

start-all: # e.g. make start-all start=1,1 goal=0,0
	@for REMOTE_HOST in ${MAXBOT_IPS}; do \
		echo "start mappo on $${REMOTE_HOST}"; \
		ssh ${REMOTE_USER}@$${REMOTE_HOST} 'cd ${DEPLOY_TAG}/deploy; \
			make run-mappo start=${start} goal=${goal}'; \
	done

stop-all:
	@for REMOTE_HOST in ${MAXBOT_IPS}; do \
		echo "stop mappo on $${REMOTE_HOST}"; \
		ssh ${REMOTE_USER}@$${REMOTE_HOST} 'cd ${DEPLOY_TAG}/deploy; \
			make clean-node'; \
	done

rotate-all: # TODO
	@echo "restore all maxbot oritations to their inital state"

set-init-pose-all:
	@if [ -n "${MAXBOT_IP_1}" ]; then \
		echo "set init pose on ${MAXBOT_IP_1}"; \
		ssh ${REMOTE_USER}@${MAXBOT_IP_1} 'cd ${DEPLOY_TAG}/deploy; \
			make set-init-pose x=0.6 y=0.4 ori_z=0 ori_w=1'; \
	fi
	@if [ -n "${MAXBOT_IP_2}" ]; then \
		echo "set init pose on ${MAXBOT_IP_2}"; \
		ssh ${REMOTE_USER}@${MAXBOT_IP_2} 'cd ${DEPLOY_TAG}/deploy; \
			make set-init-pose x=0.6 y=-0.4 ori_z=0 ori_w=1'; \
	fi
	@if [ -n "${MAXBOT_IP_3}" ]; then \
		echo "set init pose on ${MAXBOT_IP_3}"; \
		ssh ${REMOTE_USER}@${MAXBOT_IP_3} 'cd ${DEPLOY_TAG}/deploy; \
			make set-init-pose x=0 y=0.4 ori_z=0 ori_w=1'; \
	fi
	@if [ -n "${MAXBOT_IP_4}" ]; then \
		echo "set init pose on ${MAXBOT_IP_4}"; \
		ssh ${REMOTE_USER}@${MAXBOT_IP_4} 'cd ${DEPLOY_TAG}/deploy; \
			make set-init-pose x=0 y=-0.4 ori_z=0 ori_w=1'; \
	fi
	@if [ -n "${MAXBOT_IP_5}" ]; then \
		echo "set init pose on ${MAXBOT_IP_5}"; \
		ssh ${REMOTE_USER}@${MAXBOT_IP_5} 'cd ${DEPLOY_TAG}/deploy; \
			make set-init-pose x=-0.6 y=0.4 ori_z=0 ori_w=1'; \
	fi
	@if [ -n "${MAXBOT_IP_6}" ]; then \
		echo "set init pose on ${MAXBOT_IP_6}"; \
		ssh ${REMOTE_USER}@${MAXBOT_IP_6} 'cd ${DEPLOY_TAG}/deploy; \
			make set-init-pose x=-0.6 y=-0.4 ori_z=0 ori_w=1'; \
	fi

launch-navigation-all:
	@for REMOTE_HOST in ${MAXBOT_IPS}; do \
		echo "launch navigation on $${REMOTE_HOST}"; \
		ssh ${REMOTE_USER}@$${REMOTE_HOST} 'cd ${DEPLOY_TAG}/deploy; \
			make launch-navigation'; \
	done

kill-navigation-all:
	@for REMOTE_HOST in ${MAXBOT_IPS}; do \
		echo "kill navigation on $${REMOTE_HOST}"; \
		ssh ${REMOTE_USER}@$${REMOTE_HOST} 'cd ${DEPLOY_TAG}/deploy; \
			make kill-navigation'; \
	done
