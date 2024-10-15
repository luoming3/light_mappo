help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=mappo

run-container:
	@docker run --init -it -d --name ${CONTAINER_NAME} --network=host \
	-v $(shell pwd)/models:/app/deploy/models \
	-v $(shell pwd)/ros_ws/src/maxbot_real/map:/app/ros_ws/src/maxbot_real/map \
	${IMAGE_NAME}:${IMAGE_TAG}

run-vnc-container:
	@docker run --init -it -d --name ${CONTAINER_NAME}-vnc --network=host \
	-v $(shell pwd)/models:/app/deploy/models \
	-v $(shell pwd)/ros_ws/src/maxbot_real/map:/app/ros_ws/src/maxbot_real/map \
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
