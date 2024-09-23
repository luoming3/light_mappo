help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=mappo

run-container:
	@docker run --init -it -d --name ${CONTAINER_NAME} \
	-v $(shell pwd)/models:/app/deploy/models \
	${IMAGE_NAME}:${IMAGE_TAG}

remove-container:
	@docker rm -f ${CONTAINER_NAME}

exec-container:
	@docker exec -it ${CONTAINER_NAME} bash

load-image:
	@for i in $(shell ls image/*.gz); do \
		docker load < $$i ; \
	done
	@echo "load image successfully"
