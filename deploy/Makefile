help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=light_mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=light_mappo

run-container:
	@docker run --init -it -d --name ${CONTAINER_NAME} \
	-v ./models:/app/deploy/models \
	${IMAGE_NAME}:${IMAGE_TAG}

remove-container:
	@docker rm -f ${CONTAINER_NAME}

exec-container:
	@docker exec -it ${CONTAINER_NAME} bash