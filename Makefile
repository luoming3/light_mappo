help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=light_mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=light_mappo

run-all:
	bash ./run_all.sh

build-image:
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

save-image:  # save docker image
	docker save ${IMAGE_NAME}:${IMAGE_TAG} | gzip > deploy/image/${IMAGE_NAME}_${IMAGE_TAG}.tar.gz
