help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=mappo

run-all:
	bash ./run_all.sh

build-image:
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

save-image:  # save docker image
	docker save ${IMAGE_NAME}:${IMAGE_TAG} | gzip > deploy/image/${IMAGE_NAME}_${IMAGE_TAG}.tar.gz

package:
	@tar -cvf deploy.tar deploy
