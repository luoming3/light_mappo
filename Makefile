help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=mappo

clean-up:
	@rm -rf ros_ws/build
	@rm -rf ros_ws/devel
	@rm -f ros_ws/src/CMakeLists.txt

build-image: clean-up
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

save-image:  # save docker image
	docker save ${IMAGE_NAME}:${IMAGE_TAG} | gzip > deploy/image/${IMAGE_NAME}_${IMAGE_TAG}.tar.gz

save-vnc-image:
	docker save ${IMAGE_NAME}:${IMAGE_TAG}-vnc | gzip > deploy/image/${IMAGE_NAME}_${IMAGE_TAG}_vnc.tar.gz

package:
	@rm -f deploy.tar*
	@rm -rf deploy/logs/*.log
	@tar -cvf deploy.tar deploy
	@sha256sum deploy.tar > deploy.tar.sha256

run-container:
	make -C deploy/ run-container

remove-container:
	make -C deploy/ remove-container

restart-container:
	make -C deploy/ restart-container

exec-container:
	make -C deploy/ exec-container

build-vnc-image: build-image
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG}-vnc -f Dockerfile_vnc .

update-tag:
	@last_release=$(shell grep -P -o "release-\d*" README.md) ; \
	latest_release=release-$(shell date +"%Y%m%d") ; \
	sed -i "s/$${last_release}/$${latest_release}/g" deploy/Makefile ; \
	sed -i "s/$${last_release}/$${latest_release}/g" README.md
