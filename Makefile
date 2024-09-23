help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=mappo

build-image:
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

save-image:  # save docker image
	docker save ${IMAGE_NAME}:${IMAGE_TAG} | gzip > deploy/image/${IMAGE_NAME}_${IMAGE_TAG}.tar.gz

package:
	@tar -cvf deploy.tar deploy

run-container:
	make -C deploy/ run-container

remove-container:
	make -C deploy/ remove-container

exec-container:
	make -C deploy/ exec-container

# ros command
launch-gmapping:
	roslaunch maxbot_real maxbot_real_gmapping.launch

launch-save-map:
	roslaunch maxbot_real maxbot_real_map_save.launch

launch-amcl:
	roslaunch maxbot_real maxbot_real_amcl_pose.launch
