help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=mappo
MAPPO_NODE_NAME=mappo_node

clean-up:
	@rm -rf ros_ws/build
	@rm -rf ros_ws/devel
	@rm -f ros_ws/src/CMakeLists.txt

build-image: clean-up
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

save-image:  # save docker image
	docker save ${IMAGE_NAME}:${IMAGE_TAG} | gzip > deploy/image/${IMAGE_NAME}_${IMAGE_TAG}.tar.gz

package:
	@rm -f deploy.tar*
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

# ros command
launch-gmapping:
	roslaunch maxbot_real maxbot_real_gmapping.launch

launch-save-map:
	roslaunch maxbot_real maxbot_real_map_save.launch

launch-amcl:
	roslaunch maxbot_real maxbot_real_amcl_pose.launch

launch-navigation:
	roslaunch maxbot_real maxbot_real_navigation.launch

run-mappo: # e.g. make run-mappo start=1,1 goal=0,0
	@bash ./run_mappo_node.sh

kill-mappo:
	@rosnode kill ${MAPPO_NODE_NAME}
	@rostopic pub -1 /cmd_vel geometry_msgs/Twist '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}'
