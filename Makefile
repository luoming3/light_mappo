help:  # list all targets
	@grep ^[a-z] Makefile

IMAGE_NAME=mappo
IMAGE_TAG=ros-noetic-focal
CONTAINER_NAME=mappo
MAPPO_NODE_NAME=mappo_node
AMCL_INIT_NODE_NAME=amcl_init_node

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

run-vnc-container:
	make -C deploy/ run-vnc-container

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
	@if rosnode list | grep -o ${MAPPO_NODE_NAME} ; then \
		rosnode kill ${MAPPO_NODE_NAME} ; \
	fi
	@rostopic pub -1 /cmd_vel geometry_msgs/Twist '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}'
	@echo "kill mappo_node and stop maxbot"

init-amcl:
	@python3 ./ros_ws/src/maxbot_real/scripts/amcl_init_node.py

kill-init-amcl:
	@if rosnode list | grep -o ${AMCL_INIT_NODE_NAME} ; then \
		rosnode kill ${AMCL_INIT_NODE_NAME} ; \
	fi
	@rostopic pub -1 /cmd_vel geometry_msgs/Twist '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}'
	@echo "kill init_amcl and stop maxbot"

clean-node: kill-init-amcl kill-mappo
	@echo "y" | rosnode cleanup

init: init-amcl

rotate: # TODO
	@echo "rotate maxbot to the inital state"
