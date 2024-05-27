## 1. build docker image

### 1.1. build the base image

```sh
# clone the base image
mkdir -p ~/git
cd ~/git && git clone https://github.com/NVIDIA-Omniverse/IsaacSim-dockerfiles.git

# log in to NGC
docker login nvcr.io

# build isaac sim image for ubuntu20.04
docker build --pull -t \
  isaac-sim:2023.1.0-ubuntu20.04 \
  --build-arg ISAACSIM_VERSION=2023.1.0 \
  --file Dockerfile.2023.1.0-ubuntu20.04 .

# build isaac sim image for ubuntu22.04
docker build --pull -t \
  isaac-sim:2023.1.0-ubuntu22.04 \
  --build-arg ISAACSIM_VERSION=2023.1.0 \
  --file Dockerfile.2023.1.0-ubuntu22.04 .
```

### 1.2. build isaacsim-vnc-ros image

under the `your_project_path/envs/isaac_sim/docker/vnc_isaac_sim` folder:

```sh
# ubuntu20.04
docker build -t isaac-sim:2023.1.0-ubuntu20.04-noetic -f Dockerfile.2023.1.0-ubuntu20.04-noetic .

# ubuntu22.04
docker build -t isaac-sim:2023.1.0-ubuntu22.04-vnc -f Dockerfile.2023.1.0-ubuntu22.04-vnc .
```

## 2. usage

1. `docker run -d -it --gpus all -p ${expose_port}:80 isaac-sim:2023.1.0-ubuntu20.04-noetic`
    - modify the expose_port
2. Access `http://172.16.2.197:${expose_port}/vnc.html` with Chrome
    - default vncpasswd is `123456`
3. fix 'Failed to execute default Terminal Emulator.'
    - select *Applications > Settings > Preferred Applications > Utilities > Terminal Emulator* to Xfce Terminal
4. open isaac sim
    - open terminal
    - run `bash /isaac-sim/isaac-sim.sh --allow-root`
5. kill Xfce4 Power Manager avoiding blank screen
    - `ps -ef | grep power-manager | awk '{print $2}' | xargs kill -9`

### 2.1. example

```sh
docker run -d -it --gpus all -p 7020:80 \
    -v /home/tonyli/luoming/ros_ws:/root/ros_ws \
    isaac-sim:2023.1.0-ubuntu20.04-noetic
```

## 3. reference

- [Container Installation — Omniverse IsaacSim latest documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html)
- [Tiryoh/docker-ros-desktop-vnc: 🐳 Dockerfiles to provide HTML5 VNC interface to access Ubuntu LXDE + ROS](https://github.com/Tiryoh/docker-ros-desktop-vnc)