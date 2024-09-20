## 1. development

### 1.1. conda

```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
source ~/.bashrc && ~/miniconda3/bin/conda init bash
```

### 1.2. 2d environment

Simply download the code, create a Conda environment, and then run the code, adding packages as needed. Specific packages will be added later.

```
conda create -n marl python=3.8.18
conda activate marl
 
# cuda 10.1
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
 
# cuda 11.0
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
 
pip install -r requirements.txt
```
### 1.3. 3d environment

For isaac sim environment, the following command should be used.

```sh
cd path_to_isaac_sim_workspace_folder
conda env create -f environment.yml
conda activate isaac-sim

# set up environment variables.
# execute this command when open a new terminal or you can 
# set this command in ~/.bashrc so you only need to do it once
source setup_conda_env.sh

# install cuda 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# install box2d==2.3.10 (if it requires swig, install swig in "sudo apt install swig")
# pip install https://github.com/pybox2d/pybox2d/archive/refs/tags/2.3.10.tar.gz
# install requirements
cd path_to_light_mappo_folder
pip install -r requirements_3_10.txt
```

### 1.4. train

1. modify *scripts/train.sh* to adjust args
2. `bash scripts/train.sh` (under the project directory)

### 1.5. render

1. modify *scripts/render.sh*, select your model path
2. `bash scripts/render.sh` (under the project directory)

## 2. deployment

### 2.1. package

1. build image
    - `make build-image` (under project folder)
2. save image
    - `make save-image` (under project folder)
3. copy **actor.pt** to `deploy/models` directory
4. compression deploy
    - `make package` (under project folder)
5. scp the package `deploy.tar` to remote maxbot

### 2.2. deployment

1. create new directory
    - `TAG=mappo-$(date +%Y%m%d)`
    - `mkdir -p /home/ubuntu/${TAG}`
2. move the package `deploy.tar` to the created directory
3. uncompression
    - `cd /home/ubuntu/${TAG}`
    - `tar -xvf deploy.tar`
4. load image
    - `cd /home/ubuntu/${TAG}/deploy`
    - `make load-image`
5. run container
    - `make run-container`
