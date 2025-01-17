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

### 1.4. docker env

see `light_mappo/envs/isaac_sim/docker/README.md`

### 1.5. train

1. modify *config.py* to adjust args
    - add new argument `--num_save_model`
2. `python scripts/train/train.py` (under the project directory)

### 1.6. render

1. modify *scripts/render.sh*, select your model path
    - add new argument `--use_render`
2. `bash scripts/render.sh` (under the project directory)

#### 1.6.1. render bad case

- modify *scripts/render_badcase.sh*, select your model path
    - add new argument `--use_render` and `--render_badcase` and `--isaac_sim_headless`
    - change argument `--n_render_rollout_threads` to 1
- `bash scripts/render_badcase.sh`
- select 'target_cube' in isaac-sim interface, then press the 'F' key to check the car 

## 2. package & deployment

### 2.1. package

1. build image
    - `make build-image` (under project folder)
2. save image
    - `make save-image` (under project folder)
3. copy **actor.pt** to `deploy/models` directory
4. compression deploy
    - `make package` (under project folder)
5. scp the package `deploy.tar` and `deploy.tar.sha256` to remote maxbot

### 2.2. deployment

1. create new directory
    - export DEPLOY_TAG variable, see **DEPLOY_TAG** in *deploy/Makefile*
        - e.g. `export DEPLOY_TAG=release-20241118`
    - `mkdir -p ~/${DEPLOY_TAG}`
2. move the package `deploy.tar` and `deploy.tar.sha256` to the created directory
3. check and uncompress
    - `cd ~/${DEPLOY_TAG}`
    - check sha256 file: `sha256sum -c deploy.tar.sha256`
    - if package OK, uncompress package: `tar -xvf deploy.tar`
4. load image
    - `cd ~/${DEPLOY_TAG}/deploy`
    - `make load-image`
5. run container
    - `cd ~/${DEPLOY_TAG}/deploy`
    - `make run-container`

### 2.3. navigation usage for single maxbot

1. enter deploy diretory
    - `cd ~/${DEPLOY_TAG}/deploy`
2. launch navigation
    - `make launch-navigation`
3. initialize amcl for convergence
    - `make init`
4. run mappo node for given start and goal
    - `make run-mappo start=1,1 goal=0,0`
5. kill mappo node for termination
    - `make kill-mappo`

### 2.4. navigation usage for all maxbot

1. enter deploy diretory
    - `cd ~/${DEPLOY_TAG}/deploy`
2. modify MAXBOT_IP and set up ssh config
    - `make setup-ssh`
3. launch all navigation
    - `make launch-navigation-all`
4. initialization before run mappo
    - `make init-all`
5. run mappo
    - `make start-all start=0,0 goal=2,2`
6. stop mappo
    - `make stop-all`
