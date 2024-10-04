# light_mappo

## Installation

### conda

```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
source ~/.bashrc && ~/miniconda3/bin/conda init bash
```

### 2d environment

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
### 3d environment

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

## Usage

- The environment part is an empty implementation, and the implementation of the environment part in the light_mappo/envs/env_core.py file is: [Code] (https://github.com/tinyzqh/light_mappo/blob/main/envs/env_core.py)

```python
import numpy as np
class EnvCore(object):
    """
    # Environment Agent
    """
    def __init__(self):
        self.agent_num = 2 # set the number of agents(aircrafts), here set to two
        self.obs_dim = 14 # set the observation dimension of agents
        self.action_dim = 5 # set the action dimension of agents, here set to a five-dimensional

    def reset(self):
        """
        # When self.agent_num is set to 2 agents, the return value is a list, and each list contains observation data of shape = (self.obs_dim,)
        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14, ))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # When self.agent_num is set to 2 agents, the input of actions is a two-dimensional list, and each list contains action data of shape = (self.action_dim,).
        # By default, the input is a list containing two elements, because the action dimension is 5, so each element has a shape of (5,)
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(np.random.random(size=(14,)))
            sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
```


Just write this part of the code, and you can seamlessly connect with MAPPO. After env_core.py, two files, env_discrete.py and env_continuous.py, were separately extracted to encapsulate the action space and discrete action space. In elif self.continuous_action: in algorithms/utils/act.py, this judgment logic is also used to handle continuous action spaces. The # TODO here in runner/shared/env_runner.py is also used to handle continuous action spaces.

In the train.py file, choose to comment out continuous environment or discrete environment to switch the demo environment.

### train

1. modify *config.py* to adjust args
2. `python train/train.py` (under the project directory)

### render

- modify *scripts/render.sh*, select your model path
- `bash scripts/render.sh` (under the project directory)
    - get the gif in *scrips/result/run/gif*

### bad case rendering

- change the function from `runner.render()` to `runner.render_specific_episode()`
- make sure the argument `--n_render_rollout_threads` is set to 1
- the output_file need to be changed anything else but `result.log` in render.sh