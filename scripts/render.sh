#!/bin/bash

env="MyEnv"
scenario="MyEnv"
num_landmarks=3
num_agents=4
algo="mappo"  # default="mappo", choices=["rmappo", "mappo"]
exp="check"
seed=7
model_dir="/home/user/tony/isaac_sim_maxbot/light_mappo/results/run2/models"  # TODO: path to your model dir

current_dir=$(cd $(dirname $0); pwd)

echo "env is ${env}"

CUDA_VISIBLE_DEVICES=0 /home/user/.local/share/ov/pkg/isaac-sim-2023.1.1/python.sh ${current_dir}/render/render.py --env_name ${env} --algorithm_name ${algo} \
--experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
--n_training_threads 1 --n_render_rollout_threads 1 --episode_length 2048 --render_episodes 100 \
--model_dir ${model_dir} --use_render --layer_N 3 --hidden_size 128 # --isaac_sim_headless 
