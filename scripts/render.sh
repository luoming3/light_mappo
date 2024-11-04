#!/bin/bash

env="MyEnv"
scenario="MyEnv"
num_landmarks=3
num_agents=4
algo="mappo"  # default="mappo", choices=["rmappo", "mappo"]
exp="check"
seed=10
model_dir=""  # TODO: path to your model dir

current_dir=$(cd $(dirname $0); pwd)

echo "env is ${env}"

# 遍历指定路径下的所有文件夹
CUDA_VISIBLE_DEVICES=0 python -u ${current_dir}/render/render.py --env_name ${env} --algorithm_name ${algo} \
--experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
--n_training_threads 1 --n_render_rollout_threads 2000 --episode_length 2048 --render_episodes 10000 --use_render \
--model_dir ${model_dir} --layer_N 3 --hidden_size 128 --skip_frame 6 --truncation_step 256
