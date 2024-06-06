#!/bin/sh
env="isaac-sim"
scenario="share"
num_landmarks=3
num_agents=4
algo="mappo"  # default="mappo", choices=["rmappo", "mappo"]
exp=""  # TODO: use your name
seed=1

current_dir=$(cd $(dirname $0); pwd)

echo "env is ${env}"

CUDA_VISIBLE_DEVICES=0 python -u ${current_dir}/../train/train.py --env_name ${env} --algorithm_name ${algo} \
--experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
--n_training_threads 4 --n_rollout_threads 32 --episode_length 1024 --layer_N 1 --hidden_size 64 --num_env_steps 300000000
