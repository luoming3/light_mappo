#!/bin/bash

env="MyEnv"
scenario="MyEnv"
num_landmarks=3
num_agents=4
algo="mappo"  # default="mappo", choices=["rmappo", "mappo"]
exp="check"
seed=1
model_dir=""  # TODO: path to your model dir

current_dir=$(cd $(dirname $0); pwd)

echo "env is ${env}"

${ISAAC_SIM_FOLDER}/python.sh ${current_dir}/render/render.py --env_name ${env} --algorithm_name ${algo} \
--experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
--n_training_threads 1 --n_render_rollout_threads 1 --episode_length 1000 --render_episodes 5 \
--model_dir ${model_dir} --isaac_sim_headless
