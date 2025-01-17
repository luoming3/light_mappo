#!/bin/bash


env="MyEnv"
scenario="MyEnv"
num_landmarks=3
num_agents=4
algo="mappo"  # default="mappo", choices=["rmappo", "mappo"]
exp="check"
seed=1
model_dir="scripts/results/MyEnv/MyEnv/mappo/check/run68/models/latest"  # TODO: path to your model dir

current_dir=$(cd $(dirname $0); pwd)

echo "env is ${env}"

# # 遍历指定路径下的所有文件夹
# for dir in "$model_dir"/*/; do 
#     if [ -d "$dir" ]; then
#         output_file="${dir}badcase.log"
#         echo "文件夹: $output_file"
#         CUDA_VISIBLE_DEVICES=1 /home/user/.local/share/ov/pkg/isaac-sim-2023.1.1/python.sh -u ${current_dir}/render/render.py --env_name ${env} --algorithm_name ${algo} \
#         --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
#         --n_training_threads 1 --n_render_rollout_threads 1 --episode_length 2048 --use_render --render_badcase \
#         --model_dir ${dir} --layer_N 3 --hidden_size 128 > "$output_file" 2>&1
#     fi
# done

CUDA_VISIBLE_DEVICES=0 python -u ${current_dir}/render/render.py --env_name ${env} --algorithm_name ${algo} \
--experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
--n_training_threads 1 --n_render_rollout_threads 1 --episode_length 2048 --use_render --render_badcase \
--model_dir ${model_dir} --layer_N 3 --hidden_size 128
