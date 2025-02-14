"""
# @Time    : 2021/6/30 10:07 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : train.py
"""

# !/usr/bin/env python
import sys
import os
import setproctitle
import numpy as np
from pathlib import Path
import torch
import time
import pprint
import random
import yaml

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from light_mappo.config import get_config
from light_mappo.envs.env_wrappers import DummyVecEnv, SubprocVecEnv, IsaacSimEnv

from light_mappo.envs.isaac_sim import init_simulation_app

"""Train script for MPEs."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.

            from light_mappo.envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv(all_args, all_args.n_rollout_threads)

            # from envs.env_discrete import DiscreteActionEnv

            # env = DiscreteActionEnv()

            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env
    
    if all_args.env_type == "isaac_sim":
        return IsaacSimEnv(get_env_fn(0), all_args.n_rollout_threads)

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.
            from light_mappo.envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv()
            # from envs.env_discrete import DiscreteActionEnv
            # env = DiscreteActionEnv()
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env
    
    if all_args.env_type == "isaac_sim":
        return IsaacSimEnv(get_env_fn(0), all_args.n_eval_rollout_threads)

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)

    all_args = parser.parse_known_args(args)[0]

    return all_args

def modify_dr_config(config, new_para):
    new_para_list = new_para.split('_')
    new_para_list = [float(i) for i in new_para_list] 
    dr_config = config.get("domain_randomization", None)
    randomization_params = dr_config.get("randomization_params", None)
    if randomization_params is not None:
        for opt in randomization_params.keys():
            if opt == "observations":
                observations_dr_params = dr_config["randomization_params"]["observations"]
                observations_dr_params["on_interval"]["distribution_parameters"][0][1] = new_para_list[2]
                observations_dr_params["on_interval"]["distribution_parameters"][1][1] = new_para_list[3]
                observations_dr_params["on_interval"]["distribution_parameters"][2][1] = new_para_list[4]
                observations_dr_params["on_interval"]["distribution_parameters"][3][1] = new_para_list[5]
                observations_dr_params["on_interval"]["distribution_parameters"][4][1] = new_para_list[6]
            elif opt == "actions":
                actions_dr_params = dr_config["randomization_params"]["actions"]
                actions_dr_params["on_interval"]["distribution_parameters"][1] = new_para_list[1]
            elif opt == "rigid_prim_views":
                if randomization_params["rigid_prim_views"] is not None:
                    for view_name in randomization_params["rigid_prim_views"].keys():
                        if randomization_params["rigid_prim_views"][view_name] is not None:
                            force_dr_params = randomization_params["rigid_prim_views"][view_name]["force"]
                            force_dr_params["on_interval"]["distribution_parameters"][1][2] = new_para_list[0] 
    return config

def main(args):
    t1 = time.time()
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert (
        all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener"
    ) == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    setproctitle.setproctitle(
        str(all_args.experiment_name)
        + "-"
        +str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    pprint.pprint(vars(all_args))

    # create SimulationApp for import isaac sim modules
    simulation_app = init_simulation_app(all_args.isaac_sim_headless)
    from light_mappo.envs.isaac_sim.utils.scene import set_up_scene, set_up_new_scene, get_randomizer

    # set_up_scene(all_args.n_rollout_threads)
    world = set_up_new_scene(config={"all_args": all_args, "device": device,})
    # set_up_new_scene(all_args.n_rollout_threads, all_args.num_agents, all_args.use_randomize)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    # run experiments
    if all_args.share_policy:
        from light_mappo.runner.shared.env_runner import EnvRunner as Runner
    else:
        from light_mappo.runner.separated.env_runner import EnvRunner as Runner

    if all_args.use_randomize:
        with open(os.path.join(parent_dir, 'light_mappo/dr_config/domain_randomization.yaml'), 'r') as file:
            dr_config = yaml.safe_load(file)

    dr_para_list_0 = ['2_0.2_0.04_0.015_0.03_3_12', '2_0.2_0.03_0.02_0.03_3_12', '2_0.2_0.03_0.015_0.04_3_12', '2_0.2_0.03_0.015_0.03_4_12', '2_0.2_0.03_0.015_0.03_3_16', 
                    '2_0.2_0.04_0.02_0.03_3_12', '2_0.2_0.04_0.015_0.04_3_12', '2_0.2_0.04_0.015_0.03_4_12', '2_0.2_0.04_0.015_0.03_3_16', 
                    '2_0.2_0.03_0.02_0.04_3_12', '2_0.2_0.03_0.02_0.03_4_12', '2_0.2_0.03_0.02_0.03_3_16',
                    '2_0.2_0.03_0.015_0.04_4_12', '2_0.2_0.03_0.015_0.04_3_16',
                    '2_0.2_0.03_0.015_0.03_4_16']
    
    dr_para_list_1 = ['2_0.2_0.04_0.02_0.04_3_12', '2_0.2_0.04_0.02_0.03_4_12', '2_0.2_0.04_0.02_0.03_3_16', '2_0.2_0.04_0.015_0.04_4_12', '2_0.2_0.04_0.015_0.03_4_16', '2_0.2_0.04_0.015_0.04_3_16',
                    '2_0.2_0.03_0.02_0.04_4_12', '2_0.2_0.03_0.02_0.04_3_16', '2_0.2_0.03_0.02_0.03_4_16',
                    '2_0.2_0.03_0.015_0.04_4_16',
                    '2_0.2_0.04_0.02_0.04_4_12', '2_0.2_0.04_0.02_0.03_4_16', '2_0.2_0.04_0.02_0.04_3_16', '2_0.2_0.04_0.015_0.04_4_16',
                    '2_0.2_0.03_0.02_0.04_4_16']
    
    dr_para_list_2 = ['3_0.2_0.04_0.015_0.03_3_12', '3_0.2_0.03_0.02_0.03_3_12', '3_0.2_0.03_0.015_0.04_3_12', '3_0.2_0.03_0.015_0.03_4_12', '3_0.2_0.03_0.015_0.03_3_16', 
                    '3_0.2_0.04_0.02_0.03_3_12', '3_0.2_0.04_0.015_0.04_3_12', '3_0.2_0.04_0.015_0.03_4_12', '3_0.2_0.04_0.015_0.03_3_16', 
                    '3_0.2_0.03_0.02_0.04_3_12', '3_0.2_0.03_0.02_0.03_4_12', '3_0.2_0.03_0.02_0.03_3_16',
                    '3_0.2_0.03_0.015_0.04_4_12', '3_0.2_0.03_0.015_0.04_3_16',
                    '3_0.2_0.03_0.015_0.03_4_16']
    
    dr_para_list_3 = ['3_0.2_0.04_0.02_0.04_3_12', '3_0.2_0.04_0.02_0.03_4_12', '3_0.2_0.04_0.02_0.03_3_16', '3_0.2_0.04_0.015_0.04_4_12', '3_0.2_0.04_0.015_0.03_4_16', '3_0.2_0.04_0.015_0.04_3_16',
                    '3_0.2_0.03_0.02_0.04_4_12', '3_0.2_0.03_0.02_0.04_3_16', '3_0.2_0.03_0.02_0.03_4_16',
                    '3_0.2_0.03_0.015_0.04_4_16',
                    '3_0.2_0.04_0.02_0.04_4_12', '3_0.2_0.04_0.02_0.03_4_16', '3_0.2_0.04_0.02_0.04_3_16', '3_0.2_0.04_0.015_0.04_4_16',
                    '3_0.2_0.03_0.02_0.04_4_16']
    
    dr_para_list_4 = ['2_0.2_0.03_0.015_0.03_4_16']

    dr_para_list_5 = ['3_0.2_0.03_0.015_0.03_4_16']

    dr_para_list_30_0 = ['0']
    dr_para_list_30_1 = ['3_0.2_0.03_0.02_0.03_3_12']
    dr_para_list_30_2 = ['3_0.2_0.03_0.02_0.03_4_16']
    dr_para_list_30_3 = ['2_0.2_0.02_0.01_0.02_2_8']
    
    dr_para_list_20_0 = ['0']
    dr_para_list_20_1 = ['3_0.2_0.03_0.02_0.03_3_12']

    dr_para_list_10_0 = ['0']
    dr_para_list_10_1 = ['3_0.2_0.03_0.02_0.03_3_12']

    dr_para_list_hard_60_0 = ['0']
    dr_para_list_hard_60_1 = ['3_0.2_0.03_0.02_0.03_3_12']
    dr_para_list_hard_60_2 = ['2_0.2_0.02_0.01_0.02_2_8']
    dr_para_list_hard_60_3 = ['2_0.2_0.03_0.015_0.03_3_12']
    dr_para_list_hard_60_4 = ['2_0.3_0.02_0.01_0.02_2_8']
    reward_para_list = [(0.96, 0.04), (0.96, 0.05), (0.96, 0.06), (0.97, 0.04), (0.97, 0.05), (0.97, 0.06), (0.98, 0.04), (0.98, 0.05), (0.98, 0.06)]
    reward_para_list_1 = [(0.98, 0.05)]

    for i in range(len(dr_para_list_hard_60_1)):
        print(dr_para_list_hard_60_1[i])
        # run dir
        for dir_reward_thr, total_vel_thr in reward_para_list_1:
            envs.env.env.dir_reward_thr = dir_reward_thr
            envs.env.env.total_vel_thr = total_vel_thr
            run_dir = (
                Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
                / all_args.env_name
                / all_args.scenario_name
                / all_args.algorithm_name
                / all_args.experiment_name
                / "dr_test_model_hard_60_round2"
                / dr_para_list_hard_60_1[i]
            )
            if not run_dir.exists():
                os.makedirs(str(run_dir))

            # if not run_dir.exists():
            #     curr_run = "run1"
            # else:
            #     exst_run_nums = [
            #         int(str(folder.name).split("run")[1])
            #         for folder in run_dir.iterdir()
            #         if str(folder.name).startswith("run")
            #     ]
            #     if len(exst_run_nums) == 0:
            #         curr_run = "run1"
            #     else:
            #         curr_run = "run%i" % (max(exst_run_nums) + 1)
            curr_run = str(dir_reward_thr) + "_" + str(total_vel_thr)
            run_dir = run_dir / curr_run
            if not run_dir.exists():
                os.makedirs(str(run_dir))

            print(f"run_dir: {run_dir}")

            config = {
                "all_args": all_args,
                "envs": envs,
                "eval_envs": eval_envs,
                "num_agents": num_agents,
                "device": device,
                "run_dir": run_dir,
            }

            # domain randomization
            if all_args.use_randomize:
                dr_config = modify_dr_config(dr_config, dr_para_list_hard_60_1[i])
                print('model!!!!!!', dr_config)
                print(f"run_dir: {run_dir}")

                _randomizer = get_randomizer(world, config, dr_config)

                if _randomizer:
                    _randomizer.set_up_domain_randomization()
                    envs.env.env.dr_randomizer = _randomizer
            
            runner = Runner(config)
            runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()

    print(f"train time: {time.time() - t1}")
    simulation_app.close()


if __name__ == "__main__":
    main(sys.argv[1:])
