#!/usr/bin/env python
import sys
import os
import setproctitle
import numpy as np
from pathlib import Path
import torch
import pprint

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from light_mappo.config import get_config  # noqa: E402
from light_mappo.envs.env_wrappers import DummyVecEnv, IsaacSimEnv  # noqa: E402

from light_mappo.envs.isaac_sim import init_simulation_app

def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.

            from light_mappo.envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv(all_args, all_args.n_render_rollout_threads)

            # from envs.env_discrete import DiscreteActionEnv

            # env = DiscreteActionEnv()

            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env
        
    if all_args.env_type == "isaac_sim":
        return IsaacSimEnv(get_env_fn(0), all_args.n_render_rollout_threads)

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_render_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='simple', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert all_args.use_render, ("u need to set use_render be True")
    assert not (all_args.model_dir is None or all_args.model_dir == ""), ("set model_dir first")
    
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

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # create SimulationApp for import isaac sim modules
    simulation_app = init_simulation_app(all_args.isaac_sim_headless)
    from light_mappo.envs.isaac_sim.utils.scene import set_up_scene, set_up_new_scene

    # set_up_scene(all_args.n_render_rollout_threads)
    set_up_new_scene(env_num=all_args.n_render_rollout_threads, bot_num=all_args.num_agents)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    pprint.pprint(vars(all_args))

    # run experiments
    if all_args.share_policy:
        from light_mappo.runner.shared.env_runner import EnvRunner as Runner
    else:
        from light_mappo.runner.separated.env_runner import EnvRunner as Runner

    # # for bad case
    if all_args.render_badcase:
        print('badcase!!!!!!')
        runner = Runner(config)
        runner.render_specific_episode()
    else:
        model_dir_input = Path(all_args.model_dir)
        model_list = get_model_list(model_dir_input)
        for model_dir in model_list:
            all_args.model_dir = model_dir
            runner = Runner(config)
            runner.render()

    # post process
    envs.close()
    simulation_app.close()


def get_model_list(model_dir_input: Path):
    if not model_dir_input.exists():
        raise RuntimeError("model_dir is no exist")
    
    model_list = model_dir_input.glob("**/actor.pt")
    model_dir_list = [str(model.parent) for model in model_list]
    return model_dir_list


if __name__ == "__main__":
    main(sys.argv[1:])
