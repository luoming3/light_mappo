import os
import sys
import time

import imageio
import numpy as np

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from runner.shared.agent import Agent
from config import get_config
from envs.env_continuous import ContinuousActionEnv


def test_step_asyn():
    args = [
        "--model_dir",
        "C:\\Users\\ming.luo\\git\\light_mappo\\results\\MyEnv\\MyEnv\\mappo\\check\\run2\\models",
        "--use_render",
    ]
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    env = ContinuousActionEnv()
    agent = Agent(all_args, env)

    for i in range(1):
        step = 0
        episode_reward = 0
        all_frames = []

        obs = env.reset()
        image = env.render()
        all_frames.append(image)

        while True:
            agent.prep_rollout()
            actions = agent.act(obs)
            result = env.step_asyn(actions)
            all_frames.append(env.render())
            step += 1

            obs, reward, done, _ = result
            episode_reward += reward[0][0]
            if np.any(done) or step == 256:
                break

        image_dir = os.path.dirname(__file__) + "/" + "image"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        time_now = int(time.time() * 1000)
        gif_save_path = image_dir + f"/{time_now}_{step}_{episode_reward:.2f}.gif"
        imageio.mimsave(gif_save_path, all_frames, duration=1, loop=0)


if __name__ == "__main__":
    test_step_asyn()
