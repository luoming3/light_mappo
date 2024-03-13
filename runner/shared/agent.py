# coding=utf-8

import os
import sys
import time

import numpy as np
import imageio
import torch

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy  # noqa: E402
from config import get_config  # noqa: E402
from envs.env_continuous import ContinuousActionEnv  # noqa: E402
from utils.util import _t2n  # noqa: E402


class Agent(object):
    def __init__(self, all_args, env):
        self.policy = Policy(
            all_args,
            env.observation_space[0],
            env.share_observation_space[0],
            env.action_space[0],
        )
        self.args = all_args
        self.env = env

        if all_args.model_dir is not None:
            self.restore(all_args.model_dir)

    def restore(self, model_dir=None):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(model_dir) + "/actor.pt")
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.args.use_render:
            policy_critic_state_dict = torch.load(str(model_dir) + "/critic.pt")
            self.policy.critic.load_state_dict(policy_critic_state_dict)

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

    def act(self, obs):
        rnn_states = np.zeros(
            (
                1,
                self.env.num_agent,
                self.args.recurrent_N,
                self.args.hidden_size,
            ),
            dtype=np.float32,
        )
        masks = np.ones((1, self.env.num_agent, 1), dtype=np.float32)

        actions, _ = self.policy.act(
            obs,
            np.concatenate(rnn_states),
            np.concatenate(masks),
            deterministic=True,
        )

        actions = _t2n(actions)
        if self.env.action_space[0].__class__.__name__ == "Box":
            actions = np.tanh(actions)
        else:
            raise NotImplementedError

        return actions


@torch.no_grad()
def main(args):
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
            result = env.step(actions)
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
    args = sys.argv[1:]
    args.extend(
        [
            "--model_dir",
            "/home/lscm/git/light_mappo/results/box2d_car/share/mappo/luoming/run14/models",
            '--use_render',
        ]
    )
    main(args)
