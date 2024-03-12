# coding=utf-8

import torch

from runner.shared.env_runner import EnvRunner
from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy
from envs.env_continuous import ContinuousActionEnv


class Agent(object):
    def __init__(self, all_args, env: ContinuousActionEnv, device="cpu") -> None:
        self.policy = Policy(
            all_args,
            env.observation_space[0],
            env.share_observation_space,
            env.action_space[0],
            device,
        )


    def restore(self, model_dir=None):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(str(model_dir) + '/critic.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)


def main():
    pass


if __name__ == "__main__":
    main()
