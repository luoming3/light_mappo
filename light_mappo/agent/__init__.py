import os
import numpy as np
import torch
from gym import spaces

from light_mappo.algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy
from light_mappo.config import get_config
from light_mappo.utils.util import _t2n
from light_mappo.envs.isaac_sim.env_core import OBS_DIM, ACTION_SPACE

MODEL_DIR = "/app/deploy/models/actor.pt"


class Agent(object):

    def __init__(self):
        parser = get_config()
        all_args = parser.parse_known_args()[0]

        observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(OBS_DIM, ),
            dtype=np.float32,
        )
        share_observation_space = spaces.Box(low=-np.inf,
                                             high=+np.inf,
                                             shape=(OBS_DIM, ),
                                             dtype=np.float32)
        self.policy = Policy(
            all_args,
            observation_space,
            share_observation_space,
            ACTION_SPACE,
        )
        self.args = all_args
        self.num_agent = 1
        self.num_env = 1

        if os.path.isfile(MODEL_DIR):
            self.restore(MODEL_DIR)
        else:
            raise RuntimeError("Model file is not existed!")

    def restore(self, model_dir=None):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(model_dir,
                                             map_location=torch.device("cpu"))
        self.policy.actor.load_state_dict(policy_actor_state_dict)

    def prep_rollout(self):
        self.policy.actor.eval()

    def act(self, obs):
        rnn_states = np.zeros(
            (
                self.num_env,
                self.num_agent,
                self.args.recurrent_N,
                self.args.hidden_size,
            ),
            dtype=np.float32,
        )
        masks = np.ones((self.num_env, self.num_agent, 1), dtype=np.float32)

        actions, _ = self.policy.act(
            obs,
            np.concatenate(rnn_states),
            np.concatenate(masks),
            deterministic=True,
        )

        actions = _t2n(actions)
        if ACTION_SPACE.__class__.__name__ == "Box":
            actions = np.tanh(actions) * 0.16
        else:
            raise NotImplementedError

        return actions[0]
