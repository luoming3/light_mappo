import numpy as np
from gym import spaces

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

import torch
from typing import Optional, Union

import os
import sys

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from envs.env_2d import map, plotting, Astar  # noqa: E402
from envs.isaac_sim.utils.scene import get_world, set_up_scene


class EnvCore(object):
    def __init__(self) -> None:
        # isaac sim environment
        self.world = get_world()
        self.env_num = 2  # TODO: setting in config.py
        self.scene = self.world.scene

        self.agent_num = 4  # number of agent
        self.obs_dim = 10  # observation dimension of agents
        self.action_dim = 3  # set the action dimension of agents
        self.env_indices = [i for i in range(self.env_num)]
        self.action_space = spaces.Box(
            np.array([-10, -10]).astype(np.float32),
            np.array([+10, +10]).astype(np.float32),
        )  # left_wheel velocity and right_wheel velocity
        self.dest = np.zeros(shape=(self.env_num, 2))

    def reset(self, indices=[]):
        if len(indices) == 0:
            indices = self.env_indices

        car_position = np.random.randint(low=0, high=6, size=(len(indices), 2))
        self.car_position = np.concatenate(
            (
                car_position,
                np.zeros(shape=(self.env_num,1))
            ),
            axis=1
        )
        # reset cars' position and velocity
        self._reset_idx(positions=self.car_position, orientations=None, indices=indices)

        # reset targets' positions
        self.dest[indices] = np.random.randint(low=0, high=6, size=(len(indices), 2))

        # observations, shape is (env_num, agent_num, obs_dim)
        observations = self.get_observations()

        return observations

    # TODO
    def step(self, actions):
        '''
        return [[obs, reward, done, info], [obs, reward, done, info], ...] which contains all envs' info
        '''
        self.set_actions(actions)
        self.world.step(render=False)

        result = []
        for i in range(self.env_num):
            sub_agent_obs = []
            sub_agent_reward = []
            sub_agent_done = []
            sub_agent_info = []
            for i in range(self.agent_num):
                sub_agent_obs.append(np.random.random(size=(self.obs_dim,)))
                sub_agent_reward.append([np.random.rand()])
                sub_agent_done.append(False)
                sub_agent_info.append({})
            result.append([sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info])

        return result

    def get_observations(self):
        '''
        return observations whose dim is (env_num, agent_num, obs_spaces)
        '''
        car = self.scene.get_object("car_view")
        jetbot_view = self.scene.get_object("jetbot_chassis_view")

        positions, _ = car.get_world_poses()
        # only need x,y axis
        car_position = positions[:, 0:2]
        car_position = np.expand_dims(car_position, 1).repeat(self.agent_num, axis=1)
        
        # only need x,y axis
        jetbot_linear_velocities = jetbot_view.get_linear_velocities()[:, 0:2]
        jetbot_linear_velocities = jetbot_linear_velocities.reshape(self.env_num, self.agent_num, 2)
        # only need z axis
        jetbot_angular_velocities = jetbot_view.get_angular_velocities()[:, 2]
        jetbot_angular_velocities = jetbot_angular_velocities.reshape(self.env_num, self.agent_num, 1)

        jetbot_position, jetbot_orientation = jetbot_view.get_world_poses()
        # only need x,y axis
        jetbot_position = jetbot_position[:, 0:2]
        jetbot_position = jetbot_position.reshape(self.env_num, self.agent_num, 2)
        # only need z axis
        jetbot_orientation = jetbot_orientation[:, 3]
        jetbot_orientation = jetbot_orientation.reshape(self.env_num, self.agent_num, 1)

        dest_position = np.expand_dims(self.dest, 1).repeat(self.agent_num, axis=1)

        observations = np.concatenate(
            (
                dest_position,
                car_position,
                jetbot_linear_velocities,
                jetbot_angular_velocities,
                jetbot_position,
                jetbot_orientation
            ),
            axis=2
        )

        return observations
    
    # TODO
    def set_actions(self, actions):
        cars = self.scene.get_object("car_view")
        cars.set_joint_velocities(
            velocities=np.array(
                [[10., 10., 10., 10., 10., 10., 10., 10.],
                [-10., -10., -10., -10., -10., -10., -10., -10.]]
            ), 
            joint_indices=np.arange(4,12)  # revoluted joint indices
        )

    # TODO
    def render(self, mode="rgb_array"):
        pass

    def set_world_poses(
            self,
            positions: Optional[Union[np.ndarray, torch.Tensor]] = None,
            orientations: Optional[Union[np.ndarray, torch.Tensor]] = None,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,):
        car = self.scene.get_object("car_view")
        car.set_world_poses(positions, orientations, indices)

    def _reset_idx(
            self,
            positions: Optional[Union[np.ndarray, torch.Tensor]] = None,
            orientations: Optional[Union[np.ndarray, torch.Tensor]] = None,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,):
        """
        Reset the articulations to their default states
        """
        cars = self.scene.get_object("car_view")
        cars.set_world_poses(positions, orientations, indices)
        cars.set_velocities(np.zeros(6), indices=indices)