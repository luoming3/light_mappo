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
        self.car_view = self.world.scene.get_object("car_view")
        self.jetbot_view = self.world.scene.get_object("jetbot_chassis_view")

        self.agent_num = 4  # number of agent
        self.obs_dim = 10  # observation dimension of agents
        self.action_dim = 3  # set the action dimension of agents
        self.env_indices = [i for i in range(self.env_num)]
        self.action_space = spaces.Box(
            np.array([-10, -10]).astype(np.float32),
            np.array([+10, +10]).astype(np.float32),
        )  # left_wheel velocity and right_wheel velocity
        self.dest = np.zeros(shape=(self.env_num, 2))
        self.steps = np.zeros(shape=(self.env_num, 1))

    def reset(self, indices=[]):
        if len(indices) == 0:
            indices = self.env_indices

        indices_len = len(indices)
        car_position = np.random.randint(low=0, high=6, size=(indices_len, 2))
        self.car_position = np.concatenate(
            (
                car_position,
                self.car_view._default_state.positions[indices, 2:3]
            ),
            axis=1
        )
        # reset cars' position and velocity
        self._reset_idx(positions=self.car_position, orientations=None, indices=indices)

        # reset targets' positions
        self.dest[indices] = np.random.randint(low=0, high=6, size=(indices_len, 2))

        # observations, shape is (env_num, agent_num, obs_dim)
        observations = self.get_observations()

        self.steps[indices] = 0

        return observations

    def step(self, actions):
        '''
        return obs, reward, done, info which contains all envs' info where
            shape:
                obs: (self.env_num, self.agent_num, self.obs_dim)
                reward: (self.env_num, self.agent_num, 1)
                done: (self.env_num, self.agent_num)
                info: (self.env_num, self.agent_num)
        '''
        previous_car_position= self.car_view.get_world_poses()[0][:, 0:2]

        self.set_actions(actions)
        self.world.step(render=False)

        current_car_position = self.car_view.get_world_poses()[0][:, 0:2]
        goal_world_position = self.dest

        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_car_position, axis=1)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_car_position, axis=1)

        # running
        env_reward = (previous_dist_to_goal - current_dist_to_goal).reshape(self.env_num, -1)
        env_done = np.zeros((self.env_num, 1), dtype=bool)
        
        # arrival
        arrival_indices = np.where(current_dist_to_goal < 0.2)
        env_done[arrival_indices] = True
        env_reward[arrival_indices] = 10

        # failure
        failure_indices = np.where(current_dist_to_goal > 10)
        env_done[failure_indices] = True
        env_reward[failure_indices] = 0

        # truncation
        truncation_indices = np.where(self.steps==1000)
        env_done[truncation_indices] = True
        env_reward[truncation_indices] = 0

        env_obs = self.get_observations()
        env_info = [[{}] * self.agent_num for _ in range(self.env_num)]
        
        result = (
            env_obs,
            np.expand_dims(env_reward.repeat(self.agent_num, axis=1), 2),
            env_done.repeat(self.agent_num, axis=1),
            np.array(env_info)
        )
        
        self.steps += 1

        return result

    def get_observations(self):
        '''
        return observations whose dim is (env_num, agent_num, obs_spaces)
        '''
        car = self.car_view
        jetbot_view = self.jetbot_view

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
    
    def set_actions(self, actions):
        actions = actions.reshape(self.env_num, -1)
        self.car_view.set_joint_velocities(
            velocities=actions, 
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
        self.car_view.set_world_poses(positions, orientations, indices)

    def _reset_idx(
            self,
            positions: Optional[Union[np.ndarray, torch.Tensor]] = None,
            orientations: Optional[Union[np.ndarray, torch.Tensor]] = None,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,):
        """
        Reset the articulations to their default states
        """
        default_orientations = self.car_view._default_state.orientations[indices]
        default_joints_positions = self.car_view._default_joints_state.positions[indices]
        default_joints_velocities = self.car_view._default_joints_state.velocities[indices]
        default_joints_efforts = self.car_view._default_joints_state.efforts[indices]
        default_joints_positions = self.car_view._default_joints_state.positions[indices]
        default_gains_kps = self.car_view._default_kps[indices]
        default_gains_kds = self.car_view._default_kds[indices]

        self.car_view.set_world_poses(positions, default_orientations, indices)
        self.car_view.set_joint_positions(positions=default_joints_positions, indices=indices)
        self.car_view.set_joint_velocities(velocities=default_joints_velocities, indices=indices)
        self.car_view.set_joint_efforts(efforts=default_joints_efforts, indices=indices)
        self.car_view.set_gains(kps=default_gains_kps, kds=default_gains_kds, indices=indices)