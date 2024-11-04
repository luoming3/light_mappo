import numpy as np
from gym import spaces

import torch
import torch.distributions as D
from typing import Optional, Union, Tuple
import math
import random

import os
import sys

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from light_mappo.utils.util import euler_to_quaternion, quaternion_to_euler

OBS_DIM = 7
ACTION_SPACE = spaces.Box(
    np.array([-10, -10]).astype(np.float32),
    np.array([+10, +10]).astype(np.float32),
)  # left_wheel velocity and right_wheel velocity


class EnvCore(object):
    def __init__(self, all_args, env_num, device=None) -> None:
        # isaac sim environment
        self.all_args = all_args
        from light_mappo.envs.isaac_sim.utils.scene import get_world
        self.world = get_world(all_args.dt)
        self.env_num = env_num  # TODO: setting in config.py
        self.car_view = self.world.scene.get_object("car_view")
        self.jetbot_view = self.world.scene.get_object("jetbot_chassis_view")
        self.skip_frame=all_args.skip_frame

        self.agent_num = all_args.num_agents  # number of agent
        self.obs_dim = OBS_DIM  # observation dimension of agents
        self.action_dim = 3  # set the action dimension of agents
        self.env_indices = [i for i in range(self.env_num)]
        self.action_space = ACTION_SPACE
        if device is None:
            self.device = "cuda:0"
        else:
            self.device = device
        self.steps = torch.zeros(size=(self.env_num,), dtype=int, device=self.device)
        self.truncation_step = all_args.truncation_step

        self.n_car_position = torch.zeros((self.env_num, 3), device=self.device)
        self.n_orientations = torch.zeros((self.env_num, 4), device=self.device)
        self.target_pos = torch.zeros((self.env_num, 2), device=self.device)
        self.init_envs_positions = self.get_world_poses()[0]

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2, -2, 0], dtype=torch.float32, device=self.device),
            torch.tensor([2, 2, 0.0001], dtype=torch.float32, device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([0., 0., -1.], dtype=torch.float32, device=self.device) * torch.pi,
            torch.tensor([1e-8, 1e-8, 1.], dtype=torch.float32, device=self.device) * torch.pi
        )

    def reset(self, indices=[]):
        if len(indices) == 0:
            indices = self.env_indices

        car_position = self.get_random_positions(indices)
        orientations = self.get_random_orientation(indices)

        # reset cars' position and velocity
        self._reset_idx(positions=car_position, orientations=orientations, indices=indices)

        self.n_car_position[indices] = car_position
        self.n_orientations[indices] = orientations

        # observations, shape is (env_num, agent_num, obs_dim)
        observations = self.get_observations()

        self.steps[indices] = 0

        return observations
    
    def reset_specific_pos(self, car_position: Optional[Union[np.ndarray, torch.Tensor]] = None, orientations: Optional[Union[np.ndarray, torch.Tensor]] = None, indices=[]):
        if len(indices) == 0:
            indices = self.env_indices

        self.car_position = car_position

        # reset cars' position and velocity
        self._reset_idx(positions=self.car_position, orientations=orientations, indices=indices)

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
        previous_car_position= self.get_world_poses()[0][:, 0:2]
        previous_car_position.sub_(self.init_envs_positions[:, 0:2])

        # set actions
        for i in range(self.skip_frame):
            self.set_actions(actions)
            self.world.step(not self.all_args.isaac_sim_headless)

        env_obs = self.get_observations()
        current_car_position = self.get_world_poses()[0][:, 0:2]
        current_car_position.sub_(self.init_envs_positions[:, 0:2])
        goal_world_position = self.target_pos

        previous_dist_to_goal = torch.norm(goal_world_position - previous_car_position, p=2, dim=1)
        current_dist_to_goal = torch.norm(goal_world_position - current_car_position, p=2, dim=1)

        # running
        dist_reward = previous_dist_to_goal - current_dist_to_goal
        direction_reward = torch.cosine_similarity(self.car_linear_velocities, self.rpos_car_dest, dim=1)
        velocities_reward = 5 * torch.where(direction_reward > 0.95, 1., -1.) * torch.norm(self.car_linear_velocities, dim=1)
        direction_reward = torch.where(direction_reward > 0.95, direction_reward, 0)
        step_reward = -2
        total_force = torch.norm(self.y_joint_force, p=2, dim=1) + torch.norm(self.x_joint_force, p=2, dim=1)
        joint_pen = torch.log(total_force)/torch.log(torch.ones((1,), device=self.device)*100)
        joint_pen = torch.where(joint_pen > 1.4, joint_pen - 1, 0)
        env_reward = (dist_reward + direction_reward + velocities_reward + step_reward - joint_pen).reshape((self.env_num, 1))
        env_done = torch.zeros((self.env_num, 1), dtype=bool, device=self.device)
        
        # arrival
        arrival_indices = torch.where(current_dist_to_goal < 0.2)
        env_done[arrival_indices] = True
        env_reward[arrival_indices] = 100.

        # failure
        failure_indices = torch.where(current_dist_to_goal > 5)
        env_done[failure_indices] = True
        # env_reward[failure_indices] = 0.

        # truncation
        truncation_indices = torch.where(self.steps==self.truncation_step)
        env_done[truncation_indices] = True
        # env_reward[truncation_indices] = 0.

        if self.all_args.use_random_point:
            env_done[arrival_indices] = False
            self.steps[arrival_indices] = 0
            angle = 2 * math.pi * random.random()
            next_point = torch.tensor([math.cos(angle), math.sin(angle)], device=self.device) + self.positions[arrival_indices]
            next_point = torch.where(next_point > 4, 8 - next_point, next_point)
            next_point = torch.where(next_point < -4, -8 - next_point, next_point)
            self.target_pos[arrival_indices] = next_point
            self.target_pos[truncation_indices] = torch.zeros((1, 2), device=self.device)

        env_info = [[{}] * self.agent_num for _ in range(self.env_num)]
        
        result = (
            env_obs,
            torch.unsqueeze(env_reward.repeat(1, self.agent_num), 2),
            env_done.repeat(1, self.agent_num),
            env_info
        )
        
        self.steps += 1

        return result

    def get_observations(self):
        '''
        return observations whose dim is (env_num, agent_num, obs_spaces)
        '''
        car = self.car_view
        jetbot_view = self.jetbot_view

        self.positions, _ = self.get_world_poses()
        self.positions.sub_(self.init_envs_positions)
        self.positions = self.positions[:, 0:2]
        # only need x,y axis
        self.rpos_car_dest = self.target_pos - self.positions
        rpos_car_dest_norm = normalized(self.rpos_car_dest)
        rpos_car_dest_norm = rpos_car_dest_norm.unsqueeze(1).repeat(1, self.agent_num, 1)

        self.car_linear_velocities = car.get_linear_velocities()[:, 0:2]
        car_linear_velocities = self.car_linear_velocities.unsqueeze(1).repeat(1, self.agent_num, 1)

        # only need x,y axis
        jetbot_linear_velocities = jetbot_view.get_linear_velocities()[:, 0:2]
        jetbot_linear_velocities = jetbot_linear_velocities.reshape(self.env_num, self.agent_num, 2)
        # only need z axis
        # jetbot_angular_velocities = jetbot_view.get_angular_velocities()[:, 2]
        # jetbot_angular_velocities = jetbot_angular_velocities.reshape(self.env_num, self.agent_num, 1)

        jetbot_position, jetbot_orientation = jetbot_view.get_world_poses()
        jetbot_orientation = quaternion_to_euler(jetbot_orientation)
        # only need x,y axis
        # jetbot_position = jetbot_position[:, 0:2]
        # jetbot_position = jetbot_position.reshape(self.env_num, self.agent_num, 2)
        # jetbot_position.sub_(self.init_envs_positions[:, 0:2].unsqueeze(1))
        # rpos_car_jetbot_norm = normalized(jetbot_position - positions[:, 0:2].unsqueeze(1), dim=2)

        # only need w and z axis
        jetbot_orientation = jetbot_orientation[:, 2]
        jetbot_orientation = jetbot_orientation.reshape(self.env_num, self.agent_num, 1)

        joint_forces = self.car_view.get_measured_joint_forces()[:,1:5,:2]
        self.y_joint_force = joint_forces[:,:,1]
        self.x_joint_force = joint_forces[:,:,0]
        self.joint_forces = joint_forces
        joint_forces = torch.tanh(joint_forces/50)

        observations = torch.cat(
            (
                rpos_car_dest_norm,
                jetbot_linear_velocities,
                jetbot_orientation,
                joint_forces
            ),
            dim=2
        )

        return observations
    
    def set_actions(self, actions):
        actions = np.tanh(actions) * 5
        actions = torch.from_numpy(actions)
        actions = actions.reshape(self.env_num, -1)
        revolution_joint_indices = torch.arange(self.agent_num, self.agent_num * 3)
        self.car_view.set_joint_velocities(
            velocities=actions, 
            joint_indices=revolution_joint_indices  # revoluted joint indices
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
        if orientations is None:
            orientations = self.car_view._default_state.orientations[indices]
        default_joints_positions = self.car_view._default_joints_state.positions[indices]
        default_joints_velocities = self.car_view._default_joints_state.velocities[indices]
        default_joints_efforts = self.car_view._default_joints_state.efforts[indices]
        default_gains_kps = self.car_view._default_kps[indices]
        default_gains_kds = self.car_view._default_kds[indices]

        self.car_view.set_world_poses(positions, orientations, indices)
        self.car_view.set_joint_positions(positions=default_joints_positions, indices=indices)
        self.car_view.set_joint_velocities(velocities=default_joints_velocities, indices=indices)
        self.car_view.set_joint_efforts(efforts=default_joints_efforts, indices=indices)
        self.car_view.set_gains(kps=default_gains_kps, kds=default_gains_kds, indices=indices)
        self.car_view.set_velocities(torch.zeros(6, device=self.device), indices=indices)
        if not self.all_args.use_render:
            self.world.step()

    
    def get_random_positions(self, indices):
        # pos = self.init_pos_dist.sample((len(indices),))
        # pos += self.init_envs_positions[indices]
        radius = 2
        theta = 2 * torch.pi * torch.rand(len(indices), device=self.device)
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        z = torch.zeros(size=(len(indices),), device=self.device)
        pos = torch.stack((x, y, z), dim=1)
        pos += self.init_envs_positions[indices]
        return pos
    
    def get_world_poses(self, clone: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.car_view.get_world_poses(clone=clone)

    def get_random_orientation(self, indices):
        random_rpy = self.init_rpy_dist.sample((len(indices),))
        random_ori = euler_to_quaternion(random_rpy)
        return random_ori

def normalized(v: torch.tensor, dim=1):
    normalized = v / (torch.norm(v, dim=dim, keepdim=True) + 0.0000001)
    return normalized
