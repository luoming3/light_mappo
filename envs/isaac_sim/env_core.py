import numpy as np
from gym import spaces

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

import os
import sys

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from envs.env_2d import map, plotting, Astar  # noqa: E402
from envs.env_core import EnvCore as Env2d
from envs.isaac_sim.utils.scene import get_world, set_up_scene


class EnvCore(Env2d):
    def __init__(self) -> None:
        # isaac sim environment
        self.world = get_world()
        self.env_num = 2  # TODO: setting in config.py
        self.scene = self.world.scene

        self.agent_num = 4  # number of agent
        self.obs_dim = 12  # observation dimension of agents
        self.action_dim = 3  # set the action dimension of agents
        self.guide_point_num = 100  # number of guide point
        self.map = map.Map()  # 2d env map
        self.width = self.map.x_range
        self.height = self.map.y_range
        self.env_indices = [i for i in range(self.env_num)]
        self.action_space = spaces.Box(
            np.array([-10, -10]).astype(np.float32),
            np.array([+10, +10]).astype(np.float32),
        )  # left_wheel velocity and right_wheel velocity

    # TODO
    def reset(self, indeics=[]):
        # 智能体观测集合
        observations = self.get_observations()

        result = []
        for i in range(self.env_num):
            sub_agent_obs = []
            for i in range(self.agent_num):
                sub_obs = np.random.random(size=(self.obs_dim, ))
                sub_agent_obs.append(sub_obs)
            result.append(sub_agent_obs)
        return result

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

    # TODO
    def get_observations(self):
        '''
        return observations whose dim is (env_num, agent_num, obs_spaces)
        '''
        car = self.scene.get_object("car_view")
        jetbot_view = self.scene.get_object("jetbot_chassis_view")

        observations = []
        positions, orientations = car.get_world_poses()
        # only need x,y axis
        car_positon = positions[:, 0:2]
        # only need x,y axis
        linear_velocities = jetbot_view.get_linear_velocities()[:, 0:2]
        # only need z axis
        angular_velocities = jetbot_view.get_angular_velocities()[:, 2]
        jetbot_positions, jetbot_orientations = jetbot_view.get_world_poses()
        jetbot_positions = jetbot_positions[:, 0:2]
        jetbot_orientations = jetbot_orientations[:, 3]

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
