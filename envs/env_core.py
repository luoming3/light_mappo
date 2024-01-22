import numpy as np
from shapely.geometry import Polygon, Point
from shapely import intersects, within
import random

import imageio

import os
import sys

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from envs.env_2d import map, plotting, Astar  # noqa: E402
from envs.env_2d.car_racing import CarRacing


class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        self.agent_num = 4  # number of agent
        self.obs_dim = 10  # observation dimension of agents
        self.action_dim = 3  # set the action dimension of agents
        self.guide_point_num = 100  # number of guide point
        self.map = map.Map()  # 2d env map
        self.width = self.map.x_range
        self.height = self.map.y_range
        self.car_env = CarRacing()

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """

        # 随机的 agent 位置
        self.car_center = np.array(self.map.random_point()).astype(float)
        # 目标位置
        self.dest = np.array(self.map.random_point())
        # reset car env
        self.car_env.reset(car_pos=self.car_center)

        # guide point
        self.guide_points = self.get_guide_point()
        nearest_point = self.next_guide_point()

        # 智能体观测集合
        sub_agent_obs = []
        for i in range(self.agent_num):
            w = self.car_env.car.wheels[i]

            sub_obs = np.reshape(
                [
                    np.array([w.position.x, w.position.y]),
                    self.dest,
                    np.array([w.omega, w.phase]),
                    self.dest - np.array([w.position.x, w.position.y]),
                    nearest_point
                ], self.obs_dim
            )

            sub_agent_obs.append(sub_obs)

        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        self.car_env.step(actions)
        self.car_center = self.car_env.car_pos

        # get next guide point
        next_guide_point = self.next_guide_point()

        # observations after actions
        sub_agent_obs = []
        for i in range(self.agent_num):
            w = self.car_env.car.wheels[i]

            sub_obs = np.reshape(
                [
                    np.array([w.position.x, w.position.y]),
                    self.dest,
                    np.array([w.omega, w.phase]),
                    self.dest - np.array([w.position.x, w.position.y]),
                    next_guide_point
                ], self.obs_dim
            )

            sub_agent_obs.append(sub_obs)

        # information of each agent
        sub_agent_info = [{} for _ in range(self.agent_num)]

        sub_agent_reward = []
        sub_agent_done = []
        wheels_pos = ((w.position.x, w.position.y) for w in self.car_env.car.wheels)
        car = Polygon(wheels_pos)

        # Check termination conditions
        if intersects(car, Point(self.dest[0], self.dest[1])):
            sub_agent_done = [True for _ in range(self.agent_num)]
            sub_agent_reward = [[np.array(1000)] for _ in range(self.agent_num)]
            self.agents = []
        elif self.map.is_collision(car):
            sub_agent_done = [True for _ in range(self.agent_num)]
            sub_agent_reward = [[np.array(-100)] for _ in range(self.agent_num)]
            self.agents = []
        elif self.get_score(car):
            sub_agent_done = [False for _ in range(self.agent_num)]
            sub_agent_reward = [[np.array(10)] for _ in range(self.agent_num)]
        else:
            sub_agent_done = [False for _ in range(self.agent_num)]
            sub_agent_reward = self.get_reward()

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def get_reward(self):
        dist = np.linalg.norm(self.car_center - self.dest)
        sub_agent_reward = [[np.array(-1)] for _ in range(self.agent_num)]

        return sub_agent_reward

    def render(self, mode="rgb_array"):
        if mode == 'rgb_array':
            plot = plotting.Plotting(target=self.dest)
            plot.plot_map()
            wheels = [(w.position.x, w.position.y) for w in self.car_env.car.wheels]
            plot.plot_car(wheels)
            plot.plot_guide_point(self.guide_points)
            image = plot.save_image()

            return image

    def get_guide_point(self):
        start = tuple(self.car_center.astype(int).tolist())
        end = tuple(self.dest.astype(int).tolist())
        astar = Astar.AStar(start, end, "euclidean")
        path, _ = astar.searching()
 
        # return guide points
        path.reverse()
        return np.array(path[1:-1])

    def get_score(self, car):
        for i, point in enumerate(self.guide_points):
            if intersects(car, Point(*point)):
                self.guide_points = self.guide_points[i+1:]
                return True

        return False

    def next_guide_point(self):
        while len(self.guide_points) > 1:
            first_point = self.guide_points[0]
            second_point = self.guide_points[1]
            angle = (second_point - first_point).dot(self.car_center - first_point)

            # 夹角是钝角
            if angle < 0:
                return first_point
            else:
                self.guide_points = self.guide_points[1:]
        return self.dest


def test_env(times=10, render=False, mode='rgb_array'):
    '''
    test the validation of env
    '''
    env = EnvCore()

    for i in range(times):
        env.reset()

        all_frames = []
        if render:
            image = env.render(mode=mode)
            all_frames.append(image)

        step = 0
        for _ in range(1000):
            # actions = np.random.random(size=(env.agent_num,)) * 2 - 1
            # actions = np.expand_dims(env.dest - env.car_center, 0).repeat(env.agent_num, 0) / 10
            action_space = env.car_env.action_space
            actions = np.array([action_space.sample() for i in range(env.agent_num)])
            result = env.step(actions=actions)
            if render:
                all_frames.append(env.render()) 
            step += 1

            sub_agent_obs, done = result[1], result[2]
            if np.all(done):
                break

        if render and mode == 'rgb_array':
            import os

            image_dir = os.path.dirname(__file__) + "/" + "image"
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            gif_save_path = image_dir + f"/{i}_{step}.gif"
            imageio.mimsave(gif_save_path, all_frames, duration=1, loop=0)


if __name__ == "__main__":
    test_env(times=10, render=True)
