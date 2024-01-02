import numpy as np
from shapely.geometry import Polygon, Point
from shapely import intersects, within
import random

L = 100
W = 100
field = Polygon([(0, 0), (0, L), (W, L), (W, 0)])


class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        self.agent_num = 4  # number of agent
        self.obs_dim = 8  # observation dimension of agents
        self.action_dim = 2  # set the action dimension of agents

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """

        # 随机的 agent 位置
        x = random.random() * (W / 2)
        y = random.random() * (L / 2)

        self.wheels = {
            0: np.array((x, y)),
            1: np.array((x + 2, y)),
            2: np.array((x + 2, y + 4)),
            3: np.array((x, y + 4))
        }

        # 初始速度
        self.speed = np.array((0, 0))

        # 目标位置
        self.dest = np.array((random.random() * W, random.random() * L))

        # 智能体观测集合
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.reshape(
                np.array([
                    self.wheels[i], self.dest, self.speed,
                    self.dest - self.wheels[i]
                ]), self.obs_dim)
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        # convert the actions to speed
        action = np.sum(np.reshape(actions, (4, 2)), axis=0) / 10
        self.speed = np.clip((self.speed + action), -1, 1)
        for i in range(self.agent_num):
            self.wheels[i] = self.wheels[i] + self.speed

        # observations after actions
        sub_agent_obs = [
            np.reshape(
                np.array([
                    self.wheels[i], self.dest, self.speed,
                    self.dest - self.wheels[i]
                ]), self.obs_dim) for i in range(self.agent_num)
        ]

        # information of each agent
        sub_agent_info = [{} for _ in range(self.agent_num)]

        sub_agent_reward = []
        sub_agent_done = []
        car = Polygon(self.wheels.values())

        # Check termination conditions
        if intersects(car, Point(self.dest[0], self.dest[1])):
            sub_agent_done = [True for _ in range(self.agent_num)]
            sub_agent_reward = [[np.array(1000)] for _ in range(self.agent_num)]
            self.agents = []
        elif not within(car, field):
            sub_agent_done = [True for _ in range(self.agent_num)]
            sub_agent_reward = [[np.array(-100)] for _ in range(self.agent_num)]
            self.agents = []
        else:
            sub_agent_done = [False for _ in range(self.agent_num)]
            sub_agent_reward = self.get_reward()

        return [
            sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info
        ]

    def get_reward(self):
        car_location = np.mean(list(self.wheels.values()), axis=0)
        dist = np.linalg.norm(car_location - self.dest)
        sub_agent_reward = [[np.array(dist * -0.01)] for _ in range(self.agent_num)]

        return sub_agent_reward

if __name__ == "__main__":
    env = EnvCore()
    # print(env.reset())

    # test the validation of env
    episode = 10
    for _ in range(episode):
        env.reset()
        step = 0
        for _ in range(1000):
            actions = np.random.random(size=(8, )) * 2 - 1
            result = env.step(actions=actions)
            step += 1

            sub_agent_obs, done = result[1], result[2]
            if np.all(done):
                print(sub_agent_obs)
                break
        print(step)
