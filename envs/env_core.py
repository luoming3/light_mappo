import numpy as np
from shapely.geometry import Polygon, Point
from shapely import intersects, within
import random

import matplotlib.pyplot as plt
from matplotlib import patches
import io
import imageio


L = 100
W = 100
field = Polygon([(0, 0), (0, L), (W, L), (W, 0)])


class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        self.agent_num = 4  # number of agent
        self.obs_dim = 10  # observation dimension of agents
        self.action_dim = 2  # set the action dimension of agents
        self.guide_point_num = 100  # number of guide point

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
            3: np.array((x, y + 4)),
        }

        self.car_center = (self.wheels[0] + self.wheels[2]) / 2

        # 初始速度
        self.speed = np.array((0, 0))

        # 目标位置
        self.dest = np.array((random.random() * W, random.random() * L))

        # guide point
        self.guide_points = self.get_guide_point()
        nearest_point = self.next_guide_point()

        # 智能体观测集合
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.reshape(
                np.array(
                    [self.wheels[i], self.dest, self.speed, self.dest - self.wheels[i], nearest_point]
                ),
                self.obs_dim,
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
        # convert the actions to speed
        action = np.sum(np.reshape(actions, (4, 2)), axis=0) / 10
        self.speed = np.clip((self.speed + action), -1, 1)
        for i in range(self.agent_num):
            self.wheels[i] = self.wheels[i] + self.speed
        
        # center point of car
        self.car_center = (self.wheels[0] + self.wheels[2]) / 2

        # get next guide point
        next_guide_point = self.next_guide_point()

        # observations after actions
        sub_agent_obs = [
            np.reshape(
                np.array(
                    [self.wheels[i], self.dest, self.speed, self.dest - self.wheels[i], next_guide_point]
                ),
                self.obs_dim,
            )
            for i in range(self.agent_num)
        ]

        # information of each agent
        sub_agent_info = [{} for _ in range(self.agent_num)]

        sub_agent_reward = []
        sub_agent_done = []
        car = Polygon(self.wheels.values())

        # Check termination conditions
        if intersects(car, Point(self.dest[0], self.dest[1])):
            sub_agent_done = [True for _ in range(self.agent_num)]
            sub_agent_reward = [[np.array(100)] for _ in range(self.agent_num)]
            self.agents = []
        elif not within(car, field):
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
        car_location = np.mean(list(self.wheels.values()), axis=0)
        dist = np.linalg.norm(car_location - self.dest)
        sub_agent_reward = [[np.array(dist * -0.01)] for _ in range(self.agent_num)]

        return sub_agent_reward

    def render(self, mode="rgb_array"):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlim(-5, 105)
        ax.set_ylim(-1, 101)

        # 方框大小
        rect = patches.Rectangle(
            (0, 0), W, L, linewidth=1, edgecolor="r", facecolor="none"
        )
        # 目标
        cir = patches.Circle(self.dest, 1)

        # guide point
        if len(self.guide_points) > 0:
            guide_x, guide_y = zip(*self.guide_points)
            ax.scatter(guide_x, guide_y, color="y", s=10)

        ax.add_patch(rect)
        ax.add_patch(cir)

        # 绘制小车的位置
        patch = patches.Polygon(list(self.wheels.values()), closed=True, fc="r", ec="r")
        ax.add_patch(patch)

        # 保存在内存中
        # 创建一个内存缓冲区
        buffer = io.BytesIO()

        # 将图像保存到内存中
        plt.savefig(buffer, format="png")
        plt.close(fig)  # 关闭图像以释放内存

        # 重置缓冲区的指针到开始位置
        buffer.seek(0)

        # 使用PIL从内存中读取图像
        image = imageio.v2.imread(buffer)

        # 关闭缓冲区
        buffer.close()

        return image
    
    def get_guide_point(self):
        x1, y1 = self.car_center
        x2, y2 = self.dest

        # 生成插值点，并去除起点
        x_values = np.linspace(x1, x2, self.guide_point_num)[1:]
        y_values = np.linspace(y1, y2, self.guide_point_num)[1:]

        # return guide points
        return list(zip(x_values, y_values))

    def get_score(self, car):
        for i, point in enumerate(self.guide_points):
            if intersects(car, Point(*point)):
                self.guide_points = self.guide_points[i+1:]
                return True

        return False
    
    def next_guide_point(self):
        while len(self.guide_points) > 0:
            point = self.guide_points[0]
            car_distance = np.linalg.norm(self.car_center - self.dest)
            point_distance = np.linalg.norm(point - self.dest)

            if point_distance < car_distance:
                return point
            else:
                self.guide_points = np.delete(self.guide_points, 0, axis=0)

        return self.dest


if __name__ == "__main__":
    env = EnvCore()
    # print(env.reset())

    # test the validation of env
    episode = 1
    all_frames = []
    for _ in range(episode):
        env.reset()
        image = env.render()
        all_frames.append(image)
        step = 0
        for _ in range(1000):
            # actions = np.random.random(size=(8,)) * 2 - 1
            actions = np.expand_dims(env.dest - env.car_center, 0).repeat(env.agent_num, 0) / 10
            result = env.step(actions=actions)
            all_frames.append(env.render())
            step += 1

            sub_agent_obs, done = result[1], result[2]
            if np.all(done):
                break
        print(step)

    import os

    gif_save_path = os.path.dirname(__file__) + "/" + "render.gif"
    imageio.mimsave(gif_save_path, all_frames, duration=1, loop=0)
