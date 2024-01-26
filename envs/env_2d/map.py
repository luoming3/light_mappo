"""
Env 2D
"""

from shapely.geometry import Polygon, Point
from shapely import intersects

import random


class Map:
    def __init__(self):
        self.x_range = 51  # size of background
        self.y_range = 51
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.danger_dist = 5
        self.obs, self.risky_filed = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        obs = set()
        risky_filed = set()

        for i in range(x):
            point = (i, 0)
            obs.add(point)  # bottom boundary
            risky_filed.update(self.get_risky_point(point))
        for i in range(x):
            point = (i, y - 1)
            obs.add(point)  # top boundary
            risky_filed.update(self.get_risky_point(point))
        for i in range(y):
            point = (0, i)
            obs.add(point)  # left boundary
            risky_filed.update(self.get_risky_point(point))
        for i in range(y):
            point = (x - 1, i)
            obs.add(point)  # right boundary
            risky_filed.update(self.get_risky_point(point))

        # obstacle_1
        # for i in range(10, 21):
        #     obs.add((i, 15))
        # for i in range(self.y_range // 2):
        #     point = (self.x_range // 3, i)
        #     obs.add(point)
        #     risky_filed.update(self.get_risky_point(point))
        # # obstacle_2
        # for i in range(self.y_range // 2, self.y_range):
        #     point = (self.x_range // 3 * 2, i)
        #     obs.add(point)
        #     risky_filed.update(self.get_risky_point(point))
        # obstacle_3
        # for i in range(16):
        #     obs.add((40, i))
        risky_filed.symmetric_difference(obs)
        
        return obs, risky_filed

    def move(self, point, motion):
        return (point[0] + motion[0], point[1] + motion[1])
    
    def is_collision(self, polygon:Polygon):
        for point in self.obs:
            if intersects(polygon, Point(*point)):
                return True
        return False

    def random_point(self):
        while True:
            point = (random.randint(2, self.x_range-2), random.randint(2, self.y_range-2))
            if point not in self.obs:
                return point
    
    def get_risky_point(self, point):
        danger_zone = set()
        for i in range(self.danger_dist * 2 + 1):
            x0 = point[0] + self.danger_dist - i
            for j in range(self.danger_dist * 2 + 1):
                x1 = point[1] + self.danger_dist - j
                danger_zone.add((x0, x1))
        return danger_zone