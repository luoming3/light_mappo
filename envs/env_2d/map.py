"""
Env 2D
"""

from shapely.geometry import Polygon, Point
from shapely import intersects


class Map:
    def __init__(self):
        self.x_range = 51  # size of background
        self.y_range = 31
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = self.obs_map()

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

        for i in range(x):
            obs.add((i, 0))  # bottom boundary
        for i in range(x):
            obs.add((i, y - 1))  # top boundary
        for i in range(y):
            obs.add((0, i))  # left boundary
        for i in range(y):
            obs.add((x - 1, i))  # right boundary

        # obstacle_1
        for i in range(10, 21):
            obs.add((i, 15))
        for i in range(15):
            obs.add((20, i))
        # obstacle_2
        for i in range(15, 30):
            obs.add((30, i))
        # obstacle_3
        for i in range(16):
            obs.add((40, i))

        return obs

    def move(self, point, motion):
        return (point[0] + motion[0], point[1] + motion[1])
    
    def is_collision(self, polygon:Polygon):
        for point in self.obs:
            if intersects(polygon, Point(*point)):
                return True
        return False
