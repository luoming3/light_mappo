import unittest

from envs.isaac_sim import init_simulation_app
import numpy as np

env_num = 2


class EnvCoreCase(unittest.TestCase):
    def setUp(self):
        self.simulation_app = init_simulation_app()

    def tearDown(self):
        self.simulation_app.close()

    def test_setup_scene(self):
        from envs.isaac_sim.utils import scene
        
        scene.set_up_scene(env_num)

    def test_step(self):
        from envs.isaac_sim.env_core import EnvCore
        
        env = EnvCore()
        for i in range(3):
            env.reset()
            for i in range(1024):
                actions = np.full((env_num, 8), 10)
                obs, reward, done, info = env.step(actions=actions)

if __name__ == "__main__":
    # unittest.main()
    env_core = EnvCoreCase()
    # setup
    env_core.setUp()
    # test set up scene
    env_core.test_setup_scene()
    # test step
    env_core.test_step()
    # close
    env_core.tearDown()

