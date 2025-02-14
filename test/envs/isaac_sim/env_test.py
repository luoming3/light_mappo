import unittest
import numpy as np
import torch

import os
import sys

# Get the parent directory of the current file
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), "git/light_mappo"))
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from light_mappo.envs.isaac_sim import init_simulation_app
from light_mappo.config import get_config

env_num = 1
maxbot_num = 4

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='simple', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)

    all_args = parser.parse_known_args(args)[0]

    return all_args

class EnvCoreCase(unittest.TestCase):
    def setUp(self):
        headless = False
        self.simulation_app = init_simulation_app(headless)

    def tearDown(self):
        self.simulation_app.close()

    def test_setup_scene(self, args):
        print('args', args)
        from light_mappo.envs.isaac_sim.utils import scene
        parser = get_config()
        all_args = parse_args(args, parser)
        device = torch.device("cuda:0")
        # self.world = scene.set_up_new_scene(env_num, maxbot_num)
        self.world = scene.set_up_new_scene(config={"all_args": all_args, "device": device,})

    def test_action(self):
        car_view = self.world.scene.get_object("car_view")
        jetbot_view = self.world.scene.get_object("jetbot_chassis_view")
        for i in range(100):
            self.world.step()

        # for i in range(10):
        #     car_view.set_joint_velocities(
        #         velocities=torch.tensor(
        #             [
        #                 [1., -1., -1., 1., 1., -1., -1., 1.],
        #             ]
        #         ) * 5,
        #         joint_indices=torch.arange(4,12)  # revoluted joint indices
        #     )
        #     self.world.step(render=True) # execute one physics step and one rendering step
        #     joint_forces = car_view.get_measured_joint_forces()[:,1:5,:2]
        #     print(joint_forces[0])
        #     car_velocity = car_view.get_linear_velocities()
        #     print(f"linear_velocity: {car_velocity}")
        #     print("========\n")

        for i in range(5000):
            car_view.set_joint_velocities(
                velocities=torch.tensor(
                    [
                        [1., 1., 1., 1., 1., 1., 1., 1.],
                    ]
                ) * 5,
                joint_indices=torch.arange(4,12)  # revoluted joint indices
            ) 
            self.world.step(render=True) # execute one physics step and one rendering step
            joint_forces = car_view.get_measured_joint_forces()[:,1:5,:2]
            print(joint_forces[0])
            car_linear_velocity = car_view.get_linear_velocities()
            jetbot_angular_velocity = jetbot_view.get_angular_velocities()
            print(f"linear_velocity: {car_linear_velocity}")
            print(f"angular_velocity: {jetbot_angular_velocity}")
            print("========\n")

    def test_step(self):
        from light_mappo.envs.isaac_sim.env_core import EnvCore
        
        parser = get_config()
        all_args = parser.parse_known_args()[0]
        all_args.isaac_sim_headless = False
        env = EnvCore(all_args, env_num)
        for i in range(3):
            env.reset()
            for i in range(1024):
                actions = np.array(
                    [
                        [1., 1., 1., 1., 1., 1., 1., 1.],
                    ]
                , dtype=np.float32) * 10
                actions = actions.reshape(env_num, maxbot_num, 2)
                obs, reward, done, info = env.step(actions=actions)

if __name__ == "__main__":
    # unittest.main()
    env_core = EnvCoreCase()
    # setup
    env_core.setUp()
    # test set up scene
    env_core.test_setup_scene(sys.argv[1:])
    # test step
    env_core.test_action()
    # close
    env_core.tearDown()

