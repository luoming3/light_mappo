import os
import sys
import numpy as np
import time

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from light_mappo.agent import Agent

if __name__ == "__main__":
    agent = Agent()
    obs = np.array([[0, 0, 0, 0, 0, 0, 0]])

    for i in range(10):
        time_s = time.perf_counter()
        actions = agent.act(obs)
        print(f"act success, execute time: {time.perf_counter() - time_s}")
