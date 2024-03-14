from pettingzoo.classic import rps_v2

# env = rps_v2.env(render_mode="human")
# env.reset(seed=42)

# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()

#     if termination or truncation:
#         action = None
#     else:
#         action = env.action_space(agent).sample() # this is where you would insert your policy

#     env.step(action)
# env.close()


if __name__ == "__main__":
    import time

    step_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    }
    t1 = time.perf_counter()
    for i in range(10000000):
        a = step_map.get(i, -1)
    print(time.perf_counter() - t1)
