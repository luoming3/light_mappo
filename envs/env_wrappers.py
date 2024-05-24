"""
# @Time    : 2021/7/1 8:44 上午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_wrappers.py
Modified from OpenAI Baselines code to work with multi-agent envs
"""

import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        pass

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError


# single env
class DummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.actions = None
        self.envs = [fn() for fn in env_fns]
        num_envs = len(env_fns)
        env = self.envs[0]
        ShareVecEnv.__init__(self, num_envs, env.observation_space, env.share_observation_space, env.action_space)

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.any(done):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs] # [env_num, agent_num, obs_dim]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError


class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.parent_pipes, self.child_pipes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(parent_pipe, child_pipe, CloudpickleWrapper(env_fn)))
                   for (parent_pipe, child_pipe, env_fn) in zip(self.parent_pipes, self.child_pipes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for pipe in self.child_pipes:
            pipe.close()

        self.parent_pipes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.parent_pipes[0].recv()
        ShareVecEnv.__init__(self, nenvs, observation_space, share_observation_space, action_space)

    def step_async(self, actions):
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        for (i, done) in enumerate(dones):
            if np.any(done):
                self.parent_pipes[i].send(('reset', None))
                self.parent_pipes[i].recv()

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        obs = [pipe.recv() for pipe in self.parent_pipes]
        return np.stack(obs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for pipe in self.parent_pipes:
                pipe.recv()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        for pipe in self.parent_pipes:
            pipe.send(('render', mode))
        if mode == "rgb_array":   
            frame = [pipe.recv() for pipe in self.parent_pipes]
            return np.stack(frame) 


def worker(parent_pipe, child_pipe, env_fn_wrapper):
    '''
    worker for multiprocessing
    '''
    parent_pipe.close()

    # get env from CloudpickleWrapper
    env = env_fn_wrapper.x()

    while True:
        cmd, data = child_pipe.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            child_pipe.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            child_pipe.send((ob))
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                child_pipe.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            child_pipe.send(ob)
        elif cmd == 'close':
            env.close()
            child_pipe.close()
            break
        elif cmd == 'get_spaces':
            child_pipe.send((env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError

class IsaacSimEnv(ShareVecEnv):
    def __init__(self, env_fns, num_envs):
        self.env = env_fns()
        self.num_envs = num_envs
        super().__init__(num_envs,
                         self.env.observation_space,
                         self.env.share_observation_space,
                         self.env.action_space)

    def step(self, actions):
        return self.step_wait(actions)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self, actions):
        obs, rews, dones, infos = self.env.step(actions)

        reset_indices = []
        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    reset_indices.append(i)
            else:
                if np.any(done):
                    reset_indices.append(i)
        self.env.reset(reset_indices)

        return obs, rews, dones, infos

    def reset(self):
        obs = self.env.reset([])
        return np.array(obs)

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            return np.array(self.env.render(mode=mode))
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

    def close(self):
        self.env.close()
