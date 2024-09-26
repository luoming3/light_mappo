"""
# @Time    : 2021/7/1 7:15 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import time
import numpy as np
import torch
from runner.shared.base_runner import Runner

import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        save_episode_interval = episodes // self.num_save_model

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                
                reset_indices = []
                for (i, done) in enumerate(dones):
                    if torch.any(done):
                        reset_indices.append(i)
                        
                if reset_indices:
                    obs = self.env.reset(reset_indices)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # save model at regular intervals for rendering test
            if episode % save_episode_interval == 0:
                self.save_for_test(total_num_steps)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                # if self.env_name == "MPE":
                #     env_infos = {}
                #     for agent_id in range(self.num_agents):
                #         idv_rews = []
                #         for info in infos:
                #             if 'individual_reward' in info[agent_id].keys():
                #                 idv_rews.append(info[agent_id]['individual_reward'])
                #         agent_k = 'agent%i/individual_rewards' % agent_id
                #         env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards)
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                # self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)  # shape = [env_num, agent_num * obs_dim]
            share_obs = np.expand_dims(share_obs, 1).repeat(
                self.num_agents, axis=1
            )  # shape = shape = [env_num, agent_num， agent_num * obs_dim]
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))  # [env_num, agent_num, 1]
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))  # [env_num, agent_num, action_dim]
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )  # [env_num, agent_num, 1]
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            # actions  --> actions_env : shape:[10, 1] --> [5, 2, 5]
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            # TODO 这里改造成自己环境需要的形式即可
            # TODO Here, you can change the shape of actions_env to fit your environment
            # actions_env = actions
            actions_env = actions
            # raise NotImplementedError

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[
                        eval_actions[:, :, i]
                    ]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                eval_actions_env = eval_actions
                # raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

            if np.any(eval_dones):
                print(f"eval episode step: {eval_step+1}")
                break

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos["eval_average_episode_rewards"])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        step_list = []
        bad_case = []
        for i in range(100):
            envs.env.env.world.step(render=False)

        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            all_frames = []
            if self.all_args.env_type != 'isaac_sim':
                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0]
                    all_frames.append(image)
                else:
                    envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_render_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_render_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []
            episode_dones = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_render_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_render_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    actions_env = actions
                    # raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)
                episode_dones.append(dones)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_render_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.env_type != 'isaac_sim':
                    if self.all_args.save_gifs:
                        image = envs.render("rgb_array")[0]  # TODO: support parallel env setting
                        all_frames.append(image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                    else:
                        envs.render("human")
                
                if np.all(dones):
                    all_frames = all_frames[:-1]
                    print(step)
                    break
            
            sub_episode_dones = np.split(np.array(episode_dones), self.n_render_rollout_threads, axis=1)
            sub_episode_rewards = np.split(np.array(episode_rewards), self.n_render_rollout_threads, axis=1)
            for index, value in enumerate(sub_episode_dones):
                indices = np.where(value == True)
                if indices[0].size == 0:
                    step = self.all_args.episode_length - 1
                    bad_case.append(envs.env.env.init_envs_positions[index:index+1])
                    bad_case.append(envs.env.env.car_position[index:index+1])
                    bad_case.append(envs.env.env.orientations[index:index+1])
                else:
                    step = indices[0][0]
                average_episode_rewards = np.mean(np.sum(np.array(sub_episode_rewards[index][:step+1]), axis=0))
                print("index: " + str((episode, index)))
                print("average episode rewards is: " + str(average_episode_rewards))
                print("step: " + str(step))
                step_list.append(step)
            # average_episode_rewards = np.mean(np.sum(np.array(episode_rewards), axis=0))
            # print("average episode rewards is: " + str(average_episode_rewards))
            # print("step: " + str(step))
            # step_list.append(step)


            if self.all_args.save_gifs:
                imageio.mimsave(str(self.gif_dir) + '/' + str(episode) + '_' + '{:.2f}'.format(average_episode_rewards) + '.gif',
                                all_frames,
                                duration=self.all_args.ifi,
                                loop=0)
        suc_rate = len([num for num in step_list if num < (self.all_args.episode_length - 1)]) / len(step_list)
        step_list = np.array(step_list)
        avg_step = sum(step_list[step_list < (self.all_args.episode_length - 1)]) / (len(step_list) * suc_rate)
        print("average step: " + str(avg_step))
        print("success rate: " + str(suc_rate))

        # 保存 tensor 列表到文件
        tensor_file_name = '/'.join(self.all_args.model_dir.split('/')[:-1]) + '/tensors.pth'
        torch.save(bad_case, tensor_file_name)

        # 读取 tensor 列表
        loaded_tensors = torch.load(tensor_file_name)

        # 打印加载的 tensor
        for i, tensor in enumerate(loaded_tensors):
            print(f"Tensor {i + 1}:")
            print(tensor)

        # 指定要追加的文件名
        file_name = '/'.join(self.all_args.model_dir.split('/')[:-2]) + '/render_result.log'

        # 以追加模式打开文件
        with open(file_name, 'a') as file:
            # 将数据写入文件, 每条数据占一行
            file.write(f'Model Directory: {self.all_args.model_dir}, Avg Step: {avg_step}, Success Rate: {suc_rate}\n')

    def render_specific_episode(self):
        """Visualize the env."""
        envs = self.envs
        step_list = []
        render_rollout_threads = 1

        file_name = '/'.join(self.all_args.model_dir.split('/')[:-1]) + '/tensors.pth'
        # 读取 tensor 列表
        loaded_tensors = torch.load(file_name)
        for i in range(100):
            envs.env.env.world.step(render=False)
        # 打印加载的 tensor
        for i in range(0, len(loaded_tensors), 3):
            print(f"Tensor {i}:")
            envs.env.env.init_envs_positions = loaded_tensors[i]
            # envs.env.env.world.reset()
            obs = envs.reset_specific_pos(loaded_tensors[i + 1], loaded_tensors[i + 2])
            all_frames = []
            if self.all_args.env_type != 'isaac_sim':
                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0]
                    all_frames.append(image)
                else:
                    envs.render("human")

            rnn_states = np.zeros(
                (
                    render_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((render_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), render_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), render_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    actions_env = actions
                    # raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((render_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.env_type != 'isaac_sim':
                    if self.all_args.save_gifs:
                        image = envs.render("rgb_array")[0]  # TODO: support parallel env setting
                        all_frames.append(image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                    else:
                        envs.render("human")
                
                if np.any(dones):
                    all_frames = all_frames[:-1]
                    print(step)
                    break
            
            average_episode_rewards = np.mean(np.sum(np.array(episode_rewards), axis=0))
            print("average episode rewards is: " + str(average_episode_rewards))
            print("step: " + str(step))
            step_list.append(step)
