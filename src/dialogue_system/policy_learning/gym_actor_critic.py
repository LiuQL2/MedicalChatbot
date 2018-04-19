# -*- coding: utf-8 -*-

import gym
import sys, os
import numpy as np
sys.path.append(os.getcwd().replace("src/dialogue_system/policy_learning",""))

from src.dialogue_system.policy_learning.actor_critic_4 import ActorCritic
from collections import deque
import random

experience_replay_pool = deque(maxlen=5000)



def policy_rollout(env, agent, experience_replay_pool):
    """Run one episode."""
    # state, agent_action, reward, next_state, episode_over
    trajectory, rwds = [], []
    observation, reward, done = env.reset(), 0, False

    while not done:

        env.render()
        action = agent.take_action(observation)
        next_observation, reward, done, _ = env.step(action)
        trajectory.append((observation,action,reward,next_observation,done))
        experience_replay_pool.append((observation,action,reward,next_observation,done))
        rwds.append(reward)
        observation = next_observation

    return trajectory, rwds, experience_replay_pool


def policy_rollout2(env, agent):
    """Run one episode."""

    observation, reward, done = env.reset(), 0, False
    obs, acts, rews, dones = [], [], [], []

    while not done:

        env.render()
        obs.append(observation)

        action = agent.take_action(observation)
        observation, reward, done, _ = env.step(action)
        if int(reward) != 1:
            print("reward:",reward)

        dones.append(done)
        acts.append(action)
        rews.append(reward)

    return obs, acts, rews, dones


def process_rewards(rews):
    """Rewards -> Advantages for one episode. """

    # total reward: length of episode
    advantages = [len(rews)] * len(rews)
    # print("advantages:",advantages)
    return advantages


def sample_batch(experience_replay_pool, batch_size, batch_num=1):
    sample = []
    for _ in range(0,batch_num,1):
        batch = random.sample(list(experience_replay_pool), batch_size)
        sample.append(batch)
    return sample

env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('Pendulum-v0')
env.reset()

# hyper parameters
hparams = {
    'input_size': env.observation_space.shape[0],
    'hidden_size': 36,
    'num_actions': env.action_space.n,
    "gamma":0.98,
    "actor_learning_rate":0.001,
    "critic_learning_rate":0.001,
    "keep_prob":0.8
}


# environment params
eparams = {
    'num_batches': 4000,
    'ep_per_batch': 10
}
batch_size = 64
batch_num = 10

agent = ActorCritic(input_size=env.observation_space.shape[0],hidden_size=36, output_size=env.action_space.n,parameter=hparams)

for batch in range(eparams['num_batches']):
    print('=====\nBATCH {}\n===='.format(batch))

    batch_trajectory, batch_ad = [],[]
    for _ in range(eparams['ep_per_batch']):
        trajectory, rewards, experience_replay_pool = policy_rollout(env, agent, experience_replay_pool)
        print('Episode steps: {}'.format(len(trajectory)))
        batch_trajectory.append(trajectory)
        advantages = process_rewards(rewards)
        batch_ad.append(batch_ad)

    batch_trajectory = sample_batch(experience_replay_pool, batch_size, batch_num)
    agent.train(batch_trajectory)

    #
    # b_obs, b_acts, b_rews = [], [], []
    #
    # for _ in range(eparams['ep_per_batch']):
    #     obs, acts, rews, dones = policy_rollout2(env, agent)
    #
    #     print('Episode steps: {}'.format(len(obs)))
    #     if len(obs) == 200:
    #         print(obs[0:30])
    #
    #     b_obs.extend(obs)
    #     b_acts.extend(acts)
    #
    #     advantages = process_rewards(rews)
    #     b_rews.extend(advantages)
    #
    # # update policy
    # # normalize rewards; don't divide by 0
    # b_rews = (b_rews - np.mean(b_rews)) / (np.std(b_rews) + 1e-10)
    #
    # agent.train_actor(b_obs,b_acts, b_rews)

env.monitor.close()
