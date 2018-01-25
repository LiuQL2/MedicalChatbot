# -*- coding:utf-8 -*-

import gym
import argparse
from src.dialogue_system.policy_learning.actor_critic import ActorCritic
import gym.spaces.box

parser = argparse.ArgumentParser()
# For Actor-critic
parser.add_argument("--actor_learning_rate", dest="actor_learning_rate", type=float, default=0.001, help="the learning rate of actor")
parser.add_argument("--critic_learning_rate", dest="critic_learning_rate", type=float, default=0.001, help="the learning rate of critic")
parser.add_argument("--trajectory_pool_size", dest="trajectory_pool_size", type=int, default=100, help="the size of trajectory pool")
parser.add_argument("--gamma", dest="gamma", type=float, default=0.9, help="The discount factor of immediate reward.")

args = parser.parse_args()
parameter = vars(args)

# env = gym.make("MountainCar-v0")
env = gym.make("Acrobot-v1")
print(type(env.observation_space))
print(env.observation_space.shape)
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
hidden_size = 10


actor_critic = ActorCritic(input_size,hidden_size,output_size,parameter)

# state, agent_action, reward, next_state, episode_over
def simulate():
    total_reward = 0
    trajectory_pool = []
    episode_size = 100
    for i_episode in range(episode_size):
        observation = env.reset()
        trajectory = []
        done = False
        while done == False:
            # env.render()
            # action = env.action_space.sample()
            action = actor_critic.take_action(observation)
            next_observation, reward, done, info = env.step(action)
            total_reward += reward
            trajectory.append((observation, action,reward,next_observation, done))
            observation = next_observation
            if done:
                # print("Episode finished after {} timesteps".format(t + 1))
                break
        trajectory_pool.append(trajectory)
    return trajectory_pool, total_reward/float(episode_size)

def train(trajectory_pool):
    for trajectory in trajectory_pool:
        actor_critic.train(trajectory)

def run():
    for _i in range(0, 2000,1):
        trajectory_pool, average_reward = simulate()
        print("%3d, average reward: %4f"%(_i, average_reward))
        # if average_reward >= -110.0:
        #     break
        train(trajectory_pool)

if __name__ == "__main__":
    run()