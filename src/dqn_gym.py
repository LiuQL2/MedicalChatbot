# -*- coding:utf-8 -*-
from collections import deque
import gym
import random
import argparse
from src.dialogue_system.policy_learning.dqn import DQN
# env = gym.make('CartPole-v0')
# print(env.action_space)
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         # env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print(reward)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break


class CartPole(object):
    def __init__(self, parameter):
        self.action_sapce = [0,1]
        self.dqn = DQN(4,20,2, parameter=parameter)
        self.parameter = parameter
        self.experience_replay_pool = deque(maxlen=10000)
        self.env = gym.make('CartPole-v0')
        self.epoch_size = parameter.get("epoch_size")
        self.episodes_number = parameter.get("episodes")
        self.max_turn = parameter.get("max_turn")

    def simulate(self):
        for episode_index in range(0,self.episodes_number,1):
            self.simulation_epoch(episode_index)
            self.dqn.update_target_network()
            self.train()

    def simulation_epoch(self,index):
        total_reward = 0
        total_truns = 0
        for epoch_index in range(0,self.epoch_size, 1):
            observation = self.env.reset()
            for t in range(100):
                # env.render()
                action = self.take_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                self.record_training_sample(observation,action,reward,next_observation,done)
                total_reward += reward
                observation=next_observation
                total_truns += 1
                if done:
                    # print("Episode finished after {} timesteps".format(t + 1))
                    break
        average_reward = float(total_reward) / self.epoch_size
        average_turn = float(total_truns) / self.epoch_size
        res = {"average_reward": average_reward, "average_turn": average_turn}
        print("%3d, ave reward %s, ave turns %s" % (index, res['average_reward'], res['average_turn']))
        return res

    def record_training_sample(self, state, action, reward, next_state, done):
        self.experience_replay_pool.append((state, action, reward, next_state, done))

    def train(self):
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size",16)
        for iter in range(int(len(self.experience_replay_pool) / (batch_size))):
            batch = random.sample(self.experience_replay_pool,batch_size)
            loss = self.dqn.singleBatch(batch=batch,params=self.parameter)
            cur_bellman_err += loss["loss"]
        # print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) / len(self.experience_replay_pool), len(self.experience_replay_pool)))

    def take_action(self, state):
        greedy = random.random()
        if greedy < self.parameter.get("epsilon"):
            action = random.randint(0, len(self.action_sapce)-1)
        else:
            action = self.dqn.predict(Xs=[state])[1]
        return action



parser = argparse.ArgumentParser()
parser.add_argument("--max_turn", dest="max_turn", type=int, default=100, help="the max turn in one episode.")
parser.add_argument("--episodes", dest="episodes", type=int, default=2000, help="the number of episodes.")
parser.add_argument("--epoch_size", dest="epoch_size", type=int, default=100, help="the size of each simulation.")
parser.add_argument("--experience_replay_pool_size", dest="experience_replay_pool_size", type=int, default=50000, help="the size of experience replay.")
parser.add_argument("--gamma", dest="gamma", type=float, default=0.9, help="The discount factor of immediate reward.")
parser.add_argument("--hidden_size_dqn", dest="hidden_size_dqn", type=int, default=100, help="the hidden_size of DQN.")
parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=182, help="the input_size of DQN.")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=16, help="the batch size when training.")
parser.add_argument("--checkpoint_path",dest="checkpoint_path", type=str, default="./../model/checkpoint01/", help="the folder where models save to, ending with /.")
parser.add_argument("--log_dir", dest="log_dir", type=str, default="./../../../log/", help="directory where event file of training will be written, ending with /")
parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.1, help="the greedy of DQN")

args = parser.parse_args()
parameter = vars(args)

if __name__ == "__main__":
    cart = CartPole(parameter=parameter)
    cart.simulate()