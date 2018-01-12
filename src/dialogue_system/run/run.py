# -*- coding:utf-8 -*-

import time
import argparse
import pickle
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/run",""))

from src.dialogue_system.dialogue_manager import DialogueManager
# from src.dialogue_system.agent import AgentRandom as Agent
from src.dialogue_system.agent import AgentDQN as Agent
from src.dialogue_system.agent import AgentRule
from src.dialogue_system.user_simulator import UserRule as User
from src.dialogue_system import dialogue_configuration

from src.dialogue_system.run import RunningSteward

parser = argparse.ArgumentParser()
parser.add_argument("--action_set", dest="action_set", type=str, default='./../data/action_set.p', help='path and filename of the action set')
parser.add_argument("--slot_set", dest="slot_set", type=str, default='./../data/slot_set.p', help='path and filename of the slots set')
parser.add_argument("--goal_set", dest="goal_set", type=str, default='./../data/goal_set.p', help='path and filename of user goal')
parser.add_argument("--disease_symptom", dest="disease_symptom", type=str, default="./../data/disease_symptom.p", help="path and filename of the disease_symptom file")
parser.add_argument("--max_turn", dest="max_turn", type=int, default=40, help="the max turn in one episode.")
parser.add_argument("--episodes", dest="episodes", type=int, default=2000, help="the number of episodes.")
parser.add_argument("--epoch_size", dest="epoch_size", type=int, default=100, help="the size of each simulation.")
parser.add_argument("--experience_replay_pool_size", dest="experience_replay_pool_size", type=int, default=50000, help="the size of experience replay.")
parser.add_argument("--gamma", dest="gamma", type=float, default=0.9, help="The discount factor of immediate reward.")
parser.add_argument("--hidden_size_dqn", dest="hidden_size_dqn", type=int, default=100, help="the hidden_size of DQN.")
parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=182, help="the input_size of DQN.")
parser.add_argument("--warm_start", dest="warm_start",type=int, default=1, help="use rule policy to fill the experience replay buffer at the beginning, 1:True; 0:False")
parser.add_argument("--warm_start_episodes", dest="warm_start_episodes", type=int, default=20, help="the number of episodes of warm start.")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=16, help="the batch size when training.")
parser.add_argument("--checkpoint_path",dest="checkpoint_path", type=str, default="./../model/checkpoint/", help="the folder where models save to, ending with /.")
parser.add_argument("--log_dir", dest="log_dir", type=str, default="./../../../log/", help="directory where event file of training will be written, ending with /")
parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.1, help="the greedy of DQN")

args = parser.parse_args()
parameter = vars(args)


def run():
    slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
    action_set = pickle.load(file=open(parameter["action_set"], "rb"))
    disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))

    steward = RunningSteward(parameter=parameter)

    warm_start = parameter.get("warm_start")
    warm_start_episodes = parameter.get("warm_start_episodes")

    # Warm start.
    if warm_start == 1:
        agent = AgentRule(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
        steward.warm_start(agent=agent,episode_size=warm_start_episodes)

    episodes = parameter.get("episodes")
    agent = Agent(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)

    steward.simulate(agent=agent,episodes=episodes, train=True)


if __name__ == "__main__":
    run()