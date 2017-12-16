# -*- coding:utf-8 -*-

import time
import argparse
import pickle
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system",""))

from src.dialogue_system.dialogue_manager import DialogueManager
# from src.dialogue_system.agent import AgentRandom as Agent
# from src.dialogue_system.agent import AgentDQN as Agent
from src.dialogue_system.agent import AgentRule as Agent
from src.dialogue_system.user_simulator import UserRule as User
from src.dialogue_system import dialogue_configuration

parser = argparse.ArgumentParser()
parser.add_argument("--action_set", dest="action_set", type=str, default='./dialogue_system/data/action_set.p', help='path and filename of the action set')
parser.add_argument("--slot_set", dest="slot_set", type=str, default='./dialogue_system/data/slot_set.p', help='path and filename of the slots set')
parser.add_argument("--goal_set", dest="goal_set", type=str, default='./dialogue_system/data/goal_set.p', help='path and filename of user goal')
parser.add_argument("--disease_symptom", dest="disease_symptom", type=str, default="./dialogue_system/data/disease_symptom.p", help="path and filename of the disease_symptom file")
parser.add_argument("--max_turn", dest="max_turn", type=int, default=40, help="the max turn in one episode.")
parser.add_argument("--episodes", dest="episodes", type=int, default=1000, help="the number of episodes.")
parser.add_argument("--experience_replay_pool_size", dest="experience_replay_pool_size", type=int, default=10000, help="the size of experience replay.")

args = parser.parse_args()
parameter = vars(args)
episodes = parameter["episodes"]


def run():
    success_count = 0
    slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
    action_set = pickle.load(file=open(parameter["action_set"], "rb"))
    goal_set = pickle.load(file=open(parameter["goal_set"], "rb"))
    disease_symptom = pickle.load(file=open(parameter["disease_symptom"],"rb"))
    user = User(goal_set=goal_set, action_set=action_set,parameter=parameter)
    agent = Agent(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
    dialogue_manager = DialogueManager(user=user, agent=agent,parameter=parameter)
    for index in range(0, episodes, 1):
        dialogue_manager.initialize()
        while dialogue_manager.episode_over==False:
            dialogue_manager.next()
        if dialogue_manager.dialogue_status == dialogue_configuration.DIALOGUE_SUCCESS:
            # print("Dialogue Success!")
            success_count = success_count + 1
        if index % 100 == 0:
            print("success rate:", success_count / 100)
            success_count = 0


if __name__ == "__main__":
    run()