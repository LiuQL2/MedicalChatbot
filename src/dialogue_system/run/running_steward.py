# -*-coding: utf-8 -*-

import sys
import os
import pickle
import time
from collections import deque
import copy

sys.path.append(os.getcwd().replace("src/dialogue_system/run",""))

from src.dialogue_system.agent import AgentRule
from src.dialogue_system.agent import AgentDQN
from src.dialogue_system.user_simulator import UserRule as User
from src.dialogue_system.dialogue_manager import DialogueManager
from src.dialogue_system import dialogue_configuration


class RunningSteward(object):
    def __init__(self, parameter):
        self.epoch_size = parameter.get("epoch_size",100)
        self.parameter = parameter
        self.slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
        self.action_set = pickle.load(file=open(parameter["action_set"], "rb"))
        self.goal_set = pickle.load(file=open(parameter["goal_set"], "rb"))
        self.disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))

        user = User(goal_set=self.goal_set, action_set=self.action_set, parameter=parameter)
        agent = AgentRule(action_set=self.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, parameter=parameter)
        self.dialogue_manager = DialogueManager(user=user, agent=agent, parameter=parameter)

        self.best_result = {"success_rate":0.0, "average_reward": 0.0, "average_turn": 0,"average_wrong_disease":10}

    def simulate(self, agent, episodes, train=False):
        self.dialogue_manager.set_agent(agent=agent)
        # if train == True:
        #     for train_index in range(0,10,1):
        #         self.dialogue_manager.train()
        #         self.dialogue_manager.state_tracker.agent.dqn.update_target_network()
        #         print("%2d / %d training dqn using warm start experience buffer."%(train_index,50))
        for index in range(0, episodes,1):
            result = self.simulation_epoch(index)

            if result["success_rate"] >= self.best_result["success_rate"] and \
                    result["success_rate"] > dialogue_configuration.SUCCESS_RATE_THRESHOLD and \
                    result["average_wrong_disease"] <= self.best_result["average_wrong_disease"] and train==True:
                self.dialogue_manager.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
                self.simulation_epoch(index)
                self.dialogue_manager.state_tracker.agent.dqn.save_model(model_performance=result)
                print("The model was saved.")
                self.best_result = copy.deepcopy(result)

            self.dialogue_manager.state_tracker.agent.dqn.update_target_network()

            # if result["success_rate"] >= self.best_result["success_rate"] and \
            #             result["average_wrong_disease"] <= dialogue_configuration.AVERAGE_WRONG_DISEASE:
            #     self.best_result = copy.deepcopy(result)
            #     if isinstance(self.dialogue_manager.state_tracker.agent, AgentDQN):
            #         self.dialogue_manager.state_tracker.agent.dqn.update_target_network()
            #         print("Target network was updated.")
            #         time.sleep(10)
            #     else:
            #         pass

            # Trainning Agent with experience replay
            if train == True:
                self.dialogue_manager.train()

    def simulation_epoch(self,index):
        success_count = 0
        total_reward = 0
        total_truns = 0
        inform_wrong_disease_count = 0
        for epoch_index in range(0,self.epoch_size, 1):
            self.dialogue_manager.initialize()
            while self.dialogue_manager.episode_over == False:
                reward = self.dialogue_manager.next()
                total_reward += reward
            total_truns += self.dialogue_manager.state_tracker.turn
            inform_wrong_disease_count += self.dialogue_manager.inform_wrong_disease_count
            if self.dialogue_manager.dialogue_status == dialogue_configuration.DIALOGUE_SUCCESS:
                success_count += 1
        success_rate = float(success_count) / self.epoch_size
        average_reward = float(total_reward) / self.epoch_size
        average_turn = float(total_truns) / self.epoch_size
        average_wrong_disease = float(inform_wrong_disease_count) / self.epoch_size
        res = {"success_rate":success_rate, "average_reward": average_reward, "average_turn": average_turn, "average_wrong_disease":average_wrong_disease}
        print("%3d simulation success rate %s, ave reward %s, ave turns %s, ave wrong disease %s" % (index,res['success_rate'], res['average_reward'], res['average_turn'], res["average_wrong_disease"]))
        return res

    def warm_start(self, agent, episode_size):
        self.dialogue_manager.set_agent(agent=agent)
        for index in range(0,episode_size,1):
            self.simulation_epoch(index)
            if len(self.dialogue_manager.experience_replay_pool)==self.parameter.get("experience_replay_pool_size"):
                break