# -*-coding: utf-8 -*-

import sys
import os
import pickle
import time
import json
from collections import deque
import copy

sys.path.append(os.getcwd().replace("src/dialogue_system/run",""))

from src.dialogue_system.agent import AgentRule
from src.dialogue_system.agent import AgentDQN
from src.dialogue_system.agent import AgentActorCritic
from src.dialogue_system.user_simulator import UserRule as User
from src.dialogue_system.dialogue_manager import DialogueManager
from src.dialogue_system import dialogue_configuration


class RunningSteward(object):
    """
    The steward of running the dialogue system.
    """
    def __init__(self, parameter, checkpoint_path):
        self.epoch_size = parameter.get("epoch_size",100)
        self.parameter = parameter
        self.checkpoint_path = checkpoint_path
        self.slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
        self.action_set = pickle.load(file=open(parameter["action_set"], "rb"))
        self.goal_set = pickle.load(file=open(parameter["goal_set"], "rb"))
        self.disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))
        self.learning_curve = {}

        user = User(goal_set=self.goal_set, action_set=self.action_set, parameter=parameter)
        agent = AgentRule(action_set=self.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, parameter=parameter)
        self.dialogue_manager = DialogueManager(user=user, agent=agent, parameter=parameter)

        self.best_result = {"success_rate":0.0, "average_reward": 0.0, "average_turn": 0,"average_wrong_disease":10}

    def simulate(self, agent, epoch_number, train_mode=0):
        """
        Simulating between agent and user simulator.
        :param agent: the agent used to simulate, an instance of class Agent.
        :param epoch_number: the epoch number of simulation.
        :param train_mode: int, 1: the purpose of simulation is to train the model, 0: just for simulation and the
                           parameters of the model will not be updated.
        :return: nothing to return.
        """
        save_model = self.parameter.get("save_model")
        self.dialogue_manager.set_agent(agent=agent)
        # self.dialogue_manager.state_tracker.user.set_max_turn(max_turn=self.parameter.get('max_turn'))
        for index in range(0, epoch_number,1):
            # Training AgentDQN with experience replay
            if train_mode == 1 and isinstance(self.dialogue_manager.state_tracker.agent, AgentDQN):
                self.dialogue_manager.train()
                # Simulating and filling experience replay pool.
                self.simulation_epoch(epoch_size=self.epoch_size,train_mode=train_mode)
            # Training AgentActorCritic with sampling one trajectory.
            elif train_mode == 1 and isinstance(self.dialogue_manager.state_tracker.agent, AgentActorCritic):
                # Sample one trajectory for training.
                # for _i in range(self.epoch_size):
                    self.simulation_epoch(epoch_size=self.epoch_size,train_mode=train_mode)
                    self.dialogue_manager.train()

            # Evaluating the model.
            result = self.evaluate_model(index)
            if result["success_rate"] >= self.best_result["success_rate"] and \
                    result["success_rate"] > dialogue_configuration.SUCCESS_RATE_THRESHOLD and \
                    result["average_wrong_disease"] <= self.best_result["average_wrong_disease"] and train_mode==1:
                self.dialogue_manager.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
                self.simulation_epoch(epoch_size=self.epoch_size,train_mode=train_mode)
                if save_model == 1:
                    self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index = index, checkpoint_path=self.checkpoint_path)
                    print("The model was saved.")
                else:
                    pass
                self.best_result = copy.deepcopy(result)

    def simulation_epoch(self, epoch_size,train_mode):
        """
        Simulating one epoch when training model.
        :param epoch_size: the size of each epoch, i.e., the number of dialogue sessions of each epoch.
        :return: a dict of simulation results including success rate, average reward, average number of wrong diseases.
        """
        success_count = 0
        absolute_success_count = 0
        total_reward = 0
        total_truns = 0
        inform_wrong_disease_count = 0
        for epoch_index in range(0,epoch_size, 1):
            self.dialogue_manager.initialize(train_mode=self.parameter.get("train_mode"))
            episode_over = False
            while episode_over == False:
                reward, episode_over, dialogue_status = self.dialogue_manager.next(save_record=True,train_mode=train_mode,greedy_strategy=1)
                total_reward += reward
            total_truns += self.dialogue_manager.state_tracker.turn
            inform_wrong_disease_count += self.dialogue_manager.inform_wrong_disease_count
            if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
                if self.dialogue_manager.inform_wrong_disease_count == 0:
                    absolute_success_count += 1
        success_rate = float("%.3f" % (float(success_count) / epoch_size))
        absolute_success_rate = float("%.3f" % (float(absolute_success_count) / epoch_size))
        average_reward = float("%.3f" % (float(total_reward) / epoch_size))
        average_turn = float("%.3f" % (float(total_truns) / epoch_size))
        average_wrong_disease = float("%.3f" % (float(inform_wrong_disease_count) / epoch_size))
        res = {"success_rate":success_rate, "average_reward": average_reward, "average_turn": average_turn, "average_wrong_disease":average_wrong_disease,"ab_success_rate":absolute_success_rate}
        # print("%3d simulation success rate %s, ave reward %s, ave turns %s, ave wrong disease %s" % (index,res['success_rate'], res['average_reward'], res['average_turn'], res["average_wrong_disease"]))
        return res

    def evaluate_model(self,index):
        """
        Evaluating model during training.
        :param index: int, the simulation index.
        :return: a dict of evaluation results including success rate, average reward, average number of wrong diseases.
        """
        save_performance = self.parameter.get("save_performance")

        train_mode = self.parameter.get("train_mode")
        success_count = 0
        absolute_success_count = 0
        total_reward = 0
        total_truns = 0
        evaluate_epoch_number = self.parameter.get("evaluate_epoch_number")
        # evaluate_epoch_number = len(self.dialogue_manager.state_tracker.user.goal_set["test"])
        inform_wrong_disease_count = 0
        for epoch_index in range(0,evaluate_epoch_number, 1):
            self.dialogue_manager.initialize(train_mode=train_mode, epoch_index=epoch_index)
            episode_over = False
            while episode_over == False:
                reward, episode_over, dialogue_status = self.dialogue_manager.next(save_record=False,train_mode=train_mode,greedy_strategy=0)
                total_reward += reward
            total_truns += self.dialogue_manager.state_tracker.turn
            inform_wrong_disease_count += self.dialogue_manager.inform_wrong_disease_count
            if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
                if self.dialogue_manager.inform_wrong_disease_count == 0:
                    absolute_success_count += 1
        success_rate = float("%.3f" % (float(success_count) / evaluate_epoch_number))
        absolute_success_rate = float("%.3f" % (float(absolute_success_count) / evaluate_epoch_number))
        average_reward = float("%.3f" % (float(total_reward) / evaluate_epoch_number))
        average_turn = float("%.3f" % (float(total_truns) / evaluate_epoch_number))
        average_wrong_disease = float("%.3f" % (float(inform_wrong_disease_count) / evaluate_epoch_number))
        res = {"success_rate":success_rate, "average_reward": average_reward, "average_turn": average_turn, "average_wrong_disease":average_wrong_disease,"ab_success_rate":absolute_success_rate}
        self.learning_curve.setdefault(index, dict())
        self.learning_curve[index]["success_rate"]=success_rate
        self.learning_curve[index]["average_reward"]=average_reward
        self.learning_curve[index]["average_turn"] = average_turn
        self.learning_curve[index]["average_wrong_disease"]=average_wrong_disease
        if index % 10 ==0:
            self.__print_run_info__()
        if index % 100 == 99 and save_performance == 1:
            self.__dump_performance__(epoch_index=index)
        print("%3d simulation SR %s, ABSR %s, ave reward %s, ave turns %s, ave wrong disease %s" % (index,res['success_rate'], res["ab_success_rate"],res['average_reward'], res['average_turn'], res["average_wrong_disease"]))
        return res

    def warm_start(self, agent, epoch_number):
        """
        Warm-starting the dialogue, using the sample from rule-based agent to fill the experience replay pool for DQN.
        :param agent: the agent used to warm start dialogue system.
        :param epoch_number: the number of epoch when warm starting, and the number of dialogue sessions of each epoch
                             equals to the simulation epoch.
        :return: nothing to return.
        """
        self.dialogue_manager.set_agent(agent=agent)
        # self.dialogue_manager.state_tracker.user.set_max_turn(max_turn = 2*len(self.slot_set))
        for index in range(0,epoch_number,1):
            res = self.simulation_epoch(epoch_size=self.epoch_size,train_mode=1)
            print("%3d simulation SR %s, ABSR %s,ave reward %s, ave turns %s, ave wrong disease %s" % (
            index, res['success_rate'], res["ab_success_rate"], res['average_reward'], res['average_turn'], res["average_wrong_disease"]))
            # if len(self.dialogue_manager.experience_replay_pool)==self.parameter.get("experience_replay_pool_size"):
            #     break

    def __dump_performance__(self,epoch_index):
        agent_id = self.parameter.get("agent_id")
        dqn_id = self.parameter.get("dqn_id")
        disease_number = self.parameter.get("disease_number")
        lr = self.parameter.get("dqn_learning_rate")
        reward_for_success = self.parameter.get("reward_for_success")
        reward_for_fail = self.parameter.get("reward_for_fail")
        reward_for_not_come_yet = self.parameter.get("reward_for_not_come_yet")
        reward_for_inform_right_symptom = self.parameter.get("reward_for_inform_right_symptom")

        max_turn = self.parameter.get("max_turn")
        minus_left_slots = self.parameter.get("minus_left_slots")
        gamma = self.parameter.get("gamma")
        epsilon = self.parameter.get("epsilon")
        run_id = self.parameter.get('run_id')

        if agent_id == 1:
            file_name = "learning_rate_d" + str(disease_number) + "_e" + "_agent" + str(agent_id) + \
                        "_dqn" + str(dqn_id) + "_T" + str(max_turn) + "_lr" + str(lr) + "_RFS" + str(reward_for_success) +\
                          "_RFF" + str(reward_for_fail) + "_RFNCY" + str(reward_for_not_come_yet) + "_RFIRS" + str(reward_for_inform_right_symptom) +\
                          "_mls" + str(minus_left_slots) + "_gamma" + str(gamma) + "_epsilon" + str(epsilon) + "_RID" + str(run_id) + "_" + str(epoch_index) + ".p"
        else:
            file_name = "learning_rate_d" + str(disease_number) + "_e" + "_agent" + str(agent_id) + \
                        "_T" + str(max_turn) + "_lr" + str(lr) + "_RFS" + str(reward_for_success) +\
                          "_RFF" + str(reward_for_fail) + "_RFNCY" + str(reward_for_not_come_yet) + "_RFIRS" + str(reward_for_inform_right_symptom) +\
                          "_mls" + str(minus_left_slots) + "_gamma" + str(gamma) + "_epsilon" + str(epsilon)  + "_RID" + str(run_id) + "_" + str(epoch_index) + ".p"

        pickle.dump(file=open(self.parameter.get("performance_save_path") + file_name, "wb"), obj=self.learning_curve)

    def __print_run_info__(self):
        # print(json.dumps(self.parameter, indent=2))
        agent_id = self.parameter.get("agent_id")
        dqn_id = self.parameter.get("dqn_id")
        disease_number = self.parameter.get("disease_number")
        lr = self.parameter.get("dqn_learning_rate")
        reward_for_success = self.parameter.get("reward_for_success")
        reward_for_fail = self.parameter.get("reward_for_fail")
        reward_for_not_come_yet = self.parameter.get("reward_for_not_come_yet")
        reward_for_inform_right_symptom = self.parameter.get("reward_for_inform_right_symptom")

        max_turn = self.parameter.get("max_turn")
        minus_left_slots = self.parameter.get("minus_left_slots")
        gamma = self.parameter.get("gamma")
        epsilon = self.parameter.get("epsilon")
        data_set_name = self.parameter.get("goal_set").split("/")[-2]
        run_id = self.parameter.get('run_id')
        info = "learning_rate_d" + str(disease_number) + "_agent" + str(agent_id) + \
               "_dqn" + str(dqn_id) + "_T" + str(max_turn) + "_lr" + str(lr) + "_RFS" + str(reward_for_success) +\
                          "_RFF" + str(reward_for_fail) + "_RFNCY" + str(reward_for_not_come_yet) + "_RFIRS" + str(reward_for_inform_right_symptom) +\
                          "_mls" + str(minus_left_slots) + "_gamma" + str(gamma) + "_epsilon" + str(epsilon)  + "_RID" + str(run_id) + "_data" + str(data_set_name)

        print("[INFO]:", info)