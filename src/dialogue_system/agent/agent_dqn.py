# -*-coding:utf-8 -*
"""
The agent will maintain two ranked list of candidate disease and symptoms, the two list will be updated every turn based
on the information agent collected. The two ranked list will affect each other according <disease-symptom> pairs.
Agent will choose the first symptom with request as the agent action aiming to ask if the user has the symptom. The rank
model will change if the user's answer is no in continual several times.
"""

import numpy as np
import copy
import random
import json
import sys, os
from collections import deque
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))

from src.dialogue_system import dialogue_configuration
from src.dialogue_system.policy_learning import DQN
from src.dialogue_system.agent.agent import Agent


class AgentDQN(Agent):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        super(AgentDQN, self).__init__(action_set=action_set, slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
        self.dqn = DQN()

    def initialize(self):
        self.candidate_disease_list = []
        self.candidate_symptom_list = []
        self.agent_action = {
            "turn":None,
            "action":None,
            "request_slots":{},
            "inform_slots":{},
            "explicit_inform_slots":{},
            "implicit_inform_slots":{},
            "speaker":"agent"
        }

    def next(self, state, turn):
        # TODO (Qianlong): take action condition on current state.
        self.agent_action["turn"] = turn
        state_rep = self._state_to_representation_history(state=state) # sequence representation.
        print(state_rep)
        print(state_rep.shape)
        # state_rep = np.array(state_rep)
        # print(state_rep)
        self.dqn.get_state(state_rep=state_rep)
        self.agent_action["action"] = "request"
        self.agent_action["request_slots"] = {
            random.choice(list(self.slot_set.keys())): dialogue_configuration.VALUE_UNKNOWN
        }
        return self.agent_action