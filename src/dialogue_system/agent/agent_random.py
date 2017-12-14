# -*- coding: utf-8 -*-
"""
An agent that randomly choose an action from action_set.
"""
import random
from collections import deque

import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))


class AgentRandom(object):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        self.action_set = action_set
        self.slot_set = slot_set
        self.disease_symptom = disease_symptom
        self.turn = 0
        self.max_turn = parameter["max_turn"]

    def next(self, state,turn):
        self.turn = turn
        action = random.choice(self.action_set.keys())
        agent_action = {
            "action":action,
            "speaker":"agent"
        }
        return agent_action

    def initialize(self):
        pass