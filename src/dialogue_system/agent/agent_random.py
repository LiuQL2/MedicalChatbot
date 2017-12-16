# -*- coding: utf-8 -*-
"""
An agent that randomly choose an action from action_set.
"""
import random

import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))

from src.dialogue_system.agent.agent import Agent


class AgentRandom(Agent):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        super(AgentRandom, self).__init__(action_set=action_set, slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
        self.max_turn = parameter["max_turn"]

    def next(self, state,turn):
        self.agent_action["turn"] = turn
        action = random.choice(self.action_set.keys())
        agent_action = {
            "action":action,
            "speaker":"agent"
        }
        return agent_action