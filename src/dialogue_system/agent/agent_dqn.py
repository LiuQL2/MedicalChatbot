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
        input_size = parameter.get("input_size_dqn", 182)
        hidden_size = parameter.get("hidden_size_dqn", 100)
        output_size = len(self.action_sapce)
        checkpoint_path = parameter.get("checkpoint_path")
        self.dqn = DQN(input_size=input_size, hidden_size=hidden_size,output_size=output_size, checkpoint_path=checkpoint_path)

    def next(self, state, turn):
        # TODO (Qianlong): take action condition on current state.
        self.agent_action["turn"] = turn
        state_rep = self.state_to_representation_last(state=state) # sequence representation.

        action_index = self.dqn.predict(Xs=[state_rep])[1]
        agent_action = self.action_sapce[action_index]
        agent_action["turn"] = turn
        agent_action["speaker"] = "agent"
        return agent_action, action_index

    def train(self, batch):
        return self.dqn.singleBatch(batch=batch,params=self.parameter)

