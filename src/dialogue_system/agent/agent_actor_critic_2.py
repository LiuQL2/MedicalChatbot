#-*- coding:utf-8 -*-
"""
An RL-based agent which learns policy using actor-critic algorithm.
"""

import random
import numpy as np
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))

from src.dialogue_system.policy_learning.actor_critic_4 import ActorCritic
from src.dialogue_system.agent.agent import Agent


class AgentActorCritic(Agent):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        super(AgentActorCritic, self).__init__(action_set=action_set, slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
        input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn")
        output_size = len(self.action_sapce)
        self.actor_critic = ActorCritic(input_size=input_size, hidden_size=hidden_size,output_size=output_size, parameter=parameter)

    def next(self, state, turn, greedy_strategy=None):
        # TODO (Qianlong): take action condition on current state.
        self.agent_action["turn"] = turn
        state_rep = self.state_to_representation_last(state=state) # sequence representation.
        action_index = self.actor_critic.take_action(state_rep)
        agent_action = self.action_sapce[action_index]

        agent_action["turn"] = turn
        agent_action["speaker"] = "agent"

        return agent_action, action_index

    def train(self, trajectories):
        self.actor_critic.train(trajectories=trajectories)

    def update_target_network(self):
        pass

    def save_model(self, model_performance,episodes_index, checkpoint_path = None):
        pass