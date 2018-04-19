# -*-coding:utf-8 -*
"""
The agent will maintain two ranked list of candidate disease and symptoms, the two list will be updated every turn based
on the information agent collected. The two ranked list will affect each other according <disease-symptom> pairs.
Agent will choose the first symptom with request as the agent action aiming to ask if the user has the symptom. The rank
model will change if the user's answer is no in continual several times.
"""

import random
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))

from src.dialogue_system.agent.agent import Agent


class AgentDQN(Agent):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        super(AgentDQN, self).__init__(action_set=action_set, slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
        input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 100)
        output_size = len(self.action_sapce)
        dqn_id = self.parameter.get("dqn_id")
        if dqn_id == 0:
            from src.dialogue_system.policy_learning import DQN0 as DQN
        elif dqn_id == 1:
            from src.dialogue_system.policy_learning import DQN1 as DQN
        elif dqn_id == 2:
            from src.dialogue_system.policy_learning import DQN2 as DQN
        elif dqn_id == 3:
            from src.dialogue_system.policy_learning import DQN3 as DQN

        self.dqn = DQN(input_size=input_size, hidden_size=hidden_size,output_size=output_size, parameter=parameter)

    def next(self, state, turn,greedy_strategy):
        # TODO (Qianlong): take action condition on current state.
        self.agent_action["turn"] = turn
        state_rep = self.state_to_representation_last(state=state) # sequence representation.

        if greedy_strategy == 1:
            greedy = random.random()
            if greedy < self.parameter.get("epsilon"):
                action_index = random.randint(0, len(self.action_sapce) - 1)
            else:
                action_index = self.dqn.predict(Xs=[state_rep])[1]
        # Evaluating mode.
        else:
            action_index = self.dqn.predict(Xs=[state_rep])[1]

        agent_action = self.action_sapce[action_index]
        agent_action["turn"] = turn
        agent_action["speaker"] = "agent"

        return agent_action, action_index

    def train(self, batch):
        loss = self.dqn.singleBatch(batch=batch,params=self.parameter)
        return loss

    def update_target_network(self):
        self.dqn.update_target_network()

    def save_model(self, model_performance,episodes_index, checkpoint_path = None):
        self.dqn.save_model(model_performance=model_performance, episodes_index = episodes_index, checkpoint_path=checkpoint_path)
