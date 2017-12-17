# -*- coding:utf-8 -*-

import pickle
from collections import deque
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/dialogue_manager",""))

from src.dialogue_system.state_tracker import StateTracker as StateTracker
from src.dialogue_system import dialogue_configuration
from src.dialogue_system.agent import AgentRandom as Agent
from src.dialogue_system.user_simulator import UserRule as User


class DialogueManager(object):
    def __init__(self, user, agent, parameter):
        self.state_tracker = StateTracker(user=user, agent=agent, parameter=parameter)
        self.parameter = parameter
        self.episode_over = False
        self.dialogue_status = dialogue_configuration.NOT_COME_YET

    def next(self):
        state = self.state_tracker.get_state()
        agent_action = self.state_tracker.agent.next(state=state,turn=self.state_tracker.turn)
        # print("turn:%2d, agent action:" % self.state_tracker.turn, agent_action)
        self.state_tracker.state_updater(agent_action=agent_action)


        user_action, self.episode_over, self.dialogue_status = self.state_tracker.user.next(agent_action=agent_action,turn=self.state_tracker.turn)
        # print("turn:%2d, user  action:" % self.state_tracker.turn, user_action)
        self.state_tracker.state_updater(user_action=user_action)

        if self.state_tracker.turn == self.state_tracker.max_turn:
            self.episode_over = True
        else:
            pass

        reward = self._reward_function()
        self.state_tracker.agent.record_training_sample(
            state=state,
            agent_action=agent_action,
            next_state=self.state_tracker.get_state(),
            reward=reward,
            episode_over=self.episode_over
        )

    def initialize(self):
        self.state_tracker.initialize()
        self.episode_over = False
        self.dialogue_status = dialogue_configuration.NOT_COME_YET
        user_action = self.state_tracker.user.initialize()
        self.state_tracker.state_updater(user_action=user_action)
        self.state_tracker.agent.initialize()

    def _reward_function(self):
        if self.dialogue_status == dialogue_configuration.NOT_COME_YET:
            return dialogue_configuration.REWARD_FOR_NOT_COME_YET
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_SUCCESS:
            return dialogue_configuration.REWARD_FOR_DIALOGUE_SUCCESS
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_FAILED:
            return dialogue_configuration.REWARD_FOR_DIALOGUE_FAILED
        elif self.dialogue_status == dialogue_configuration.INFORM_WRONG_DISEASE:
            return dialogue_configuration.REWARD_FOR_INFORM_WRONG_DISEASE