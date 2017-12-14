# -*- coding:utf-8 -*-

import pickle
from collections import deque
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/dialogue_manager",""))

from src.dialogue_system.state_tracker import StateTracker as StateTracker
from src.dialogue_system.agent import AgentRandom as Agent
from src.dialogue_system.user_simulator import UserRule as User


class DialogueManager(object):
    def __init__(self, user, agent, parameter):
        self.state_tracker = StateTracker(user=user, agent=agent, parameter=parameter)
        self.parameter = parameter
        self.episode_over = False
        self.experience_replay_pool = deque(maxlen=self.parameter["experience_replay_pool_size"])

    def next(self):
        state = self.state_tracker.get_state()
        agent_action = self.state_tracker.agent.next(state=state,turn=self.state_tracker.turn)
        print("turn:%2d, agent action:" % self.state_tracker.turn, agent_action)
        self.state_tracker.state_updater(agent_action=agent_action)

        user_action, self.episode_over, dialogue_status = self.state_tracker.user.next(agent_action=agent_action,turn=self.state_tracker.turn)
        print("turn:%2d, user  action:" % self.state_tracker.turn, user_action)
        self.state_tracker.state_updater(user_action=user_action)

        if self.state_tracker.turn == self.state_tracker.max_turn:
            self.episode_over = True
        else:
            pass
        return self.episode_over

    def initialize(self):
        self.state_tracker.initialize()
        self.episode_over = False
        self.experience_replay_pool = deque(maxlen=self.parameter["experience_replay_pool_size"])
        user_action = self.state_tracker.user.initialize()
        self.state_tracker.state_updater(user_action=user_action)
        self.state_tracker.agent.initialize()
