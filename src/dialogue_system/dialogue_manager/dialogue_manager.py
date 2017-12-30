# -*- coding:utf-8 -*-

import pickle
import json
import random
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
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))

    def next(self):
        # Agent takes action.
        state = self.state_tracker.get_state()
        agent_action, action_index = self.state_tracker.agent.next(state=state,turn=self.state_tracker.turn)
        self.state_tracker.state_updater(agent_action=agent_action)
        # print("turn:%2d, agent action:" % (self.state_tracker.turn -1) , agent_action)
        # print("turn:%2d, state for agent:\n" % (self.state_tracker.turn -1) , json.dumps(state))

        # User takes action.
        user_action, self.episode_over, self.dialogue_status = self.state_tracker.user.next(agent_action=agent_action,turn=self.state_tracker.turn)
        self.state_tracker.state_updater(user_action=user_action)
        # print("turn:%2d, user  action:" % (self.state_tracker.turn - 1), user_action)
        # print("turn:%2d, update after user :" % (self.state_tracker.turn - 1), state)

        if self.state_tracker.turn == self.state_tracker.max_turn:
            self.episode_over = True
        else:
            pass

        reward = self._reward_function()
        self.record_training_sample(
            state=state,
            agent_action=action_index,
            next_state=self.state_tracker.get_state(),
            reward=reward,
            episode_over=self.episode_over
        )

        return reward

    def initialize(self):
        self.state_tracker.initialize()
        self.episode_over = False
        self.dialogue_status = dialogue_configuration.NOT_COME_YET
        user_action = self.state_tracker.user.initialize()
        self.state_tracker.state_updater(user_action=user_action)
        self.state_tracker.agent.initialize()
        # print("turn:%2d, user  action:" % (self.state_tracker.turn - 1), user_action)

    def _reward_function(self):
        if self.dialogue_status == dialogue_configuration.NOT_COME_YET:
            return dialogue_configuration.REWARD_FOR_NOT_COME_YET
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_SUCCESS:
            return dialogue_configuration.REWARD_FOR_DIALOGUE_SUCCESS
        elif self.dialogue_status == dialogue_configuration.DIALOGUE_FAILED:
            return dialogue_configuration.REWARD_FOR_DIALOGUE_FAILED
        elif self.dialogue_status == dialogue_configuration.INFORM_WRONG_DISEASE:
            return dialogue_configuration.REWARD_FOR_INFORM_WRONG_DISEASE

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
        state = self.state_tracker.agent.state_to_representation_last(state)
        next_state = self.state_tracker.agent.state_to_representation_last(next_state)
        self.experience_replay_pool.append((state, agent_action, reward, next_state, episode_over))

    def set_agent(self,agent):
        self.state_tracker.set_agent(agent=agent)

    def train(self):
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size",16)
        for iter in range(int(len(self.experience_replay_pool) / (batch_size))):
            batch = random.sample(self.experience_replay_pool,batch_size)
            loss = self.state_tracker.agent.train(batch=batch)
            cur_bellman_err += loss["loss"]
        print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) / len(self.experience_replay_pool), len(self.experience_replay_pool)))
