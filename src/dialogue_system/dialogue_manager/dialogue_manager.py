# -*- coding:utf-8 -*-

import pickle
import json
import copy
import random
from collections import deque
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/dialogue_manager",""))

from src.dialogue_system.state_tracker import StateTracker as StateTracker
from src.dialogue_system import dialogue_configuration
from src.dialogue_system.agent import AgentRandom
from src.dialogue_system.agent import AgentDQN
from src.dialogue_system.agent import AgentActorCritic
from src.dialogue_system.user_simulator import UserRule as User


class DialogueManager(object):
    """
    Dialogue manager of this dialogue system.
    """
    def __init__(self, user, agent, parameter):
        self.state_tracker = StateTracker(user=user, agent=agent, parameter=parameter)
        self.parameter = parameter
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
        self.inform_wrong_disease_count = 0
        self.trajectory_pool = deque(maxlen=self.parameter.get("trajectory_pool_size",100))
        self.trajectory = []
        self.dialogue_output_file = parameter.get("dialogue_file")
        self.save_dialogue = parameter.get("save_dialogue")

    def next(self,save_record,train_mode, greedy_strategy):
        """
        The next two turn of this dialogue session. The agent will take action first and then followed by user simulator.
        :param save_record: bool, save record?
        :param train_mode: int, 1: the purpose of simulation is to train the model, 0: just for simulation and the
                           parameters of the model will not be updated.
        :return: immediate reward for taking this agent action.
        """
        # Agent takes action.
        state = self.state_tracker.get_state()
        agent_action, action_index = self.state_tracker.agent.next(state=state,turn=self.state_tracker.turn,greedy_strategy=greedy_strategy)
        self.state_tracker.state_updater(agent_action=agent_action)
        # print("turn:%2d, state for agent:\n" % (state["turn"]) , json.dumps(state))

        # User takes action.
        user_action, reward, episode_over, dialogue_status = self.state_tracker.user.next(agent_action=agent_action,turn=self.state_tracker.turn)
        self.state_tracker.state_updater(user_action=user_action)
        # print("turn:%2d, update after user :\n" % (state["turn"]), json.dumps(state))

        # if self.state_tracker.turn == self.state_tracker.max_turn:
        #     episode_over = True

        if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_DISEASE:
            self.inform_wrong_disease_count += 1

        # if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
        #     print("success:", self.state_tracker.user.state)
        # elif dialogue_status == dialogue_configuration.DIALOGUE_STATUS_NOT_COME_YET:
        #     print("not come:", self.state_tracker.user.state)
        # else:
        #     print("failed:", self.state_tracker.user.state)
        # if len(self.state_tracker.user.state["rest_slots"].keys()) ==0:
        #     print(self.state_tracker.user.goal)
        #     print(dialogue_status,self.state_tracker.user.state)

        if save_record == True:
            self.record_training_sample(
                state=state,
                agent_action=action_index,
                next_state=self.state_tracker.get_state(),
                reward=reward,
                episode_over=episode_over
            )
        else:
            pass

        # Output the dialogue.
        if episode_over == True and self.save_dialogue == 1 and train_mode == 0:
            state = self.state_tracker.get_state()
            goal = self.state_tracker.user.get_goal()
            self.__output_dialogue(state=state, goal=goal)

        # Record this episode.
        if episode_over == True:
            self.trajectory_pool.append(copy.deepcopy(self.trajectory))

        return reward, episode_over,dialogue_status

    def initialize(self,train_mode=1, epoch_index=None):
        self.trajectory = []
        self.state_tracker.initialize()
        self.inform_wrong_disease_count = 0
        user_action = self.state_tracker.user.initialize(train_mode = train_mode, epoch_index=epoch_index)
        self.state_tracker.state_updater(user_action=user_action)
        self.state_tracker.agent.initialize()
        # print("#"*30 + "\n" + "user goal:\n", json.dumps(self.state_tracker.user.goal))
        # state = self.state_tracker.get_state()
        # print("turn:%2d, initialized state:\n" % (state["turn"]), json.dumps(state))

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
        state = self.state_tracker.agent.state_to_representation_last(state)
        next_state = self.state_tracker.agent.state_to_representation_last(next_state)
        self.experience_replay_pool.append((state, agent_action, reward, next_state, episode_over))
        self.trajectory.append((state, agent_action, reward, next_state, episode_over))

    def set_agent(self,agent):
        self.state_tracker.set_agent(agent=agent)

    def train(self):
        if isinstance(self.state_tracker.agent, AgentDQN):
            self.__train_dqn()
            self.state_tracker.agent.update_target_network()
        elif isinstance(self.state_tracker.agent, AgentActorCritic):
            self.__train_actor_critic()
            self.state_tracker.agent.update_target_network()

    def __train_dqn(self):
        """
        Train dqn.
        :return:
        """
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size",16)
        for iter in range(int(len(self.experience_replay_pool) / (batch_size))):
            batch = random.sample(self.experience_replay_pool,batch_size)
            loss = self.state_tracker.agent.train(batch=batch)
            cur_bellman_err += loss["loss"]
        print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) / len(self.experience_replay_pool), len(self.experience_replay_pool)))

    def __train_actor_critic(self):
        """
        Train actor-critic.
        :return:
        """
        trajectory_pool = list(self.trajectory_pool)
        batch_size = self.parameter.get("batch_size",16)
        for index in range(0, len(self.trajectory_pool), batch_size):
            stop = max(len(self.trajectory_pool),index + batch_size)
            batch_trajectory = trajectory_pool[index:stop]
            self.state_tracker.agent.train(trajectories=batch_trajectory)


    def __output_dialogue(self,state, goal):
        history = state["history"]
        file = open(file=self.dialogue_output_file,mode="a+",encoding="utf-8")
        file.write("User goal: " + str(goal)+"\n")
        for turn in history:
            speaker = turn["speaker"]
            action = turn["action"]
            inform_slots = turn["inform_slots"]
            request_slots = turn["request_slots"]
            file.write(speaker + ": " + action + "; inform_slots:" + str(inform_slots) + "; request_slots:" + str(request_slots) + "\n")
        file.write("\n\n")
        file.close()