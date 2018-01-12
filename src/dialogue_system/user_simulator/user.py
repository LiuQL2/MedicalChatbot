# -*- coding:utf-8 -*-
"""
Basic user simulator, random choice action.

# Structure of agent_action:
agent_action = {
    "turn":0,
    "speaker":"agent",
    "action":"request",
    "request_slots":{},
    "inform_slots":{},
    "explicit_inform_slots":{},
    "implicit_inform_slots":{}
}

# Structure of user_action:
user_action = {
    "turn": 0,
    "speaker": "user",
    "action": "request",
    "request_slots": {},
    "inform_slots": {},
    "explicit_inform_slots": {},
    "implicit_inform_slots": {}
}

# Structure of user goal.
{
  "consult_id": "10002219",
  "disease_tag": "上呼吸道感染",
  "goal": {
    "request_slots": {
      "disease": "UNK"
    },
    "explicit_inform_slots": {
      "呼吸不畅": true,
      "发烧": true
    },
    "implicit_inform_slots": {
      "厌食": true,
      "鼻塞": true
    }
  }

"""

import random
import copy

import sys,os
sys.path.append(os.getcwd().replace("src/dialogue_system",""))

from src.dialogue_system import dialogue_configuration


class User(object):
    def __init__(self, goal_set, action_set, parameter):
        self.goal_set = goal_set
        self.action_set = action_set
        self.max_turn = parameter["max_turn"]
        self.parameter = parameter
        self._init()

    def initialize(self):
        self._init()

        # Initialize rest slot for this user.
        # 初始的时候request slot里面必有disease，然后explicit_inform_slots里面所有slot全部取出进行用户主诉的构建，若explicit里面没
        # 有slot，初始就只有一个request slot，里面是disease，因为implicit_inform_slots是需要与agent交互的过程中才能发现的，患者自己并
        # 不能发现自己隐含的一些症状。
        goal = self.goal["goal"]
        self.state["action"] = "request"
        self.state["request_slots"]["disease"] = dialogue_configuration.VALUE_UNKNOWN

        for slot in goal["explicit_inform_slots"].keys():
            # self.state["explicit_inform_slots"][slot] = goal["explicit_inform_slots"][slot]
            self.state["inform_slots"][slot] = goal["explicit_inform_slots"][slot]
        for slot in goal["implicit_inform_slots"].keys():
            self.state["rest_slots"][slot] = "implicit_inform_slots" # Remember where the rest slot comes from.
        for slot in goal["explicit_inform_slots"].keys():
            if slot not in self.state["request_slots"].keys():
                self.state["rest_slots"][slot] = "explicit_inform_slots"
        for slot in goal["request_slots"].keys():
            if slot not in self.state["request_slots"].keys():
                self.state["rest_slots"][slot] = "request_slots"
        user_action = self._assemble_user_action()
        return user_action

    def _init(self):
        """
        used for initializing an instance or an episode.
        :return: Nothing
        """
        self.state = {
            "turn":1,
            "action":None,
            "history":{}, # For slots that have been informed.
            "request_slots":{}, # For slots that user requested in this turn.
            "inform_slots":{}, # For slots that belong to goal["request_slots"] or other slots not in explicit/implicit_inform_slots.
            "explicit_inform_slots":{}, # For slots that belong to goal["explicit_inform_slots"]
            "implicit_inform_slots":{}, # For slots that belong to goal["implicit_inform_slots"]
            "rest_slots":{} # For slots that have not been informed.
        }
        if self.parameter.get("train_mode") is True:
            self.goal = random.choice(self.goal_set["train"])
        else:
            self.goal = random.choice(self.goal_set["test"])
        self.episode_over = False
        self.dialogue_status = dialogue_configuration.NOT_COME_YET
        self.constraint_check = dialogue_configuration.CONSTRAINT_CHECK_FAILURE

    def _assemble_user_action(self):
        user_action = {
            "turn":self.state["turn"],
            "action":self.state["action"],
            "speaker":"user",
            "request_slots":self.state["request_slots"],
            "inform_slots":self.state["inform_slots"],
            "explicit_inform_slots":self.state["explicit_inform_slots"],
            "implicit_inform_slots":self.state["implicit_inform_slots"]
        }
        return user_action

    def next(self, agent_action, turn):
        agent_act_type = agent_action["action"]
        self.state["turn"] = turn
        if self.state["turn"] > self.max_turn:
            self.episode_over = True
            self.state["action"] = dialogue_configuration.CLOSE_DIALOGUE
            self.dialogue_status = dialogue_configuration.DIALOGUE_FAILED
        else:
            pass

        if self.episode_over is not True:
            self.state["history"].update(self.state["inform_slots"])
            self.state["history"].update(self.state["explicit_inform_slots"])
            self.state["history"].update(self.state["implicit_inform_slots"])

            self.state["inform_slots"].clear()
            self.state["explicit_inform_slots"].clear()
            self.state["implicit_inform_slots"].clear()

            # Response according to different action type.
            if agent_act_type == dialogue_configuration.CLOSE_DIALOGUE:
                self._response_closing(agent_action=agent_action)
            elif agent_act_type == "request":
                self._response_request_same(
                    agent_action=agent_action)  # explicit/implicit_inform_slots are handled in the same way.
                # self._response_request_different(agent_action=agent_action) # explicit/implicit_inform_slots are handled differently.
            elif agent_act_type == dialogue_configuration.THANKS:
                self._response_thanks(agent_action=agent_action)
            elif agent_act_type == "confirm_answer":
                self._response_confirm_answer_same(agent_action=agent_action)  # Explicit/implicit_inform_slots are handled in the same way.
                # self._response_confirm_answer_different(agent_action=agent_action) # Explicit/implicit_inform_slots are handled differently.
            elif agent_act_type == "inform":
                self._response_inform_same(agent_action=agent_action)  # Explicit/implicit_inform_slots are handled in the same way.
                # self._response_inform_different(agent_action=agent_action) # Explicit/implicit_inform_slots are handled differently.
            elif agent_act_type == "explicit_inform":
                self._response_inform_same(agent_action=agent_action)  # Explicit/implicit_inform_slots are handled in the same way.
                # self._response_inform_different(agent_action=agent_action) # Explicit/implicit_inform_slots are handled differently.
            elif agent_act_type == "implicit_inform":
                self._response_inform_same(agent_action=agent_action)  # Explicit/implicit_inform_slots are handled in the same way.
                # self._response_inform_different(agent_action=agent_action) # Explicit/implicit_inform_slots are handled differently.
            user_action = self._assemble_user_action()
            return user_action, self.episode_over, self.dialogue_status
        else:
            pass

    def _response_closing(self, agent_action):
        self.state["action"] = dialogue_configuration.THANKS
        self.episode_over = True


    #############################################
    # Response for request where explicit_inform_slots and implicit_slots are handled in the same way.
    ##############################################
    def _response_request_same(self, agent_action):
        """
        The user informs slot must be one of implicit_inform_slots, because the explicit_inform_slots are all informed
        at beginning.
        # It would be easy at first whose job is to answer the implicit slot requested by agent.
        :param agent_action:
        :return:
        """
        # TODO (Qianlong): response to request action.
        if len(agent_action["request_slots"].keys()) > 0:
            for slot in agent_action["request_slots"].keys():
                # The requested slots are come from explicit_inform_slots.
                if slot in self.goal["goal"]["explicit_inform_slots"].keys():
                    self.state["action"] = "inform"
                    self.state["inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                    self.state["action"] = "inform"
                    self.state["inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                else:
                    if len(self.state["request_slots"].keys()) == 0 and len(self.state["rest_slots"].keys()) == 0:
                        self.state["action"] = dialogue_configuration.THANKS
                    else:
                        self.state["action"] = "not_sure"
                        self.state["inform_slots"][slot] = dialogue_configuration.I_DO_NOT_KNOW

        # The case where the agent action type is request, but nothing in agent request_slots, which should not appear.
        # A randomized slot will be chosen to inform agent if the rest_slots is not empty.
        else:
            if len(self.state["rest_slots"].keys()) > 0:
                slot = random.choice(self.state["rest_slots"].keys())
                if slot in self.goal["goal"]["explicit_inform_slots"].keys():# The case should not appear.
                    self.state["action"] = "inform"
                    self.state["inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                    self.state["action"] = "inform"
                    self.state["inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                else:
                    self.state["action"] = "not_sure"
                    self.state["inform_slots"] = dialogue_configuration.I_DO_NOT_KNOW
            else:
                self.state["action"] = dialogue_configuration.THANKS

    #############################################
    # Response for request where explicit_inform_slots and implicit_inform_slots are handled differently.
    #############################################
    def _response_request_different(self, agent_action):
        """
        The user informs slot must be one of implicit_inform_slots, because the explicit_inform_slots are all informed
        at beginning.
        # It would be easy at first whose job is to answer the implicit slot requested by agent.
        :param agent_action:
        :return:
        """
        # TODO (Qianlong): response to request action.
        if len(agent_action["request_slots"].keys()) > 0:
            for slot in agent_action["request_slots"].keys():
                # The requested slots are
                if slot in self.goal["goal"]["explicit_inform_slots"].keys():
                    self.state["action"] = "explicit_inform"
                    self.state["explicit_inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                    self.state["action"] = "implicit_inform"
                    self.state["implicit_inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                else:
                    if len(self.state["request_slots"].keys()) == 0 and len(self.state["rest_slots"].keys()) == 0:
                        self.state["action"] = dialogue_configuration.THANKS
                    else:
                        self.state["action"] = "not_sure"
                        self.state["implicit_inform_slots"][slot] = dialogue_configuration.I_DO_NOT_KNOW

        # The case where the agent action type is request, but nothing in agent request_slots, which should not appear.
        # A randomized slot will be chosen to inform if the rest_slots is not empty.
        else:
            if len(self.state["rest_slots"].keys()) > 0:
                slot = random.choice(self.state["rest_slots"].keys())
                if slot in self.goal["goal"]["explicit_inform_slots"].keys():  # The case should not appear.
                    self.state["action"] = "explicit_inform"
                    self.state["explicit_inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                    self.state["action"] = "implicit_inform"
                    self.state["implicit_inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                else:
                    self.state["action"] = "not_sure"
                    self.state["inform_slots"] = dialogue_configuration.I_DO_NOT_KNOW
            else:
                self.state["action"] = dialogue_configuration.THANKS

    #############################################
    # This may be useful in future. It will be easy at first.
    #############################################
    def _response_request3(self, agent_action):
        """
        对agent的request做出回复，首先利用agent里面的request slot在user goal里面的inform_slots进行寻找答案，能找到的就返回，不能找
        的就设置为不知道，
        :param agent_action:
        :return:
        """
        # TODO (Qianlong): response to request action.
        if len(agent_action["request_slots"].keys()) > 0: # Agent requested some slots in the user goal request_slots.
            slot = agent_action["request_slots"].keys()[0]

            # The case where the requested slot is in the user explicit inform slots.
            if slot in self.goal["goal"]["explicit_inform_slots"]:
                self.state["action"] = "explicit_inform"
                self.state["explicit_inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                if slot in self.state["request_slots"].keys(): self.state["request_slots"].pop(slot)

            # The case where the requested slot is in the user implicit inform slots.
            elif slot in self.goal["goal"]["implicit_inform_slots"]:
                self.state["action"] = "implicit_inform"
                self.state["implicit_inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                if slot in self.state["request_slots"].keys(): self.state["request_slots"].pop(slot)

            # The case where the requested slot in the user goal request_slots and has been answered.
            elif slot in self.goal["goal"]["request_slots"].keys() and slot not in self.state["rest_slots"].keys() and slot in self.state["history"].keys():
                self.state["action"] = "inform"
                self.state["request_slots"].clear()
                self.state["inform_slots"][slot] = self.state["history"][slot]

            # The case where the requested slot in the user goal request_slots, but not be answered yet.
            elif slot in self.goal["goal"]["request_slots"].keys() and slot in self.state["rest_slots"].keys():
                self.state["action"] = "request"# Confirm question.
                self.state["request_slots"][slot] = dialogue_configuration.VALUE_UNKNOWN

                ########################################################################################
                # Inform the rest slots in explicit/implicit_inform_slots, the implicit slots will be informed only when
                # all explicit slots have been informed.
                ########################################################################################
                for slot in self.state["rest_slots"].keys():
                    if slot in self.goal["goal"]["explicit_inform_slots"].keys():
                        self.state["explicit_inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                    elif slot in self.goal["goal"]["implicit_inform_slots"].keys() and \
                        len(set(self.goal["goal"]["explicit_inform_slots"].keys()) & set(self.state["history"].keys()) \
                            - set(self.goal["goal"]["explicit_inform_slots"].keys())) == 0:
                        self.state["implicit_inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]

                # Remove slots in the sate rest_slots which have been informed in this turn.
                for slot in self.state["explicit_inform_slots"].keys():
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)
                for slot in self.state["implicit_inform_slots"].keys():
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)

            # The case where the requested slot neither in goal request_slots nor in explicit/implicit goal inform slots
            else:
                if len(self.state["request_slots"].keys()) == 0 and len(self.state["rest_slots"].keys()) == 0:
                    self.state["action"] = dialogue_configuration.THANKS
                else:
                    self.state["action"] = "inform"
                self.state["inform_slots"][slot] = dialogue_configuration.I_DO_NOT_CARE

        # The case where the agent action type is request, but nothing in agent request_slots, which should not appear.
        else:
            if len(self.state["rest_slots"]) > 0:
                slot = random.choice(self.state["rest_slots"].keys())
                if slot in self.goal["goal"]["explicit_inform_slots"].keys():
                    self.state["action"] = "explicit_inform"
                    self.state["explicit_inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                    self.state["rest_slots"].pop(slot)
                elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                    self.state["action"] = "implicit_inform"
                    self.state["implicit_inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                    self.state["rest_slots"].pop(slot)
                elif slot in self.goal["goal"]["request_slots"].keys():
                    self.state["action"] = "request"
                    self.state["request_slots"][slot] = self.goal["goal"]["request_slots"][slot]
                else:# this should no appear
                    self.state["action"] = "inform"
                    self.state["inform_slots"][slot] = dialogue_configuration.I_DO_NOT_CARE
            else:
                self.state["action"] = dialogue_configuration.THANKS

    #############################################
    # Response confirm_answer where explicit_inform_slots and implicit_inform_slots are handled in the same way.
    #############################################
    def _response_confirm_answer_same(self, agent_action):
        # TODO (Qianlong): response to confirm answer action. I don't think it is right.
        if len(self.state["rest_slots"].keys()) > 0:
            slot = random.choice(list(self.state["rest_slots"].keys()))
            if slot in self.goal["goal"]["request_slots"].keys():
                self.state["action"] = "request"
                self.state["request_slots"][slot] = dialogue_configuration.VALUE_UNKNOWN
            elif slot in self.goal["goal"]["explicit_inform_slots"].keys():
                self.state["action"] = "inform"
                self.state["inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                self.state["rest_slots"].pop(slot)
            elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                self.state["action"] = "inform"
                self.state["inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                self.state["rest_slots"].pop(slot)
        else:
            self.state["action"] = dialogue_configuration.THANKS

    #############################################
    # Response confirm_answer where explicit_inform_slots and implicit_inform_slots are handled differently.
    #############################################
    def _response_confirm_answer_different(self, agent_action):
        # TODO (Qianlong): response to confirm answer action. I don't think it is right.
        if len(self.state["rest_slots"].keys()) > 0:
            slot = random.choice(self.state["rest_slots"].keys())
            if slot in self.goal["goal"]["request_slots"].keys():
                self.state["action"] = "request"
                self.state["request_slots"][slot] = dialogue_configuration.VALUE_UNKNOWN
            elif slot in self.state["explicit_inform_slots"].keys():
                self.state["action"] = "explicit_inform"
                self.state["explicit_inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                self.state["rest_slots"].pop(slot)
            elif slot in self.state["implicit_inform_slots"].keys():
                self.state["action"] = "implicit_inform"
                self.state["implicit_inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                self.state["rest_slots"].pop(slot)
        else:
            self.state["action"] = dialogue_configuration.THANKS

    ##########################################
    # Response for thanks.
    ##########################################
    def _response_thanks(self, agent_action):
        # TODO (Qianlong): response to thanks action.
        self.episode_over = True
        self.dialogue_status = dialogue_configuration.DIALOGUE_SUCCESS

        request_slot_set = copy.deepcopy(list(self.state["request_slots"].keys()))
        if "disease" in request_slot_set:
            request_slot_set.remove("disease")
        rest_slot_set = copy.deepcopy(list(self.state["rest_slots"].keys()))
        if "disease" in rest_slot_set:
            rest_slot_set.remove("disease")

        if len(request_slot_set) > 0 or len(rest_slot_set) > 0:
            self.dialogue_status = dialogue_configuration.DIALOGUE_FAILED

        for slot in self.state["history"].keys():
            if slot in self.goal["goal"]["explicit_inform_slots"].keys() and \
                self.state["history"][slot] != self.goal["goal"]["explicit_inform_slots"][slot]:
                self.dialogue_status = dialogue_configuration.DIALOGUE_FAILED
            elif slot in self.goal["goal"]["implicit_inform_slots"].keys() and \
                self.state["history"][slot] != self.goal["goal"]["implicit_inform_slots"][slot]:
                self.dialogue_status = dialogue_configuration.DIALOGUE_FAILED
        if "disease" in agent_action["inform_slots"].keys():
            if agent_action["inform_slots"]["disease"] == dialogue_configuration.VALUE_NO_MATCH:
                self.dialogue_status = dialogue_configuration.DIALOGUE_FAILED
        if self.constraint_check == dialogue_configuration.CONSTRAINT_CHECK_FAILURE:
            self.dialogue_status = dialogue_configuration.DIALOGUE_FAILED

    ##########################################
    # Response for inform where explicit_inform_slots and implicit_inform_slots are handled in the same way.
    ##########################################
    def _response_inform_same(self, agent_action):
        # TODO (Qianlong): response to inform action.
        agent_all_inform_slots = copy.deepcopy(agent_action["inform_slots"])
        agent_all_inform_slots.update(agent_action["explicit_inform_slots"])
        agent_all_inform_slots.update(agent_action["implicit_inform_slots"])

        user_all_inform_slots = copy.deepcopy(self.goal["goal"]["explicit_inform_slots"])
        user_all_inform_slots.update(self.goal["goal"]["implicit_inform_slots"])

        if "taskcomplete" in agent_action["inform_slots"].keys(): # check all the constraints from agents with user goal
            self.state["action"] = dialogue_configuration.THANKS
            self.constraint_check = dialogue_configuration.CONSTRAINT_CHECK_SUCCESS
            if agent_action["inform_slots"]["taskcomplete"] == dialogue_configuration.VALUE_NO_MATCH:
                self.state["history"]["disease"] = dialogue_configuration.VALUE_NO_MATCH
                if "disease" in self.state["rest_slots"].keys(): self.state["rest_slots"].pop("disease")
                if "disease" in self.state["request_slots"].keys(): self.state["request_slots"].pop("disease")

            #  Deny, if the answers from agent can not meet the constraints of user
            for slot in user_all_inform_slots.keys():
                if slot not in agent_all_inform_slots or agent_all_inform_slots[slot] != user_all_inform_slots[slot]:
                    self.state["action"] = "deny"
                    # TODO (Qianlong): don't know why this should be cleared.
                    self.state["request_slots"].clear()
                    self.state["inform_slots"].clear()
                    self.state["explicit_inform_slots"].clear()
                    self.state["implicit_inform_slots"].clear()
                    self.constraint_check = dialogue_configuration.CONSTRAINT_CHECK_FAILURE
                    break
        # The agent informed the right disease and dialogue is over.
        elif "disease" in agent_action["inform_slots"].keys() and agent_action["inform_slots"]["disease"] == self.goal["disease_tag"]:
            self.state["action"] = dialogue_configuration.CLOSE_DIALOGUE
            self.dialogue_status = dialogue_configuration.DIALOGUE_SUCCESS
            self.episode_over = True
            self.state["inform_slots"].clear()
            self.state["explicit_inform_slots"].clear()
            self.state["implicit_inform_slots"].clear()
            if "disease" in self.state["rest_slots"]: self.state["rest_slots"].pop("disease")
        # The agent informed wrong disease and the dialogue will go on if not reach the max_turn.
        elif "disease" in agent_action["inform_slots"].keys() and agent_action["inform_slots"]["disease"] != self.goal["disease_tag"]:
            self.state["action"] = "deny"
            self.dialogue_status = dialogue_configuration.INFORM_WRONG_DISEASE

        else: # Task is not completed.
            for slot in agent_all_inform_slots.keys():
                self.state["history"][slot] = agent_all_inform_slots[slot]

                # The slot comes from explicit/implicit_inform_slots of user.
                if slot in user_all_inform_slots.keys():
                    if agent_all_inform_slots[slot] == user_all_inform_slots[slot]:
                        if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)

                        if len(self.state["request_slots"].keys()) > 0:
                            self.state["action"] = "request"
                        elif len(self.state["rest_slots"]) > 0:# The state["rest_slots"] is not empty.
                            rest_slot_set = copy.deepcopy(list(self.state['rest_slots'].keys()))
                            if "disease" in rest_slot_set:
                                rest_slot_set.remove("disease")

                            if len(rest_slot_set) > 0:
                                inform_slot = random.choice(rest_slot_set)
                                if inform_slot in self.goal["goal"]["explicit_inform_slots"].keys():
                                    self.state["inform_slots"][inform_slot] = self.goal["goal"]["explicit_inform_slots"][inform_slot]
                                    self.state["action"] = "inform"
                                    self.state["rest_slots"].pop(inform_slot)
                                elif inform_slot in self.goal["goal"]["implicit_inform_slots"].keys():
                                    self.state["inform_slots"][inform_slot] = self.goal["goal"]["implicit_inform_slots"][inform_slot]
                                    self.state["action"] = "inform"
                                    self.state["rest_slots"].pop(inform_slot)
                                elif inform_slot in self.goal["goal"]["request_slots"].keys():# This case will not appear
                                    self.state["request_slots"][inform_slot] = dialogue_configuration.VALUE_UNKNOWN
                                    self.state["action"] = "request"
                                    self.state["rest_slots"].pop(inform_slot)
                            else:
                                self.state["request_slots"]["disease"] = dialogue_configuration.VALUE_UNKNOWN
                                self.state["action"] = "request"
                    else: # != value  Should we deny here or ?
                        ########################################################################
                        # TODO When agent informs(slot=value), where the value is different with the constraint in user goal, Should we deny or just inform the correct value?
                        ########################################################################
                        if slot in self.goal["goal"]["explicit_inform_slots"].keys():
                            self.state["action"] = "inform"
                            self.state["inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                        elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                            self.state["action"] = "inform"
                            self.state["inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                        if slot in self.state["rest_slots"]: self.state["rest_slots"].pop(slot)

                else:
                    if slot in self.state["request_slots"].keys(): self.state["request_slots"].pop(slot)
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)

                    if len(self.state["request_slots"]) > 0:
                        request_slot_set = list(self.state["request_slots"].keys())
                        if "disease" in request_slot_set:
                            request_slot_set.remove("disease")

                        if len(request_slot_set) > 0:
                            request_slot = random.choice(request_slot_set)
                        else:
                            request_slot = "disease"

                        self.state["request_slots"][request_slot] = dialogue_configuration.VALUE_UNKNOWN
                        self.state["action"] = "request"
                    elif len(self.state["rest_slots"].keys()) > 0:
                        rest_slot_set = list(self.state["rest_slots"].keys())
                        if "disease" in rest_slot_set: rest_slot_set.remove("disease")
                        if len(rest_slot_set) > 0:
                            inform_slot = random.choice(rest_slot_set)
                            if inform_slot in self.goal["goal"]["explicit_inform_slots"].keys():
                                self.state["inform_slots"][inform_slot] = self.goal["goal"]["explicit_inform_slots"][inform_slot]
                                self.state["action"] = "inform"
                                self.state["rest_slots"].pop(inform_slot)
                                if "disease" in self.state["rest_slots"].keys():
                                    self.state["request_slots"]["disease"] = dialogue_configuration.VALUE_UNKNOWN
                                    self.state["action"] = "request"
                            elif inform_slot in self.goal["goal"]["implicit_inform_slots"].keys():
                                self.state["inform_slots"][inform_slot] = self.goal["goal"]["implicit_inform_slots"][inform_slot]
                                self.state["action"] = "inform"
                                self.state["rest_slots"].pop(inform_slot)
                                if "disease" in self.state["rest_slots"].keys():
                                    self.state["request_slots"]["disease"] = dialogue_configuration.VALUE_UNKNOWN
                                    self.state["action"] = "request"
                            elif inform_slot in self.goal["goal"]["request_slots"].keys():  # This case will not appear
                                self.state["request_slots"][inform_slot] = dialogue_configuration.VALUE_UNKNOWN
                                self.state["action"] = "request"
                        else:
                                self.state["request_slots"]["disease"] = dialogue_configuration.VALUE_UNKNOWN
                                self.state["action"] = "request"
                    else:
                        self.state["action"] = dialogue_configuration.THANKS

    ##########################################
    # Response for inform where explicit_inform_slots and implicit_inform_slots are handled differently.
    ##########################################
    def _response_inform_different(self, agent_action):
        # TODO (Qianlong): response to inform action.
        agent_all_inform_slots = copy.deepcopy(agent_action["inform_slots"])
        agent_all_inform_slots.update(agent_action["explicit_inform_slots"])
        agent_all_inform_slots.update(agent_action["implicit_inform_slots"])

        user_all_inform_slots = copy.deepcopy(self.goal["goal"]["explicit_inform_slots"])
        user_all_inform_slots.update(self.goal["goal"]["implicit_inform_slots"])

        if "taskcomplete" in agent_action["inform_slots"].keys(): # check all the constraints from agents with user goal
            self.state["action"] = dialogue_configuration.THANKS
            self.constraint_check = dialogue_configuration.CONSTRAINT_CHECK_SUCCESS
            if agent_action["inform_slots"]["taskcomplete"] == dialogue_configuration.VALUE_NO_MATCH:
                self.state["history"]["disease"] = dialogue_configuration.VALUE_NO_MATCH
                if "disease" in self.state["rest_slots"].keys(): self.state["rest_slots"].pop("disease")
                if "disease" in self.state["request_slots"].keys(): self.state["request_slots"].pop("disease")

            #  Deny, if the answers from agent can not meet the constraints of user
            for slot in user_all_inform_slots.keys():
                if slot not in agent_all_inform_slots or agent_all_inform_slots[slot] != user_all_inform_slots[slot]:
                    self.state["action"] = "deny"
                    # TODO (Qianlong): don't know why this should be cleared.
                    self.state["request_slots"].clear()
                    self.state["inform_slots"].clear()
                    self.state["explicit_inform_slots"].clear()
                    self.state["implicit_inform_slots"].clear()
                    self.constraint_check = dialogue_configuration.CONSTRAINT_CHECK_FAILURE
                    break
        # The agent informed the right disease and dialogue is over.
        elif "disease" in agent_action["inform_slots"].keys() and agent_action["inform_slots"]["disease"] == self.goal["disease_tag"]:
            self.state["action"] = dialogue_configuration.CLOSE_DIALOGUE
            self.dialogue_status = dialogue_configuration.DIALOGUE_SUCCESS
            self.episode_over = True
            self.state["inform_slots"].clear()
            self.state["explicit_inform_slots"].clear()
            self.state["implicit_inform_slots"].clear()
            if "disease" in self.state["rest_slots"]: self.state["rest_slots"].pop("disease")
        # The agent informed wrong disease and the dialogue will go on if not reach the max_turn.
        elif "disease" in agent_action["inform_slots"].keys() and agent_action["inform_slots"]["disease"] != self.goal["disease_tag"]:
            self.state["action"] = "deny"
            self.dialogue_status = dialogue_configuration.INFORM_WRONG_DISEASE

        else:
            for slot in agent_all_inform_slots.keys():
                self.state["history"][slot] = agent_all_inform_slots[slot]

                # The slot comes from explicit/implicit_inform_slots of user.
                if slot in user_all_inform_slots.keys():
                    if agent_all_inform_slots[slot] == user_all_inform_slots[slot]:
                        if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)

                        if len(self.state["request_slots"].keys()) > 0:
                            self.state["action"] = "request"
                        elif len(self.state["rest_slots"]) > 0:# The state["rest_slots"] is not empty.
                            rest_slot_set = copy.deepcopy(list(self.state['rest_slots'].keys()))
                            if "disease" in rest_slot_set:
                                rest_slot_set.remove("disease")

                            if len(rest_slot_set) > 0:
                                inform_slot = random.choice(rest_slot_set)
                                if inform_slot in self.goal["goal"]["explicit_inform_slots"].keys():
                                    self.state["explicit_inform_slots"][inform_slot] = self.goal["goal"]["explicit_inform_slots"][inform_slot]
                                    self.state["action"] = "explicit_inform"
                                    self.state["rest_slots"].pop(inform_slot)
                                elif inform_slot in self.goal["goal"]["implicit_inform_slots"].keys():
                                    self.state["implicit_inform_slots"][inform_slot] = self.goal["goal"]["implicit_inform_slots"][inform_slot]
                                    self.state["action"] = "implicit_inform"
                                    self.state["rest_slots"].pop(inform_slot)
                                elif inform_slot in self.goal["goal"]["request_slots"].keys():# This case will not appear
                                    self.state["request_slots"][inform_slot] = dialogue_configuration.VALUE_UNKNOWN
                                    self.state["action"] = "request"
                                    self.state["rest_slots"].pop(inform_slot)
                            else:
                                self.state["request_slots"]["disease"] = dialogue_configuration.VALUE_UNKNOWN
                                self.state["action"] = "request"
                    else: # != value  Should we deny here or ?
                        ########################################################################
                        # TODO When agent informs(slot=value), where the value is different with the constraint in user goal, Should we deny or just inform the correct value?
                        ########################################################################
                        if slot in self.goal["goal"]["explicit_inform_slots"].keys():
                            self.state["action"] = "explicit_inform"
                            self.state["explicit_inform_slots"][slot] = self.goal["goal"]["explicit_inform_slots"][slot]
                        elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                            self.state["action"] = "implicit_inform"
                            self.state["implicit_inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot]
                        if slot in self.state["rest_slots"]: self.state["rest_slots"].pop(slot)

                else:
                    if slot in self.state["request_slots"].keys(): self.state["request_slots"].pop(slot)
                    if slot in self.state["rest_slots"].keys(): self.state["rest_slots"].pop(slot)

                    if len(self.state["request_slots"]) > 0:
                        request_slot_set = list(self.state["request_slots"].keys())
                        if "disease" in request_slot_set:
                            request_slot_set.remove("disease")

                        if len(request_slot_set) > 0:
                            request_slot = random.choice(request_slot_set)
                        else:
                            request_slot = "disease"

                        self.state["request_slots"][request_slot] = dialogue_configuration.VALUE_UNKNOWN
                        self.state["action"] = "request"
                    elif len(self.state["rest_slots"].keys()) > 0:
                        rest_slot_set = list(self.state["rest_slots"].keys())
                        if "disease" in rest_slot_set: rest_slot_set.remove("disease")
                        if len(rest_slot_set) > 0:
                            inform_slot = random.choice(rest_slot_set)
                            if inform_slot in self.goal["goal"]["explicit_inform_slots"].keys():
                                self.state["explicit_inform_slots"][inform_slot] = self.goal["goal"]["explicit_inform_slots"][inform_slot]
                                self.state["action"] = "explicit_inform"
                                self.state["rest_slots"].pop(inform_slot)
                                if "disease" in self.state["rest_slots"].keys():
                                    self.state["request_slots"]["disease"] = dialogue_configuration.VALUE_UNKNOWN
                                    self.state["action"] = "request"
                            elif inform_slot in self.goal["goal"]["implicit_inform_slots"].keys():
                                self.state["implicit_inform_slots"][inform_slot] = self.goal["goal"]["implicit_inform_slots"][inform_slot]
                                self.state["action"] = "implicit_inform"
                                self.state["rest_slots"].pop(inform_slot)
                                if "disease" in self.state["rest_slots"].keys():
                                    self.state["request_slots"]["disease"] = dialogue_configuration.VALUE_UNKNOWN
                                    self.state["action"] = "request"
                            elif inform_slot in self.goal["goal"]["request_slots"].keys():  # This case will not appear
                                self.state["request_slots"][inform_slot] = dialogue_configuration.VALUE_UNKNOWN
                                self.state["action"] = "request"
                        else:
                                self.state["request_slots"]["disease"] = dialogue_configuration.VALUE_UNKNOWN
                                self.state["action"] = "request"
                    else:
                        self.state["action"] = dialogue_configuration.THANKS