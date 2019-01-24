# -*- coding:utf-8 -*-
"""
State tracker of the dialogue system, which tracks the state of the dialogue during interaction.
"""

import sys, os
import copy
import copy
sys.path.append(os.getcwd().replace("src/dialogue_system/state_tracker", ""))

from src.dialogue_system import dialogue_configuration


class StateTracker(object):
    def __init__(self, user, agent,parameter):
        self.user = user
        self.agent = agent
        self._init()

    def get_state(self):
        return copy.deepcopy(self.state)

    def state_updater(self, user_action=None, agent_action=None):
        assert (user_action is None or agent_action is None), "user action and agent action cannot be None at the same time."
        self.state["turn"] = self.turn
        if user_action is not None:
            self._state_update_with_user_acion(user_action=user_action)
        elif agent_action is not None:
            self._state_update_with_agent_action(agent_action=agent_action)
        self.turn += 1

    def initialize(self):
        self._init()

    def _init(self):
        self.turn = 0
        self.state = {
            "agent_action":None,
            "user_action":None,
            "turn":self.turn,
            "current_slots":{
                "user_request_slots":{},
                "agent_request_slots":{},
                "inform_slots":{},
                "explicit_inform_slots":{},
                "implicit_inform_slots":{},
                "proposed_slots":{},
                "wrong_diseases":[]
            },
            "history":[]
        }

    def set_agent(self, agent):
        self.agent = agent

    def _state_update_with_user_acion(self, user_action):
        # Updating dialog state with user_action.
        self.state["user_action"] = user_action
        temp_action = copy.deepcopy(user_action)
        temp_action["current_slots"] = copy.deepcopy(self.state["current_slots"])# Save current_slots for every turn.
        self.state["history"].append(temp_action)
        for slot in user_action["request_slots"].keys():
            self.state["current_slots"]["user_request_slots"][slot] = user_action["request_slots"][slot]

        # Inform_slots.
        inform_slots = list(user_action["inform_slots"].keys())
        if "disease" in inform_slots and user_action["action"] == "deny":
            if user_action["inform_slots"]["disease"] not in self.state["current_slots"]["wrong_diseases"]:
                self.state["current_slots"]["wrong_diseases"].append(user_action["inform_slots"]["disease"])
        if "disease" in inform_slots: inform_slots.remove("disease")
        for slot in inform_slots:
            if slot in self.user.goal["goal"]["request_slots"].keys():
                self.state["current_slots"]["proposed_slots"][slot] = user_action["inform_slots"][slot]
            else:
                self.state["current_slots"]['inform_slots'][slot] = user_action["inform_slots"][slot]
            if slot in self.state["current_slots"]["agent_request_slots"].keys():
                self.state["current_slots"]["agent_request_slots"].pop(slot)

        # TODO (Qianlong): explicit_inform_slots and implicit_inform_slots are handled differently.
        # Explicit_inform_slots.
        explicit_inform_slots = list(user_action["explicit_inform_slots"].keys())
        if "disease" in explicit_inform_slots and user_action["action"] == "deny":
            if user_action["inform_slots"]["disease"] not in self.state["current_slots"]["wrong_diseases"]:
                self.state["current_slots"]["wrong_diseases"].append(user_action["explicit_inform_slots"]["disease"])
        if "disease" in explicit_inform_slots: explicit_inform_slots.remove("disease")
        for slot in explicit_inform_slots:
            if slot in self.user.goal["goal"]["request_slots"].keys():
                self.state["current_slots"]["proposed_slots"][slot] = user_action["explicit_inform_slots"][slot]
            else:
                self.state["current_slots"]["explicit_inform_slots"][slot] = user_action["explicit_inform_slots"][slot]
            if slot in self.state["current_slots"]["agent_request_slots"].keys():
                self.state["current_slots"]["agent_request_slots"].pop(slot)
        # Implicit_inform_slots.
        implicit_inform_slots = list(user_action["implicit_inform_slots"].keys())
        if "disease" in implicit_inform_slots and user_action["action"] == "deny":
            if user_action["inform_slots"]["disease"] not in self.state["current_slots"]["wrong_diseases"]:
                self.state["current_slots"]["wrong_diseases"].append(user_action["implicit_inform_slots"]["disease"])
        if "disease" in implicit_inform_slots: implicit_inform_slots.remove("disease")
        for slot in implicit_inform_slots:
            if slot in self.user.goal["goal"]["request_slots"].keys():
                self.state["current_slots"]["proposed_slots"][slot] = user_action["implicit_inform_slots"][slot]
            else:
                self.state["current_slots"]["implicit_inform_slots"][slot] = user_action["implicit_inform_slots"][slot]
            if slot in self.state["current_slots"]["agent_request_slots"].keys():
                self.state["current_slots"]["agent_request_slots"].pop(slot)

    def _state_update_with_agent_action(self, agent_action):
        # Updating dialog state with agent_action.

        explicit_implicit_slot_value = copy.deepcopy(self.user.goal["goal"]["explicit_inform_slots"])
        explicit_implicit_slot_value.update(self.user.goal["goal"]["implicit_inform_slots"])

        self.state["agent_action"] = agent_action
        temp_action = copy.deepcopy(agent_action)
        temp_action["current_slots"] = copy.deepcopy(self.state["current_slots"])# save current_slots for every turn.
        self.state["history"].append(temp_action)
        # import json
        # print(json.dumps(agent_action, indent=2))
        for slot in agent_action["request_slots"].keys():
            self.state["current_slots"]["agent_request_slots"][slot] = agent_action["request_slots"][slot]

        # Inform slots.
        for slot in agent_action["inform_slots"].keys():
            # The slot is come from user's goal["request_slots"]
            slot_value = agent_action["inform_slots"][slot]
            if slot in self.user.goal["goal"]["request_slots"].keys() and slot_value == self.user.goal["disease_tag"]:
                self.state["current_slots"]["proposed_slots"][slot] = agent_action["inform_slots"][slot]
            elif slot in explicit_implicit_slot_value.keys() and slot_value == explicit_implicit_slot_value[slot]:
                self.state["current_slots"]["inform_slots"][slot] = agent_action["inform_slots"][slot]
            # Remove the slot if it is in current_slots["user_request_slots"]
            if slot in self.state["current_slots"]["user_request_slots"].keys():
                self.state["current_slots"]["user_request_slots"].pop(slot)

        # TODO (Qianlong): explicit_inform_slots and implicit_inform_slots are handled differently.
        # Explicit_inform_slots.
        for slot in agent_action["explicit_inform_slots"].keys():
            # The slot is come from user's goal["request_slots"]
            slot_value = agent_action["explicit_inform_slots"][slot]
            if slot in self.user.goal["goal"]["request_slots"].keys() and slot_value == self.user.goal["disease_tag"]:
                self.state["current_slots"]["proposed_slots"][slot] = agent_action["explicit_inform_slots"][slot]
            elif slot in explicit_implicit_slot_value.keys() and slot_value == explicit_implicit_slot_value[slot]:
                self.state["current_slots"]["explicit_inform_slots"][slot] = agent_action["explicit_inform_slots"][slot]
            # Remove the slot if it is in current_slots["user_request_slots"]
            if slot in self.state["current_slots"]["user_request_slots"].keys():
                self.state["current_slots"]["user_request_slots"].pop(slot)

        # Implicit_inform_slots.
        for slot in agent_action["implicit_inform_slots"].keys():
            # The slot is come from user's goal["request_slots"]
            slot_value = agent_action["implicit_inform_slots"][slot]
            if slot in self.user.goal["goal"]["request_slots"].keys() and slot_value == self.user.goal["disease_tag"]:
                self.state["current_slots"]["proposed_slots"][slot] = agent_action["implicit_inform_slots"][slot]
            elif slot in explicit_implicit_slot_value.keys() and slot_value == explicit_implicit_slot_value[slot]:
                self.state["current_slots"]["implicit_inform_slots"][slot] = agent_action["implicit_inform_slots"][slot]
            # Remove the slot if it is in current_slots["user_request_slots"]
            if slot in self.state["current_slots"]["user_request_slots"].keys():
                self.state["current_slots"]["user_request_slots"].pop(slot)