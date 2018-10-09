# -*- coding: utf-8 -*-
"""
Rule-based agent.
"""

import copy
import random
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))

from src.dialogue_system.agent import Agent
from src.dialogue_system import dialogue_configuration


class AgentRule(Agent):
    """
    Rule-based agent.
    """
    def __init__(self,action_set, slot_set, disease_symptom, parameter):
        super(AgentRule,self).__init__(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)

    def next(self, state, turn, greedy_strategy):
        candidate_disease_symptoms = self._get_candidate_disease_symptoms(state=state)
        disease = candidate_disease_symptoms["disease"]
        candidate_symptoms = candidate_disease_symptoms["candidate_symptoms"]

        self.agent_action["request_slots"].clear()
        self.agent_action["explicit_inform_slots"].clear()
        self.agent_action["implicit_inform_slots"].clear()
        self.agent_action["inform_slots"].clear()
        self.agent_action["turn"] = turn

        if len(candidate_symptoms) == 0:
            self.agent_action["action"] = "inform"
            self.agent_action["inform_slots"]["disease"] = disease
        else:
            symptom = random.choice(candidate_symptoms)
            self.agent_action["action"] = "request"
            self.agent_action["request_slots"].clear()
            self.agent_action["request_slots"][symptom] = dialogue_configuration.VALUE_UNKNOWN
        agent_action = copy.deepcopy(self.agent_action)
        agent_action.pop("turn")
        agent_action.pop("speaker")
        agent_index = self.action_sapce.index(agent_action)
        return self.agent_action, agent_index

    def _get_candidate_disease_symptoms(self, state):
        """
        Comparing state["current_slots"] with disease_symptom to identify which disease the user may have.
        :param state: a dict, the current dialogue state gotten from dialogue state tracker..
        :return: a list of candidate symptoms.
        """
        inform_slots = state["current_slots"]["inform_slots"]
        inform_slots.update(state["current_slots"]["explicit_inform_slots"])
        inform_slots.update(state["current_slots"]["implicit_inform_slots"])
        wrong_diseases = state["current_slots"]["wrong_diseases"]

        # Calculate number of informed symptom for each disease.
        disease_match_number = {}
        for disease in self.disease_symptom.keys():
            disease_match_number[disease] = {}
            disease_match_number[disease]["yes"] = 0
            disease_match_number[disease]["not_sure"] = 0
            disease_match_number[disease]["deny"] = 0

        for slot in inform_slots.keys():
            for disease in disease_match_number.keys():
                if inform_slots[slot] in self.disease_symptom[disease]["symptom"] and inform_slots[slot] == True:
                    disease_match_number[disease]["yes"] += 1
                elif inform_slots[slot] in self.disease_symptom[disease]["symptom"] and inform_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                    disease_match_number[disease]["not_sure"] += 1
                elif inform_slots[slot] in self.disease_symptom[disease]["symptom"] and inform_slots[slot] == dialogue_configuration.I_DENY:
                    disease_match_number[disease]["deny"] += 1

        # Get the ratio of informed symptom number to the number of symptoms of each disease.
        disease_score = {}
        for disease in disease_match_number.keys():
            yes_score = float(disease_match_number[disease]["yes"]) / len(self.disease_symptom[disease]["symptom"])
            not_sure_score = float(disease_match_number[disease]["not_sure"]) / len(self.disease_symptom[disease]["symptom"])
            deny_score = float(disease_match_number[disease]["deny"]) / len(self.disease_symptom[disease]["symptom"])
            disease_score[disease] = yes_score - 0.5*not_sure_score - deny_score

        # Get the most probable disease that has not been wrongly informed
        sorted_diseases = sorted(disease_score.items(), key=lambda d: d[1], reverse=True)
        for disease in sorted_diseases:
            if disease[0] not in wrong_diseases:
                match_disease = disease[0]
                break
        # match_disease = max(disease_score.items(), key=lambda x: x[1])[0] # Get the most probable disease that the user have.
        # Candidate symptom list of symptoms that belong to the most probable disease but have't been informed yet.
        candidate_symptoms = []
        for symptom in self.disease_symptom[match_disease]["symptom"]:
            if symptom not in inform_slots.keys():
                candidate_symptoms.append(symptom)
        return {"disease":match_disease,"candidate_symptoms":candidate_symptoms}