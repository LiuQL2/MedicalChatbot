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
    def __init__(self,action_set, slot_set, disease_symptom, parameter):
        super(AgentRule,self).__init__(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)

    def next(self, state, turn):
        candidate_disease_symptoms = self._get_candidate_disease_symptoms(state=state)
        disease = candidate_disease_symptoms["disease"]
        candidate_symptoms = candidate_disease_symptoms["candidate_symptoms"]
        if len(candidate_symptoms) == 0:
            self.agent_action["action"] = "inform"
            self.agent_action["inform_slots"]["disease"] = disease
        else:
            symptom = random.choice(candidate_symptoms)
            self.agent_action["action"] = "request"
            self.agent_action["request_slots"][symptom] = dialogue_configuration.VALUE_UNKNOWN

        return self.agent_action

    def _get_candidate_disease_symptoms(self, state):
        """
        Comparing state["current_slots"] with disease_symptom to identify which disease the user possibly have.
        :param state: Dialogue state defined in state_tracker.
        :return: Candidate symptoms list.
        """
        inform_slots = state["current_slots"]["inform_slots"]
        inform_slots.update(state["current_slots"]["explicit_inform_slots"])
        inform_slots.update(state["current_slots"]["implicit_inform_slots"])

        # Calculate number of informed symptom for each disease.
        disease_match_number = {}
        for disease in self.disease_symptom.keys():
            disease_match_number[disease] = 0

        for slot in inform_slots.keys():
            for disease in disease_match_number.keys():
                if inform_slots[slot] in self.disease_symptom[disease]["symptom"]:
                    disease_match_number[disease] += 1
        # Get the ratio of informed symptom number to the number of each disease.
        for disease in disease_match_number.keys():
            match_number = copy.deepcopy(disease_match_number[disease])
            disease_match_number[disease] = float(match_number) / len(self.disease_symptom[disease]["symptom"])

        match_disease = max(disease_match_number.items(), key=lambda x: x[1])[0] # Get the most probable disease that the user have.
        # Candidate symptom list of symptoms that belong to the most probable disease but have't been informed yet.
        candidate_symptoms = []
        for symptom in self.disease_symptom[match_disease]["symptom"]:
            if symptom not in inform_slots.keys():
                candidate_symptoms.append(symptom)
        return {"disease":match_disease,"candidate_symptoms":candidate_symptoms}