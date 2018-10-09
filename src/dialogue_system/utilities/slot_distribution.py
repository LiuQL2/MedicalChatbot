# -*- coding:utf-8 -*-

import json
import pickle
import pandas as pd


class SlotDistributor(object):
    def __init__(self, goal_set, slot_set, disease_symptom):
        self.goal_set = pickle.load(open(goal_set, "rb"))
        self.slot_set = pickle.load(open(slot_set, "rb"))
        self.disease_symptom = pickle.load(open(disease_symptom, "rb"))
        self.symptom_distribution = {}
        self.slot_set.pop("disease")
        for symptom in self.slot_set.keys():
            if symptom != "disease" :
                self.symptom_distribution[symptom] = {}
                self.symptom_distribution[symptom]["total"] = 0.0
                self.symptom_distribution[symptom]["total_explicit"] = 0.0
                self.symptom_distribution[symptom]["total_implicit"] = 0.0
                for disease in self.disease_symptom.keys():
                    print(disease)
                    self.symptom_distribution[symptom][disease] = {}
                    self.symptom_distribution[symptom][disease]["total"] = 0.0
                    self.symptom_distribution[symptom][disease]["implicit"] = 0.0
                    self.symptom_distribution[symptom][disease]["explicit"] = 0.0

    def calculate(self):
        """
        统计每一个symptom在每种疾病下出现的个数，分为总的个数，在explicit的个数，在implicit的个数。
        :return:
        """
        key_list = self.goal_set.keys()
        for key in key_list:
            for goal in self.goal_set[key]:
                disease = goal["disease_tag"]
                if goal["consult_id"] == "10000894":
                    print(goal)
                    exit(0)

                for symptom in goal["goal"]["explicit_inform_slots"].keys():
                    self.symptom_distribution[symptom]["total"] += 1
                    self.symptom_distribution[symptom]["total_explicit"] += 1
                    self.symptom_distribution[symptom][disease]["total"] += 1
                    self.symptom_distribution[symptom][disease]["explicit"] += 1
                for symptom in goal["goal"]["implicit_inform_slots"].keys():
                    self.symptom_distribution[symptom]["total"] += 1
                    self.symptom_distribution[symptom]["total_implicit"] += 1
                    self.symptom_distribution[symptom][disease]["total"] += 1
                    self.symptom_distribution[symptom][disease]["implicit"] += 1

    def write(self,file_name):

        pickle.dump(file=open(file_name, "wb"), obj=self.symptom_distribution)

    def to_dataframe(self):

        explicit_data = pd.DataFrame(index=list(self.slot_set.keys()), columns=list(self.disease_symptom.keys()))
        implicit_data = pd.DataFrame(index=list(self.slot_set.keys()), columns=list(self.disease_symptom.keys()))
        total_data = pd.DataFrame(index=list(self.slot_set.keys()), columns=list(self.disease_symptom.keys()))
        for symptom, value in self.symptom_distribution.items():
            for disease in self.disease_symptom.keys():
                if value["total_explicit"] > 0.0:
                    explicit_data.loc[symptom, disease] = value[disease]["explicit"] / value["total_explicit"]
                else:
                    explicit_data.loc[symptom, disease] = 0.0
                if value["total_implicit"] > 0.0:
                    implicit_data.loc[symptom, disease] = value[disease]["implicit"] / value["total_implicit"]
                else:
                    implicit_data.loc[symptom, disease] = 0.0
                if value["total"] > 0.0:
                    total_data.loc[symptom, disease] = value[disease]["total"] / value["total"]
                else:
                    total_data.loc[symptom, disease] = 0.0

        explicit_data.to_excel("./../../../resources/symptom_distribution/explicit.xlsx", sheet_name="explicit_distribution" )
        implicit_data.to_excel("./../../../resources/symptom_distribution/implicit.xlsx", sheet_name="implicit_distribution" )
        total_data.to_excel("./../../../resources/symptom_distribution/total.xlsx", sheet_name="total_distribution" )





if __name__ == "__main__":
    slot_set = "./../data/10_diseases/slot_set.p"
    goal_set = "./../data/10_diseases/goal_set.p"
    disease_symptom = "./../data/10_diseases/disease_symptom.p"
    distributor = SlotDistributor(goal_set,slot_set,disease_symptom)
    distributor.calculate()
    distributor.write("./../../../resources/symptom_distribution/symptom_distribution.p")
    distributor.to_dataframe()