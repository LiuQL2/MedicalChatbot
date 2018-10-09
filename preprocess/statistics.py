# -*- coding:utf-8 -*-
"""
Used for statistics of user goal
疾病数量 每一种疾病对应的数据量、平均每一轮对话用户的symptom有多少（显性、隐性）
"""

import json
import pickle
import csv
import pandas as pd


class StatisticsOfUserGoal(object):
    def __init__(self, data_file):
        self.file_name = data_file
        goal_set = pickle.load(file=open(self.file_name, "rb"))
        self.goal_set = goal_set["train"] + goal_set["test"] + goal_set["validate"]
        self.information = {}


# """
# {
# 	"consult_id":123,
# 	"request_slots":{
# 	"disease": "UNK"
# 	},
# 	"explicit_inform_slots":{
# 	"咳嗽":true
# 	},
# 	"implicit_inform_slots":{
# 	}
# }
# """

    def statistics(self):
        for goal in self.goal_set:
            print(json.dumps(goal,indent=2))
            disease = goal["disease_tag"]
            if disease not in self.information.keys():
                self.information[disease] = {}
                self.information[disease]["user_number"] = 0
                self.information[disease]["explicit_number"] = 0
                self.information[disease]["implicit_number"] = 0
        for goal in self.goal_set:
            disease = goal["disease_tag"]
            explicit_inform_slots = goal["goal"]["explicit_inform_slots"]
            implicit_inform_slots = goal["goal"]["implicit_inform_slots"]
            if len(goal["goal"]["explicit_inform_slots"].keys()) >= 0 and \
                len(goal["goal"]["implicit_inform_slots"].keys()) >= 0:

                self.information[disease]["user_number"] += 1
                self.information[disease]["explicit_number"] += len(explicit_inform_slots.keys())
                self.information[disease]["implicit_number"] += len(implicit_inform_slots.keys())

        disease_list = list(self.information.keys())
        for disease in disease_list:
            explicit_number = self.information[disease]["explicit_number"]
            implicit_number = self.information[disease]["implicit_number"]
            self.information[disease]["explicit_number"] = float(explicit_number) / self.information[disease]["user_number"]
            self.information[disease]["implicit_number"] = float(implicit_number) / self.information[disease]["user_number"]
        print(json.dumps(self.information))

    def write_file(self, save_file):
        data_file = open(save_file, "w",encoding="utf-8")
        writer = csv.writer(data_file)
        writer.writerow(["disease", "user_number", "explicit_number", "implicit_number"])
        for disease in self.information.keys():
            writer.writerow([disease, self.information[disease]["user_number"], self.information[disease]["explicit_number"],self.information[disease]["implicit_number"]])
        data_file.close()


class StatisticsOfDiseaseSymptom(object):
    def __init__(self,disease_symptom_file):
        self.disease_symptom_file = disease_symptom_file
        self.disease_list = []

    def statistics(self):
        disease_symptoms = {}
        data_file = open(file=self.disease_symptom_file,mode="r", encoding="utf-8")
        for line in data_file:
            line = json.loads(line)
            self.disease_list.append(line["name"])
            disease_symptoms[line["name"]] = line["symptom"].keys()
            print(line)

        result = pd.DataFrame(index=self.disease_list,columns=self.disease_list)

        for index1 in range(0, len(self.disease_list), 1):
            for index2 in range(index1, len(self.disease_list), 1):
                count = 0
                for symptom1 in disease_symptoms[self.disease_list[index1]]:
                    if symptom1 in disease_symptoms[self.disease_list[index2]]: count += 1
                result.loc[self.disease_list[index2], self.disease_list[index1]] = count
        data_file.close()
        self.result = result

    def save(self, file_name):
        # self.result.to_csv(file_name,encoding="utf-8")
        self.result.to_excel(file_name, sheet_name="Sheet1")



if __name__ == "__main__":
    # statics for the goal set, e.g., average number of explicit symptoms, average of number of implicit symptoms and the
    # number of user goal of each disease.

    data_file = "./../src/dialogue_system/data/dataset/label/goal_set.p"
    save_file = "./../resources/goal_set_statistics.csv"
    save_file = "/Users/qianlong/Desktop/goal_set_statistics.csv"

    stata = StatisticsOfUserGoal(data_file=data_file)
    stata.statistics()
    stata.write_file(save_file=save_file)


    # statistics of overlap in symptoms for different diseases.
    # data_file = "./../resources/top_disease_symptom_aligned.json"
    # save_to = "./../resources/overlap_disease_symptom.xlsx"
    # statistics = StatisticsOfDiseaseSymptom(data_file)
    # statistics.statistics()
    # statistics.save(save_to)