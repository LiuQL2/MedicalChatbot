# -*- coding:utf-8 -*-
"""
Used for statistics of user goal
疾病数量 每一种疾病对应的数据量、平均每一轮对话用户的symptom有多少（显性、隐性）
"""

import json
import pickle
import csv


class Statistics(object):
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



if __name__ == "__main__":
    data_file = "./../src/dialogue_system/data/goal_set.p"
    save_file = "./../resources/goal_set_statistics.csv"
    stata = Statistics(data_file=data_file)
    stata.statistics()
    stata.write_file(save_file=save_file)