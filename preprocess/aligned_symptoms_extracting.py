# -*-coding: utf-8 -*-
"""
对主诉症状、问答症状进行归一。
使用简单的字符串相识度匹配。
"""
import csv
import pandas
import difflib
import time
import json
import Levenshtein


class SymptomAligner(object):
    """
    Aligning spoken symptom on writing symptom.
    """
    def __init__(self, aligned_symptom_file, threshold, hand_crafted_symptom=True):
        self.hand_crafted_symptom = hand_crafted_symptom
        self.threshold = threshold
        self.aligned_symptom = dict()
        data_file = open(aligned_symptom_file, "r", encoding="utf-8")
        for line in data_file:
            line = json.loads(line)
            self.aligned_symptom.setdefault(line["name"],dict())
            self.aligned_symptom[line["name"]]["symptom"] = line["symptom"]
            self.aligned_symptom[line["name"]]["src_symptom"] = line["src_symptom"]
        data_file.close()

    def align(self, spoken_symptom):
        """
        Return the writing symptom given a spoken symptom using the similarity score between strings.
        :param spoken_symptom: spoken_symptom
        :return: writing symptom aligned with the spoken_symptom.
        """
        similarity_score = {}
        if self.hand_crafted_symptom == True:
            for disease in self.aligned_symptom.keys():
                for key, value in self.aligned_symptom[disease]["symptom"].items():
                    similarity_score[key] = Levenshtein.ratio(spoken_symptom.replace("小儿", ""), key.replace("小儿", ""))
                    for symptom in value:
                        score = Levenshtein.ratio(spoken_symptom.replace("小儿", ""), symptom.replace("小儿", ""))
                        if score > similarity_score[key]:
                            similarity_score[key] = score
        else:
            for disease in self.aligned_symptom.keys():
                for symptom in self.aligned_symptom[disease]["symptom"]:
                    similarity_score[symptom] = Levenshtein.ratio(spoken_symptom.replace("小儿", ""), symptom.replace("小儿", ""))

            # for key, value in self.aligned_symptom[disease]["src_symptom"].items():
            #     score = Levenshtein.ratio(spoken_symptom.replace("小儿", ""), key.replace("小儿", ""))

        writing_symptom = sorted(similarity_score, key=lambda x:similarity_score[x])[-1]
        score = similarity_score[writing_symptom]
        if score >= self.threshold:
            # print("writing_symptom:", writing_symptom, "score:", score,"spoken_symptom:", spoken_symptom)
            return writing_symptom
        else:
            return None


class DataLoader(object):
    def __init__(self,threshold, disease_symptom_aligned_file, hand_crafted_symptom, top_disease_list):
        self.top_disease_list = top_disease_list
        self.symptom_aligner = SymptomAligner(disease_symptom_aligned_file, threshold=threshold,hand_crafted_symptom=hand_crafted_symptom)
        self.sample = {}
        self.symptom_slots = set()
        self.deny_list = ["不","否","没有","没"]

    def load_self_report(self, self_report_file):
        """
        用来对主诉内容的症状进行归一化处理。
        :param self_report_file:
        :return:
        """
        data_reader = csv.reader(open(self_report_file, "r",encoding="utf-8"))
        for line in data_reader:
            # print(line)
            if line[5] not in self.top_disease_list: continue
            self.sample.setdefault(line[4], dict())
            self.sample[line[4]]["request_slots"] = {"disease":line[5]}
            self.sample[line[4]].setdefault("explicit_inform_slots", dict())
            self.sample[line[4]].setdefault("implicit_inform_slots", dict())
            try:
                index = line.index("")
            except:
                index = len(line)
            symptom_list = line[8:index]
            for symptom in symptom_list:
                spoken_symptom = symptom.replace("\n","")
                writing_symptom = self.symptom_aligner.align(spoken_symptom)
                if writing_symptom != None:
                    self.sample[line[4]]["explicit_inform_slots"][spoken_symptom] = writing_symptom

    def load_conversation(self, conversation_file):
        """
        用来对conversation的症状数据进行归一化处理。
        :param conversation_file:
        :return:
        """
        data_file = open(conversation_file, mode="r", encoding="utf-8")
        for line in data_file:
            line = line.split("\t")
            temp_line = []
            for index in range(0, len(line)):
                if len(line[index]) != 0: temp_line.append(line[index])
            line = temp_line

            # 判断是否是四种疾病下的conversation，然后进行症状归一化。
            # if line[0] in self.sample.keys() and str(line[1]) == str(2):#只抽取患者说的症状。2：患者说的话，3：医生说的话
            if line[0] in self.sample.keys():#抽取患者和医生说的症状。
                for index in range(3, len(line)):
                    spoken_symptom = line[index].replace("\n","")
                    writing_symptom = self.symptom_aligner.align(spoken_symptom)
                    if writing_symptom != None and (spoken_symptom not in self.sample[line[0]]["explicit_inform_slots"].keys()):
                        self.sample[line[0]]["implicit_inform_slots"][spoken_symptom] = writing_symptom
        data_file.close()

    def write_slot_value(self, file_name):
        data_file = open(file_name,mode="w")
        for key, value in self.sample.items():
            line = {}
            line["consult_id"] = key
            line["disease_tag"] = self.sample[key]["request_slots"]["disease"]
            line["goal"] = {}
            line["goal"].setdefault("request_slots", dict())
            line["goal"].setdefault("explicit_inform_slots", dict())
            line["goal"].setdefault("implicit_inform_slots", dict())
            line["goal"]["request_slots"]["disease"] = "UNK"

            for spoken_symptom, writing_symptom in value["explicit_inform_slots"].items():
                print("spoken:",spoken_symptom, writing_symptom)
                line["goal"]["explicit_inform_slots"][writing_symptom] = self._true_or_false(spoken_symptom, writing_symptom)
                self.symptom_slots.add(writing_symptom)
            for spoken_symptom, writing_symptom in value["implicit_inform_slots"].items():
                print("spoken:",spoken_symptom, writing_symptom)
                if writing_symptom in line["goal"]["explicit_inform_slots"].keys(): continue
                line["goal"]["implicit_inform_slots"][writing_symptom] = self._true_or_false(spoken_symptom,writing_symptom)
                self.symptom_slots.add(writing_symptom)

            data_file.write(json.dumps(line) + "\n")
        data_file.close()

    def write(self, file_name):
        data_file = open(file_name,mode="w")
        data_file.write(json.dumps(self.sample) + "\n")
        data_file.close()

    def write_slots(self, file_name):
        data_file = open(file_name, mode="w",encoding="utf-8")
        for symptom in self.symptom_slots:
            data_file.write(symptom + "\n")
        data_file.close()

    def _true_or_false(self, spoken_symptom, writing_symptom):
        exception_symptom_list = ["烦躁不安","呼吸不畅"]
        exception_list = ["不舒服"]
        return_value = True
        for s_ in self.deny_list:
            if s_ in spoken_symptom and writing_symptom not in exception_symptom_list:
                return_value = False

        for e in exception_list:
            if e in spoken_symptom:
                return_value = True
        return return_value



if __name__ == "__main__":
    threshold = 0.2
    disease_symptom_aligned_file = "./../resources/top_disease_symptom_aligned.json"
    report_loader = DataLoader(threshold=threshold,disease_symptom_aligned_file=disease_symptom_aligned_file)
    report_loader.load_self_report("./../resources/top_self_report_extracted_symptom.csv")
    print("Conversation:")
    time.sleep(5)
    report_loader.load_conversation("/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/origin_file/conversation_symptom.txt")

    report_loader.write("./../resources/goal_spoken_writing_"+str(threshold) + ".json")
    report_loader.write_slot_value("./../resources/goal_slot_value_"+str(threshold) + ".json")
