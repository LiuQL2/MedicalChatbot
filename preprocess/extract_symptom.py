# -*- coding: utf-8 -*-
"""
根据top disease文件中的consult id，从主诉症状文件、对话内容症状抽取口语表达的症状。
"""

import csv
import pandas as pd


class SelfReportSymptomExtractor(object):
    def __init__(self):
        self.symptom = {}
        self.symptom["上呼吸道感染"] = set()
        self.symptom["小儿支气管炎"] = set()
        self.symptom["小儿腹泻"] = set()
        self.symptom["小儿消化不良"] = set()
        self.symptom["小儿感冒"] = set()
        self.symptom["小儿咳嗽"] = set()
        self.symptom["新生儿黄疸"] = set()
        self.symptom["小儿便秘"] = set()
        self.symptom["急性支气管炎"] = set()
        self.symptom["小儿支气管肺炎"] = set()

    def extract(self, file_name):
        reader = csv.reader(open(file_name, encoding="utf-8", mode="r"))
        for line in reader:
            if line[5] in self.symptom.keys():
                self._extract_symptom(line)

    def _extract_symptom(self,line):
        for index in range(7, len(line)):
            print(line)
            if len(line[index]) > 0:
                self.symptom[line[5]].add(line[index])

    def save(self,save_file):
        writer = csv.writer(open(save_file, encoding="utf-8", mode="w"))
        for key in self.symptom.keys():
            for symptom in self.symptom[key]:
                # writer.writerow([key] + list(self.symptom[key]))
                writer.writerow([key,symptom])


class ConversationSymptomExtractor(object):
    def __init__(self):
        self.symptom={}
        self.symptom["上呼吸道感染"] = set()
        self.symptom["小儿支气管炎"] = set()
        self.symptom["小儿腹泻"] = set()
        self.symptom["小儿消化不良"] = set()
        self.symptom["小儿感冒"] = set()
        self.symptom["小儿咳嗽"] = set()
        self.symptom["新生儿黄疸"] = set()
        self.symptom["小儿便秘"] = set()
        self.symptom["急性支气管炎"] = set()
        self.symptom["小儿支气管肺炎"] = set()

    def extract(self,consult_id_file, from_file):
        self.data = pd.read_csv(consult_id_file,header=None,)
        self.data.index = self.data[4]

        data_file = open(from_file,mode="r", encoding="utf-8")
        for line in data_file:
            self._extract(line)
        data_file.close()

    def _extract(self, line):
        line = line.replace("\n", "").split('\t')
        consult_id = int(line[0])
        try:
            disease = self.data.loc[consult_id,5]
            print("*" * 30 + "\n", line)
            if disease in self.symptom.keys():
                for symptom in line[3:len(line)]:
                    self.symptom[disease].add(symptom)
                    print(disease, symptom)
                    print("add symptom")
        except:
            pass

    def save(self,save_file):
        writer = csv.writer(open(save_file, encoding="utf-8", mode="w"))
        for key in self.symptom.keys():
            for symptom in self.symptom[key]:
                writer.writerow([key] + list(self.symptom[key]))
                writer.writerow([key,symptom])




if __name__ == "__main__":
    # # Extracting symptoms from self-report.
    top_disease_symptom_file = "./../resources/top_self_report_extracted_symptom.csv"
    save_file = "./../resources/top_symptom_self_report.csv"
    extractor = SelfReportSymptomExtractor()
    extractor.extract(file_name=top_disease_symptom_file)
    extractor.save(save_file)


    # Extracting symptoms from conversations.
    consult_id_file = "./../resources/top_self_report_extracted_symptom.csv"
    conversation_file = "/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/origin_file/conversation_symptom.txt"
    save_file = "./../resources/top_symptom_conversation.csv"
    extractor = ConversationSymptomExtractor()
    extractor.extract(consult_id_file=consult_id_file,from_file=conversation_file)
    extractor.save(save_file=save_file)
