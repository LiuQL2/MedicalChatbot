# -*- coding: utf-8 -*-
"""
从包含所有主诉内容和症状的文件（self_report_extracted_symptom.csv)中抽取出前几种疾病的主诉内容和症状。
"""
import pandas as pd
import csv

# top_disease_list = ["上呼吸道感染", "小儿发热", "小儿消化不良", "小儿支气管炎","小儿腹泻"]
top_disease_list = ["上呼吸道感染", "小儿消化不良", "小儿支气管炎","小儿腹泻","小儿感冒",
                    "小儿咳嗽","新生儿黄疸","小儿便秘","急性支气管炎","小儿支气管肺炎"]

# 频率较高的top疾病数据
# top_disease_file = "./../resources/top_self_report_text.csv"
# top_disease_data = pd.read_csv(top_disease_file,sep="\t")

#患者主诉与病症匹配之后的数据，但是包含全部疾病的，需要提取top疾病
disease_symptom_file = open("./../resources/self_report_extracted_symptom.csv",mode='r',encoding="utf-8")
symptom_reader = csv.reader(disease_symptom_file)

save_file = open("./../resources/top_self_report_extracted_symptom.csv",encoding="utf-8",mode="w")
writer = csv.writer(save_file)

index = 0
for line in symptom_reader:
    if line[5] in top_disease_list:
        print(line)
        writer.writerow(line)
        index += 1
print(index)

save_file.close()
disease_symptom_file.close()


class DiseaseMatch(object):
    def __init__(self, top_disease_list, self_report_extracted_symptom_file):
        self.report_extracted_symptom_file = self_report_extracted_symptom_file
        self.top_disease_list = top_disease_list

    def match(self, save_file_name):
        report_symptom = open(self.report_extracted_symptom_file,mode='r',encoding="utf-8")
        report_symptom_reader = csv.reader(report_symptom)

        save_file = open(save_file_name,encoding="utf-8",mode="w")
        writer = csv.writer(save_file)

        index = 0
        for line in report_symptom_reader:
            if line[5] in top_disease_list:
                print(line)
                writer.writerow(line)
                index += 1
        print(index)
        save_file.close()
        report_symptom.close()
