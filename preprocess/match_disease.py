# -*- coding: utf-8 -*-
"""
从包含所有主诉内容和症状的文件（self_report_extracted_symptom.csv)中抽取出前几种疾病的主诉内容和症状。
"""
import pandas as pd
import csv

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
            if line[5] in self.top_disease_list:
                print(line)
                writer.writerow(line)
                index += 1
        print(index)
        save_file.close()
        report_symptom.close()
