# -*- coding: utf-8 -*-
"""
从原始文件中把需要分析的疾病抽取出来，也就是最后保存的结果是（日期	1级科室	2级科室	咨询ID	qid	提问内容	疾病标准名称）。
从/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/origin_file/儿科咨询疾病标注数据.xlsx 文件中把几种数量较多的疾病找
出来，包含主诉内容和疾病信息，但是没有抽取的症状信息。
"""
import pandas as pd

class TopDiseaseReporter(object):
    def __init__(self, top_disease_list, disease_file):
        self.disease_file = disease_file
        self.top_disease_list = top_disease_list
        self.data = pd.read_excel(io=disease_file, sheet_name="儿科主诉列表 (修正)")
        self.top_disease_data = pd.DataFrame(columns=self.data.columns)

    def load_data(self):
        key = self.data.columns[6]
        for index in self.data.index:
            if self.data.loc[index, key] in self.top_disease_list:
                print(index, self.data.loc[index, key])
                self.top_disease_data = self.top_disease_data.append(self.data.loc[index])
                # if len(top_disease_data.index) > 10:
                #     break
        del self.top_disease_data["Unnamed: 7"]
        del self.top_disease_data["Unnamed: 8"]
        del self.top_disease_data["Unnamed: 9"]
        del self.top_disease_data["Unnamed: 10"]
        print(self.top_disease_data)

    def save(self, save_file):
        self.top_disease_data.to_csv(save_file, sep="\t", index=True, header=True)
