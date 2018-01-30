# -*- coding: utf-8 -*-
"""
从原始文件中把需要分析的疾病抽取出来，也就是最后保存的结果是（日期	1级科室	2级科室	咨询ID	qid	提问内容	疾病标准名称）。
从/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/origin_file/儿科咨询疾病标注数据.xlsx 文件中把几种数量较多的疾病找
出来，包含主诉内容和疾病信息，但是没有抽取的症状信息。
"""

import pandas as pd
# top_disease_list = ["上呼吸道感染", "小儿发热", "小儿消化不良", "小儿支气管炎","小儿腹泻"]
top_disease_list = ["上呼吸道感染", "小儿消化不良", "小儿支气管炎","小儿腹泻","小儿感冒",
                    "小儿咳嗽","新生儿黄疸","小儿便秘","急性支气管炎","小儿支气管肺炎"]

disease_file = "/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/origin_file/儿科咨询疾病标注数据.xlsx"


data = pd.read_excel(io=disease_file,sheet_name="儿科主诉列表 (修正)")
top_disease_data = pd.DataFrame(columns=data.columns)

key = data.columns[6]
for index in data.index:
    if data.loc[index,key] in top_disease_list:
        print(index, data.loc[index,key])
        top_disease_data = top_disease_data.append(data.loc[index])
        # if len(top_disease_data.index) > 10:
        #     break
del top_disease_data["Unnamed: 7"]
del top_disease_data["Unnamed: 8"]
del top_disease_data["Unnamed: 9"]
del top_disease_data["Unnamed: 10"]
print(top_disease_data)

top_disease_data.to_csv("./../resources/top_self_report_text.csv", sep="\t",index=True, header=True)


class TopDiseaseReporter(object):
    def __init__(self, top_disease_list, disease_file):
        self.disease_file = disease_file
        self.top_disease_list = top_disease_list
        self.data = pd.read_excel(io=disease_file, sheet_name="儿科主诉列表 (修正)")
        self.top_disease_data = pd.DataFrame(columns=data.columns)

    def load_data(self):
        key = data.columns[6]
        for index in data.index:
            if data.loc[index, key] in top_disease_list:
                print(index, data.loc[index, key])
                self.top_disease_data = self.top_disease_data.append(data.loc[index])
                # if len(top_disease_data.index) > 10:
                #     break
        del self.top_disease_data["Unnamed: 7"]
        del self.top_disease_data["Unnamed: 8"]
        del self.top_disease_data["Unnamed: 9"]
        del self.top_disease_data["Unnamed: 10"]
        print(self.top_disease_data)

    def save(self, save_file):
        self.top_disease_data.to_csv(save_file, sep="\t", index=True, header=True)
