# -*- coding: utf-8 -*-

import pandas as pd
top_disease_list = ["上呼吸道感染", "小儿发热", "小儿消化不良", "小儿支气管炎","小儿腹泻"]

disease_file = "/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/儿科咨询疾病标注数据.xlsx"


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

top_disease_data.to_csv("./../resources/top_disease_data.csv", sep="\t",index=True, header=True)


import nltk
nltk.ConditionalFreqDist()