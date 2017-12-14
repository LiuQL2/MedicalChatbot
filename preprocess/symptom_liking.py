# -*- coding: utf-8 -*-

import pandas as pd

# data = pd.read_csv("./../resources/top_disease_data.csv", sep="\t",index_col=0)
# print(data)
# disease_list = list(set(data["疾病标准名称"]))
# print(data[data["疾病标准名称"] == "小儿消化不良"].sample(n=200))
# columns = ['日期', '咨询ID', 'qid', '1级科室', '2级科室', '疾病标准名称', '提问内容']
# for disease in disease_list:
#     disease_data = data[data["疾病标准名称"] == disease].sample(n=200)
#     disease_data.to_csv("./../resources/project_disease_data.csv", sep="\t",columns=columns, index=False, header=False, encoding="utf-8", mode="a+")

project_data = pd.read_csv("./../resources/project_disease_data.csv", sep="\t")
print(project_data)
consult_id_list = list(project_data['咨询ID'])
print(type(consult_id_list[0]))
conversation_file = open("/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/origin_file/BaiduDoctor",'r',encoding="utf-8")
project_conversation_file = open("./../resources/project_conversation_data.txt", mode="w",encoding="utf-8")
write = False
for line in conversation_file:
    line = line.replace("\n","")
    if "consult_id" in line:
        line = line.replace(' ', "")
        consult_id = int(line.split(":")[1])
        line = "\n" + line
        if consult_id in consult_id_list:
            write = True
    if len(line) == 0:
        write = False
    if write:
        project_conversation_file.write(line + "\n")
conversation_file.close()