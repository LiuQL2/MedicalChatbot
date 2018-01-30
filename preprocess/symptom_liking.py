# -*- coding: utf-8 -*-

"""
conversation.txt是包含所有疾病的对话内容（没有主诉），下面部分代码是从这个文件中将所要研究的几种疾病数据抽取出来一边进行下一步处理。
实际中从原始文件到可以使用的user goal文件，这个代码应该用不到。
"""

import pandas as pd

# 每种疾病抽样200条数据进行下一步的处理，并改变column的顺序。
# data = pd.read_csv("./../resources/top_self_report_text.csv", sep="\t",index_col=0)
# print(data)
# disease_list = list(set(data["疾病标准名称"]))
# print(data[data["疾病标准名称"] == "小儿消化不良"].sample(n=200))
# columns = ['日期', '咨询ID', 'qid', '1级科室', '2级科室', '疾病标准名称', '提问内容']
# for disease in disease_list:
#     disease_data = data[data["疾病标准名称"] == disease].sample(n=200)
#     disease_data.to_csv("./../resources/project_disease_data.csv", sep="\t",columns=columns, index=False, header=True, encoding="utf-8", mode="a+")


#conversation.txt是包含所有疾病的对话内容（没有主诉），下面部分代码是从这个文件中将所要研究的几种疾病数据抽取出来一边进行下一步处理。
project_data = pd.read_csv("./../resources/project_disease_data.csv", sep="\t")
print(project_data)
consult_id_list = list(project_data['咨询ID'])
print(consult_id_list[0])
conversation_file = open("/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/origin_file/conversation.txt",'r',encoding="utf-8")
project_conversation_file = open("./../resources/project_conversation_data.txt", mode="w",encoding="utf-8")
write = False
for line in conversation_file:
    line = line.replace("\n","")
    if "consult_id" in line:
        line = line.replace(' ', "")
        consult_id = str(line.split(":")[1])
        line = "\n" + line
        if consult_id in consult_id_list:
            write = True
            print(consult_id)
    if len(line) == 0:
        write = False
    if write:
        project_conversation_file.write(line + "\n")
conversation_file.close()