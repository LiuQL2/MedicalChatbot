# -*- coding: utf-8 -*-

import pandas as pd
import csv

top_disease_list = ["上呼吸道感染", "小儿发热", "小儿消化不良", "小儿支气管炎","小儿腹泻"]

# 频率较高的五种疾病数据
# top_disease_file = "./../resources/top_disease_data.csv"
# top_disease_data = pd.read_csv(top_disease_file,sep="\t")

#患者主诉与病症匹配之后的数据，但是包含全部疾病的，需要提取前五种疾病
disease_symptom_file = open("./../resources/disease_symptom.csv",mode='r',encoding="utf-8")
symptom_reader = csv.reader(disease_symptom_file)

save_file = open("./../resources/top_disease_symptom.csv",encoding="utf-8",mode="w")
writer = csv.writer(save_file)

index = 0
for line in symptom_reader:
    if line[5] in top_disease_list:
        print(line)
        # writer.writerow(line)
        index += 1
print(index)

save_file.close()
disease_symptom_file.close()



