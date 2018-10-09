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

    def load_data_consult_id(self, consult_id_list):
        """
        Match conversations from consult id.
        :param consult_id_list:
        :return:
        """
        key = self.data.columns[3]
        for index in self.data.index:
            if str(self.data.loc[index, key]) in consult_id_list:
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


if __name__ == "__main__":
    import pickle
    import json
    import random
    from symptom_liking import ReportConversation

    top_disease_list = ["上呼吸道感染", "小儿支气管炎", "小儿腹泻", "小儿消化不良", ]

    disease_file = "/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/origin_file/儿科咨询疾病标注数据.xlsx"
    top_self_report_file_save = "./../resources/880/top_self_report_text.csv"
    goal_set = pickle.load(open("./../src/dialogue_system/data/4_diseases/both/goal_set.p","rb"))
    consult_id_dict = {}
    for key in goal_set.keys():
        print(key, len(goal_set[key]))
        for goal in goal_set[key]:
            if len(goal["goal"]["explicit_inform_slots"].keys()) >= 0 and\
                len(goal["goal"]["implicit_inform_slots"].keys()) >= 1:
                consult_id_dict.setdefault(goal["disease_tag"],list())
                consult_id_dict[goal["disease_tag"]].append(goal["consult_id"])
    print(consult_id_dict.keys())

    consult_id_list = []
    for key in consult_id_dict.keys():
        print(key, len(consult_id_dict[key]))
        consult_id_list = consult_id_list + list(random.sample(consult_id_dict[key], 250))
    print(len(consult_id_list))
    print(type(consult_id_list[0]))


    conversation_file_name = "/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/origin_file/conversation.txt"
    top_conversation_save = "./../resources/880/top_conversation.txt"
    conversation = ReportConversation()
    consult_id_list = conversation.match(conversation_file_name=conversation_file_name, save_file_name=top_conversation_save,consult_id_list=consult_id_list)



    reporter = TopDiseaseReporter(top_disease_list=top_disease_list,disease_file=disease_file)
    reporter.load_data_consult_id(consult_id_list=consult_id_list)
    reporter.save(save_file=top_self_report_file_save)
    pickle.dump(file=open("./../resources/880/consult_id_list.p","wb"),obj=consult_id_list)
    print(len(consult_id_list))





