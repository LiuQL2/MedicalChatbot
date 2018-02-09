# -*- coding:utf-8 -*-

from top_disease import TopDiseaseReporter
from match_disease import DiseaseMatch
from aligned_symptoms_extracting import DataLoader

# 10 diseases
# top_disease_list = ["上呼吸道感染", "小儿消化不良", "小儿支气管炎","小儿腹泻","小儿感冒",
#                     "小儿咳嗽","新生儿黄疸","小儿便秘","急性支气管炎","小儿支气管肺炎"]

# 8 diseases
# top_disease_list = ["上呼吸道感染", "小儿支气管炎","小儿腹泻","小儿感冒"
#                     ,"新生儿黄疸","小儿便秘","急性支气管炎","小儿支气管肺炎"]

# 7 diseases.
# top_disease_list = ["上呼吸道感染", "小儿支气管炎", "小儿腹泻", "小儿感冒",
#                 "小儿咳嗽", "急性支气管炎", "小儿支气管肺炎"]

# 4 diseases.
top_disease_list = ["上呼吸道感染", "小儿支气管炎", "小儿腹泻", "小儿消化不良",]

# TODO: fist step
# 从原始文件中把需要分析的疾病抽取出来，也就是最后保存的结果是（日期	1级科室	2级科室	咨询ID	qid	提问内容	疾病标准名称）。
# 从/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/origin_file/儿科咨询疾病标注数据.xlsx 文件中把几种数量较多的疾病找
# 出来，包含主诉内容和疾病信息，但是没有抽取的症状信息。
def top_disease():
    disease_file = "/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/origin_file/儿科咨询疾病标注数据.xlsx"
    top_self_report_file_save = "./../resources/top_self_report_text.csv"

    report = TopDiseaseReporter(top_disease_list=top_disease_list, disease_file=disease_file)
    report.load_data()
    report.save(save_file=top_self_report_file_save)


# TODO: second step
# 从包含所有主诉内容和症状的文件（self_report_extracted_symptom.csv)中抽取出前几种疾病的主诉内容和症状。
def match_top_self_report():
    self_report_extracted_symptom_file = "./../resources/self_report_extracted_symptom.csv"  # 一直都不要改变，固定的文件
    save_file_name = "./../resources/top_self_report_extracted_symptom.csv"

    match = DiseaseMatch(top_disease_list=top_disease_list,
                         self_report_extracted_symptom_file=self_report_extracted_symptom_file)
    match.match(save_file_name=save_file_name)


# TODO: third step
# 对主诉症状、问答症状进行归一。
# 使用简单的字符串相识度匹配。
def symptom_normalization():
    threshold = 0.2
    disease_symptom_aligned_file = "./../resources/top_disease_symptom_aligned.json"
    top_self_report_extracted_symptom_file = "./../resources/top_self_report_extracted_symptom.csv"
    conversation_symptom_file = "/Users/qianlong/Documents/Qianlong/Research/MedicalChatbot/origin_file/conversation_symptom.txt"
    goal_spoken_writing_save = "./../resources/goal_spoken_writing_" + str(threshold) + ".json"
    goal_slot_value_save = "./../resources/goal_slot_value_" + str(threshold) + ".json"
    hand_crafted_symptom = True # 是否是人工匹配的症状归一，如果是的，对应的文件为top_disease_symptom_aligned.json，不是需要使用原始的文件。

    report_loader = DataLoader(threshold=threshold, disease_symptom_aligned_file=disease_symptom_aligned_file,
                               hand_crafted_symptom=hand_crafted_symptom,
                               top_disease_list=top_disease_list)
    report_loader.load_self_report(self_report_file=top_self_report_extracted_symptom_file)
    print("Conversation:")
    report_loader.load_conversation(conversation_file=conversation_symptom_file)

    slot_file = "./../resources/slot_set.txt"
    report_loader.write(file_name=goal_spoken_writing_save)
    report_loader.write_slot_value(file_name=goal_slot_value_save)
    report_loader.write_slots(file_name=slot_file)


if __name__ == "__main__":
    top_disease()
    # match_top_self_report()
    # symptom_normalization()