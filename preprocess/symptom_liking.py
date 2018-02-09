# -*- coding: utf-8 -*-

"""
conversation.txt是包含所有疾病的对话内容（没有主诉），下面部分代码是从这个文件中将所要研究的几种疾病数据抽取出来一边进行下一步处理。
实际中从原始文件到可以使用的user goal文件，这个代码应该用不到。
"""

import pandas as pd


class ReportConversation(object):
    def __init__(self):
        pass

    def match(self, conversation_file_name, save_file_name,report_file_name=None,consult_id_list=None):
        assert (report_file_name is None or consult_id_list is None), "no consult id is provided."
        if report_file_name != None:
            project_data = pd.read_csv(report_file_name, sep="\t")
            print(project_data)
            consult_id_list = list(project_data['咨询ID'])

        return self.__match_based_on_id__(conversation_file_name,save_file_name,consult_id_list)

    def __match_based_on_id__(self, conversation_file_name, save_file_name,consult_id_list):
        conversation_file = open(conversation_file_name, 'r', encoding="utf-8")
        save_conversation_file = open(save_file_name, mode="w", encoding="utf-8")
        found_consult_id_list = []
        write = False
        for line in conversation_file:
            line = line.replace("\n", "")
            if "consult_id" in line:
                line = line.replace(' ', "")
                consult_id = str(line.split(":")[1])
                line = "\n" + line
                if consult_id in consult_id_list:
                    found_consult_id_list.append(consult_id)
                    write = True
                    print(consult_id)
            if len(line) == 0:
                write = False
            if write:
                save_conversation_file.write(line + "\n")
        conversation_file.close()
        save_conversation_file.close()
        return found_consult_id_list


