# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import pickle
import os
import time


class Ploter(object):
    def __init__(self, performance_file):
        self.performance_file = performance_file
        self.epoch_size = 0
        self.success_rate = {}
        self.average_reward = {}
        self.average_wrong_disease = {}
        self.average_turn = {}

    def load_data(self, performance_file, label):
        performance = pickle.load(file=open(performance_file, "rb"))
        self.epoch_size = max(self.epoch_size, len(performance.keys()))
        sr, ar, awd,at = self.__load_data(performance=performance)
        self.success_rate[label] = sr
        self.average_reward[label] = ar
        self.average_wrong_disease[label] = awd
        self.average_turn[label] = at

    def __load_data(self, performance):
        success_rate = []
        average_reward = []
        average_wrong_disease = []
        average_turn = []
        for index in range(0, len(performance.keys()),1):
            print(performance[index].keys())
            success_rate.append(performance[index]["success_rate"])
            average_reward.append(performance[index]["average_reward"])
            average_wrong_disease.append(performance[index]["average_wrong_disease"])
            average_turn.append(performance[index]["average_turn"])
        return success_rate, average_reward, average_wrong_disease,average_turn


    def plot(self, save_name, label_list):
        # epoch_index = [i for i in range(0, 500, 1)]

        for label in self.success_rate.keys():
            epoch_index = [i for i in range(0, len(self.success_rate[label]), 1)]

            plt.plot(epoch_index,self.success_rate[label][0:max(epoch_index)+1], label=label, linewidth=1)
            # plt.plot(epoch_index,self.average_turn[label][0:max(epoch_index)+1], label=label+"at", linewidth=1)

        # plt.hlines(0.11,0,epoch_index,label="Random Agent", linewidth=1, colors="r")
        # plt.hlines(0.38,0,epoch_index,label="Rule Agent", linewidth=1, colors="purple")

        plt.xlabel("Simulation Epoch")
        plt.ylabel("Success Rate")
        plt.title("Learning Curve")
        # if len(label_list) >= 2:
        #     plt.legend()
        # plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(save_name,dpi=400)

        plt.show()

    @staticmethod
    def get_dirlist(path, key_word_list=None, no_key_word_list=None):
        file_name_list = os.listdir(path)  # 获得原始json文件所在目录里面的所有文件名称
        if key_word_list == None and no_key_word_list == None:
            temp_file_list = file_name_list
        elif key_word_list != None and no_key_word_list == None:
            temp_file_list = []
            for file_name in file_name_list:
                have_key_words = True
                for key_word in key_word_list:
                    if key_word not in file_name:
                        have_key_words = False
                        break
                    else:
                        pass
                if have_key_words == True:
                    temp_file_list.append(file_name)
        elif key_word_list == None and no_key_word_list != None:
            temp_file_list = []
            for file_name in file_name_list:
                have_no_key_word = False
                for no_key_word in no_key_word_list:
                    if no_key_word in file_name:
                        have_no_key_word = True
                        break
                if have_no_key_word == False:
                    temp_file_list.append(file_name)
        elif key_word_list != None and no_key_word_list != None:
            temp_file_list = []
            for file_name in file_name_list:
                have_key_words = True
                for key_word in key_word_list:
                    if key_word not in file_name:
                        have_key_words = False
                        break
                    else:
                        pass
                have_no_key_word = False
                for no_key_word in no_key_word_list:
                    if no_key_word in file_name:
                        have_no_key_word = True
                        break
                    else:
                        pass
                if have_key_words == True and have_no_key_word == False:
                    temp_file_list.append(file_name)

        return temp_file_list


if __name__ == "__main__":
    # file_name = "./../model/dqn/learning_rate/learning_rate_d4_e999_agent1_dqn1.p"
    # file_name = "/Users/qianlong/Desktop/learning_rate_d4_e_agent1_dqn1_T22_lr0.001_SR44_mls0_gamma0.95_epsilon0.1_1499.p"
    # save_name = file_name + ".png"
    # ploter = Ploter(file_name)
    # ploter.load_data(performance_file=file_name, label="DQN Agent")
    # ploter.plot(save_name, label_list=["DQN Agent"])

    # ploter.load_data("./../model/dqn/learning_rate/learning_rate_d7_e999_agent1_dqn1.p",label="d7a1q1")
    # ploter.load_data("./../model/dqn/learning_rate/learning_rate_d10_e999_agent1_dqn0.p",label="d10a1q0")
    # ploter.load_data("./../model/dqn/learning_rate/learning_rate_d10_e999_agent1_dqn1.p",label="d10a1q1")
    # ploter.plot(save_name, label_list=["d7a1q0", "d7a1q1", "d10a1q0", "d10a1q1"])


    # Draw learning curve from directory.
    path = "/Volumes/LIUQL/dataset/1100/learning_rate/"
    save_path = "/Volumes/LIUQL/dataset/1100/learning_curve/"
    no_key_word_list = ["_99.", "_199.", "_299.", "_399."]
    performance_file_list = Ploter.get_dirlist(path=path,key_word_list=["_1499"],no_key_word_list=["_99.","_199.","_299.","_399."])
    print("file_number:", len(performance_file_list))
    time.sleep(8)

    for file_name in performance_file_list:
        print(file_name)
        performance_file = path + file_name
        save_name = save_path + file_name + ".png"
        ploter = Ploter(performance_file=performance_file)
        ploter.load_data(performance_file=performance_file,label="DQN Agent")
        ploter.plot(save_name=save_name,label_list=["DQN Agent"])