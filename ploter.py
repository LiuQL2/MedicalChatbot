# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import pickle


class Ploter(object):
    def __init__(self, performance_file):
        self.performance_file = performance_file
        self.performance = pickle.load(file=open(self.performance_file, "rb"))
        self.epoch_index = []
        self.success_rate = []
        self.average_reward = []
        self.average_wrong_disease = []
        self.__prepare_data()

    def __prepare_data(self, epoch_size = 50):
        epoch_size = max(epoch_size, len(self.performance.keys()))
        for epoch_index in range(0, epoch_size, 1):
            self.epoch_index.append(epoch_index)
            self.success_rate.append(self.performance[epoch_index]["success_rate"])
            self.average_reward.append(self.performance[epoch_index]["average_reward"])
            self.average_wrong_disease.append(self.performance[epoch_index]["average_wrong_disease"])

    def plot(self, save_name):
        size = 1499
        plt.plot(self.epoch_index[0:size],self.success_rate[0:size], label="DQN Agent", linewidth=1)
        plt.xlabel("Simulation Epoch")
        plt.ylabel("Success Rate")
        plt.title("Learning Curve")
        plt.hlines(0.23,0,size,label="Rule Agent", linewidth=1, colors="purple")
        plt.hlines(0.06,0,size,label="Random Agent", linewidth=1, colors="r")
        plt.grid(True)
        # plt.legend(loc="lower right")
        plt.legend(loc='center right')
        # plt.savefig(save_name,dpi=400)

        plt.show()


if __name__ == "__main__":
    file_name = "./src/dialogue_system/model/dqn/learning_rate/learning_rate_d4_e999_agent1_dqn1.p"
    file_name = "./src/dialogue_system/data/dataset/label/result/learning_rate/learning_rate_d4_e_agent1_dqn1_T22_lr0.001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma1.0_epsilon0.1_1499.p"
    file_name = '/Users/qianlong/Desktop/acl/learning_rate_d4_e_agent1_dqn1_T22_lr0.001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma1.0_epsilon0.1_RID8_1499.p'
    save_name = file_name + ".png"
    ploter = Ploter(file_name)
    ploter.plot(save_name)