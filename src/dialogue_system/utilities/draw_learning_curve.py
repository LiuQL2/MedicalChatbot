# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import pickle


class Ploter(object):
    def __init__(self, performance_file):
        self.performance_file = performance_file
        self.epoch_size = 0
        self.success_rate = {}
        self.average_reward = {}
        self.average_wrong_disease = {}

    def load_data(self, performance_file, label):
        performance = pickle.load(file=open(performance_file, "rb"))
        self.epoch_size = max(self.epoch_size, len(performance.keys()))
        sr, ar, awd = self.__load_data(performance=performance)
        self.success_rate[label] = sr
        self.average_reward[label] = ar
        self.average_wrong_disease[label] = awd

    def __load_data(self, performance):
        success_rate = []
        average_reward = []
        average_wrong_disease = []
        for index in range(0, len(performance.keys()),1):
            success_rate.append(performance[index]["success_rate"])
            average_reward.append(performance[index]["average_reward"])
            average_wrong_disease.append(performance[index]["average_wrong_disease"])
        return success_rate, average_reward, average_wrong_disease


    def plot(self, save_name, label_list):
        for label in label_list:
            epoch_index = [i for i in range(0, len(self.success_rate[label]), 1)]
            epoch_index = [i for i in range(0, 500, 1)]
            plt.plot(epoch_index,self.success_rate[label][0:max(epoch_index)+1], label=label, linewidth=1)

        plt.xlabel("Simulation Epoch")
        plt.ylabel("Success Rate")
        plt.title("Learning Curve")
        # plt.hlines(0.15,0,400,label="Random Agent", linewidth=1, colors="r")
        # plt.hlines(0.41,0,400,label="Rule Agent", linewidth=1, colors="purple")
        if len(label_list) >= 2:
            plt.legend()
        # plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(save_name,dpi=400)

        plt.show()


if __name__ == "__main__":
    file_name = "./../model/dqn/learning_rate/learning_rate_d7_e999_agent1_dqn0.p"
    save_name = file_name + ".png"
    ploter = Ploter(file_name)
    ploter.load_data(performance_file=file_name, label="d7a1q0")
    ploter.load_data("./../model/dqn/learning_rate/learning_rate_d7_e999_agent1_dqn1.p",label="d7a1q1")
    ploter.load_data("./../model/dqn/learning_rate/learning_rate_d10_e999_agent1_dqn0.p",label="d10a1q0")
    ploter.load_data("./../model/dqn/learning_rate/learning_rate_d10_e999_agent1_dqn1.p",label="d10a1q1")
    ploter.plot(save_name, label_list=["d7a1q0", "d7a1q1", "d10a1q0", "d10a1q1"])
